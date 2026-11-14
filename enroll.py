"""
Thêm người mới bằng dòng lệnh (chạy trên Pi qua SSH, dùng cho chủ nhà đầu tiên hoặc khi cần).

LƯU Ý: khi service nhận diện đang chạy, nó giữ camera nên script này không mở được camera.
Hoặc dừng service trước:  sudo systemctl stop face-door   (xong thì: sudo systemctl start face-door)
Hoặc thêm người mới ngay từ app (khuyến nghị) mà không cần terminal.
"""
import sys
import time
import argparse

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from face_store import (DB_PATH, load_db, save_db, load_members,
                        validate_name, name_taken)

# Phải GIỐNG với code nhận diện, nếu không embedding sẽ không so khớp được
MODEL_NAME = "buffalo_l"
DET_SIZE = (320, 320)
CTX_ID = -1
MIN_FACE_SIZE = 60

CAMERA_SOURCE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def load_model():
    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)
    return app


def extract_embedding(app, img):
    """Trả về embedding của mặt lớn nhất, hoặc None nếu ảnh không dùng được."""
    faces = app.get(img)
    if len(faces) == 0:
        return None, "không thấy mặt"
    if len(faces) > 1:
        return None, "có nhiều hơn 1 mặt"
    face = faces[0]
    w = face.bbox[2] - face.bbox[0]
    h = face.bbox[3] - face.bbox[1]
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None, f"mặt quá nhỏ ({int(w)}x{int(h)}px)"
    return face.normed_embedding.astype(np.float32), None


def enroll_from_images(app, name, image_paths):
    embeddings = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Bỏ qua {path}: không đọc được file")
            continue
        emb, err = extract_embedding(app, img)
        if emb is None:
            print(f"Bỏ qua {path}: {err}")
            continue
        embeddings.append(emb)
    return embeddings


def enroll_from_camera(app, name, n_samples=8, interval=1.0):
    """Chụp trực tiếp từ camera, mỗi `interval` giây lấy 1 mẫu."""
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera. Nếu service nhận diện đang chạy, "
                           "dừng nó trước: sudo systemctl stop face-door")

    print(f"Chuẩn bị chụp {n_samples} mẫu cho '{name}'.")
    print("Hãy nhìn camera và xoay đầu nhẹ qua lại giữa các lần chụp. Bắt đầu sau 3 giây...")
    time.sleep(3)

    embeddings = []
    last_shot = 0
    try:
        while len(embeddings) < n_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            if time.time() - last_shot < interval:
                continue
            emb, err = extract_embedding(app, frame)
            if emb is None:
                print(f"  (bỏ qua: {err})")
                last_shot = time.time()
                continue
            embeddings.append(emb)
            last_shot = time.time()
            print(f"  Đã lấy mẫu {len(embeddings)}/{n_samples}")
    finally:
        cap.release()
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Thêm người mới vào models/face_database.pkl")
    parser.add_argument("name", help="Tên người cần thêm")
    parser.add_argument("--images", nargs="+", help="Danh sách ảnh (nếu không có sẽ dùng camera)")
    parser.add_argument("--samples", type=int, default=8, help="Số mẫu khi chụp bằng camera")
    parser.add_argument("--replace", action="store_true",
                        help="Ghi đè nếu tên đã tồn tại (mặc định: thêm vào embedding hiện có)")
    args = parser.parse_args()

    name, err = validate_name(args.name)
    if err:
        sys.exit(f"Tên không hợp lệ ({err}). Tối đa 30 ký tự, không chứa / \\ \" , và không trùng 'unknown'.")

    db = load_db(DB_PATH)
    members = load_members(db)
    # Cùng tên khác hoa/thường (vd. 'minh' và 'Minh') dễ gây nhầm lẫn -> chặn
    if name not in db and name_taken(members, db, name):
        sys.exit(f"Đã có người tên gần giống '{name}' (khác hoa/thường). Hãy dùng đúng tên cũ hoặc đặt tên khác.")

    app = load_model()

    if args.images:
        new_embs = enroll_from_images(app, name, args.images)
    else:
        new_embs = enroll_from_camera(app, name, n_samples=args.samples)

    if len(new_embs) < 3:
        print(f"Chỉ có {len(new_embs)} embedding hợp lệ, quá ít. Thử lại với ảnh rõ hơn. DB KHÔNG bị thay đổi.")
        sys.exit(1)

    if name in db and not args.replace:
        db[name] = list(db[name]) + new_embs
        print(f"'{name}' đã tồn tại, thêm {len(new_embs)} embedding vào dữ liệu cũ.")
    else:
        db[name] = new_embs
        print(f"Đã lưu '{name}' với {len(new_embs)} embedding.")

    save_db(db, DB_PATH)
    members = load_members(db)          # tự tạo hồ sơ cho người mới (owner nếu thuộc INITIAL_OWNERS)
    m = members[name]
    print(f"Database hiện có {len(db)} người: {list(db.keys())}")
    print(f"Hồ sơ: role={m['role']}, can_unlock={m['can_unlock']}")
    if not m["can_unlock"]:
        print("Người này CHƯA được mở cửa. Chủ nhà cấp quyền trong app (nút 'Cấp lại quyền').")
    print(f"Database: {DB_PATH}")
    print("Nếu service nhận diện đang chạy, nó tự tải lại sau vài giây (không cần restart).")


if __name__ == "__main__":
    main()
