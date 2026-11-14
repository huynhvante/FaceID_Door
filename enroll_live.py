import time
import argparse
import threading
import sys

import cv2
from flask import Flask, Response

from enroll import CAMERA_WIDTH, CAMERA_HEIGHT, load_model, extract_embedding
from face_store import (DB_PATH, load_db, save_db, load_members,
                        validate_name, name_taken)

HINTS = [
    "Nhin thang vao camera",
    "Quay dau nhe sang TRAI",
    "Quay dau nhe sang PHAI",
    "Ngang dau nhe",
    "Cui dau nhe",
    "Nhin thang, cuoi nhe",
    "Quay nhe sang TRAI",
    "Quay nhe sang PHAI",
]

# LƯU Ý: service nhận diện (face-door) đang giữ camera và cổng 5000. Dừng nó trước khi chạy file này:
#   sudo systemctl stop face-door   ...   sudo systemctl start face-door
# Hoặc thêm người mới ngay từ app (khuyến nghị) mà không cần file này.
app_web = Flask(__name__)
latest_jpeg = None
lock = threading.Lock()


def open_source(source):
    # source là số (camera USB) hoặc URL (rtsp://..., http://...)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, int):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)   # Windows
        elif sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)    # Raspberry Pi / Linux
        else:
            cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {source}")
    return cap


@app_web.route("/")
def index():
    return '<h3>Enroll - xem truc tiep</h3><img src="/video" style="max-width:100%">'


@app_web.route("/video")
def video():
    def gen():
        while True:
            with lock:
                data = latest_jpeg
            if data is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def publish(frame):
    global latest_jpeg
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with lock:
            latest_jpeg = buf.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--source", default="0", help="0 = camera USB, hoặc URL stream")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--interval", type=float, default=1.5, help="giây giữa 2 lần lấy mẫu")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    name, err = validate_name(args.name)
    if err:
        sys.exit(f"Tên không hợp lệ ({err}). Tối đa 30 ký tự, không chứa / \\ \" , và không trùng 'unknown'.")
    args.name = name

    db = load_db(DB_PATH)
    members = load_members(db)
    if name not in db and name_taken(members, db, name):
        sys.exit(f"Đã có người tên gần giống '{name}' (khác hoa/thường).")

    app = load_model()
    cap = open_source(args.source)

    threading.Thread(
        target=lambda: app_web.run(host="0.0.0.0", port=args.port, threaded=True),
        daemon=True).start()
    print(f"Mở trình duyệt: http://<IP-cua-Pi>:{args.port}")
    print("Bắt đầu sau 5 giây, hãy vào vị trí...")

    embeddings = []
    start = time.time() + 5
    last_try = 0
    last_box = None
    status = ""

    try:
        while len(embeddings) < args.samples:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            now = time.time()
            show = frame.copy()
            hint = HINTS[len(embeddings) % len(HINTS)]

            if now < start:
                status = f"Chuan bi... {start - now:.0f}s"
            elif now - last_try >= args.interval:
                last_try = now
                emb, err = extract_embedding(app, frame)
                faces = app.get(frame)
                last_box = faces[0].bbox.astype(int) if len(faces) == 1 else None
                if emb is None:
                    status = f"Bo qua: {err}"
                else:
                    embeddings.append(emb)
                    status = f"Da lay mau {len(embeddings)}/{args.samples}"
                    print(status)

            if last_box is not None:
                x1, y1, x2, y2 = last_box
                cv2.rectangle(show, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(show, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            cv2.putText(show, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            publish(show)
    finally:
        cap.release()

    if args.name in db and not args.replace:
        db[args.name] = list(db[args.name]) + embeddings
    else:
        db[args.name] = embeddings
    save_db(db, DB_PATH)
    m = load_members(db)[args.name]      # tự tạo hồ sơ cho người mới
    print(f"Đã lưu '{args.name}' ({len(embeddings)} embedding). DB có {len(db)} người.")
    print(f"Hồ sơ: role={m['role']}, can_unlock={m['can_unlock']} (cấp quyền mở cửa trong app nếu cần).")


if __name__ == "__main__":
    main()