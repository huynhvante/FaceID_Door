import os
import sys

# Chạy bằng python của venv thì site-packages đã nằm sẵn trong sys.path nên dòng này không làm gì.
_VENV_SITE = "/home/hvt/code/face_ID/venv/lib/python3.14/site-packages"
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.append(_VENV_SITE)

import csv
import json
import queue
import re
import time
import threading
import traceback
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, request
from insightface.app import FaceAnalysis
import paho.mqtt.client as mqtt

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from face_store import (
    BASE_DIR, DB_PATH, TRASH_KEEP_DAYS,
    load_db, save_db, load_members, save_members, public_list,
    validate_name, normalize_name, name_taken, ascii_name, now_iso,
    move_to_trash, purge_trash, files_signature,
)


# ============ CẤU HÌNH CHUNG ============

MODEL_NAME = "buffalo_l"               # đổi sang "buffalo_s" nếu muốn nhẹ hơn
DET_SIZE = (320, 320)
CTX_ID = -1                            # -1 = CPU

THRESHOLD = 0.8                        # Ngưỡng Cosine Similarity của InsightFace
FRAME_SKIP = 5
MIN_FACE_SIZE = 60

# Tách riêng Cooldown để cửa không mở lặp lại và không làm đầy thẻ nhớ
UNLOCK_COOLDOWN_SECONDS = 15.0         # Mở cửa xong nghỉ 15s (dài hơn 5s giữ chốt của Servo)
FAIL_COOLDOWN_SECONDS = 5.0            # Thất bại nghỉ 5s
UNKNOWN_LOG_COOLDOWN_SECONDS = 10.0    # Người lạ chỉ chụp ảnh & ghi log mỗi 10s/lần

CAMERA_SOURCE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SAVE_LOG = False                            # True = lưu log nhận diện vào file recognition_log.csv
LOG_FILE = os.path.join(BASE_DIR, "recognition_log.csv")
SAVE_SNAPSHOT = False                            # True = lưu ảnh người lạ & người được mở khoá vào thư mục snapshots/
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")

SHOW_WINDOW = False                    # False nếu chạy headless
DEBUG_FRAMES = False                   # True = in dòng [Debug] cho MỖI lần quét (rất nhiều log khi chạy service)


# ============ CẤU HÌNH MÀN HÌNH TFT ============

TFT_ENABLED = True
TFT_ROTATE = 90                        # đổi theo hướng bạn thấy đúng khi test
TFT_FPS = 15
TFT_SPEED_HZ = 24_000_000
TFT_MIRROR = True                      # True = lật gương ảnh trên TFT (chữ KHÔNG bị lật)


# ============ CẤU HÌNH LIVE STREAM ============

STREAM_ENABLED = True
STREAM_HOST = "127.0.0.1"              # An toàn qua SSH tunnel (đổi "0.0.0.0" nếu xem qua LAN)
STREAM_PORT = 5000
STREAM_FPS = 12
STREAM_JPEG_QUALITY = 70
STREAM_USER = "admin"
STREAM_PASSWORD = os.environ.get("STREAM_PASSWORD")

OVERLAY_HOLD_SECONDS = 1.5
BANNER_HOLD_SECONDS = 2.0              # Giữ dòng chữ UNLOCKED / FAIL trên màn hình 2 giây


# ============ CẤU HÌNH MEDIAPIPE (LẤY ĐIỂM MỐC MỐNG MẮT 468 & 473) ============

FACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")


# ============ CẤU HÌNH ĐỒNG TỬ (GAZE) & CHỚP MẮT (LIVENESS) ============

GAZE_HOLD_SECONDS = 2.0                # Nhìn thẳng liên tục đủ 2.0 giây
VERIFY_TIMEOUT_SECONDS = 6.0           # Quá 6.0 giây không đạt thì huỷ phiên
GAZE_GRACE_SECONDS = 0.5               # Ân hạn 0.5s để nhịp chớp mắt không làm reset bộ đếm 2s

# Ngưỡng độ lệch chuẩn hoá theo chiều rộng mắt (eye_width)
MAX_IRIS_OFFSET_X = 0.09               # Độ lệch ngang tối đa (~ ±10 độ nhìn thẳng)
MAX_IRIS_OFFSET_Y = 0.12               # Độ lệch dọc tối đa (đã chia cho eye_width nên rất ổn định)
MAX_HEAD_OFFSET_X = 0.15               # Độ lệch mũi so với 2 gò má tối đa (15%)

# Hiệu chỉnh góc đặt camera (nếu camera gắn cao/thấp hơn tầm mắt thì chỉnh GAZE_Y_OFFSET)
GAZE_X_OFFSET = 0.0
GAZE_Y_OFFSET = 0.0

# Cấu hình phát hiện chớp mắt (Nhắm rồi mở lại trong <= 0.5s)
EYE_OPEN_MIN_EAR = 0.16                # EAR < 0.16 tính là mắt đang nhắm
BLINK_MIN_DURATION = 0.04              # Nhắm tối thiểu 40ms (loại nhiễu 1 frame)
BLINK_MAX_DURATION = 0.50              # Nhắm rồi mở lại trong tối đa 0.5s mới tính là chớp mắt
REQUIRE_BLINK_LIVENESS = True          # True = Bắt buộc nhìn thẳng 2s + có 1 lần chớp mắt

RECHECK_IDENTITY_AT_END = True
DEBUG_GAZE = True


# ============ CẤU HÌNH MQTT ============

MQTT_HOST = "749e4310b50e423c8876f553789253da.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hvt"
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD")
MQTT_CLIENT_ID = "raspi-face-recognition"

TOPIC_FACE_UNLOCK = "home/livingroom/door/face_unlock"

# --- Quản lý thành viên (hợp đồng với app Android) ---
TOPIC_ENROLL_CMD = "smarthome/living_room/face/enroll/cmd"          # app -> Pi
TOPIC_ENROLL_STATUS = "smarthome/living_room/face/enroll/status"    # Pi -> app (không retained)
TOPIC_MEMBERS_CMD = "smarthome/living_room/face/members/cmd"        # app -> Pi
TOPIC_MEMBERS_STATUS = "smarthome/living_room/face/members/status"  # Pi -> app (không retained)
TOPIC_MEMBERS_LIST = "smarthome/living_room/face/members/list"      # Pi -> app (RETAINED)

mqtt_client = None
cmd_queue = queue.Queue(maxsize=50)    # MQTT thread chỉ đưa lệnh vào đây; vòng lặp chính mới xử lý


# ============ EVENT-DRIVEN: CHỈ CHẠY AI KHI CÓ CHUYỂN ĐỘNG ============
EVENT_DRIVEN = True
STANDBY_CHECK_FPS = 4          # tần số kiểm tra chuyển động (cả lúc ngủ lẫn lúc thức)
MOTION_SIZE = (160, 120)       # thu nhỏ ảnh trước khi so sánh
MOTION_PIXEL_DELTA = 25        # pixel lệch > 25/255 tính là "đổi"
MOTION_GRID = (8, 6)           # (cột, hàng) -> mỗi ô 20x20 px trên ảnh 160x120
MOTION_CELL_RATIO = 0.10       # ô được tính "có chuyển động" nếu >10% pixel trong ô đổi
MOTION_MIN_CELLS = 3           # ngưỡng dưới: cần ít nhất 3 ô (~6% khung hình)
MOTION_MAX_CELLS_FRAC = 0.8    # ngưỡng trên: >80% số ô đổi = đổi sáng toàn cục
MOTION_CONFIRM_HITS = 2        # cần 2 lần kiểm tra liên tiếp (chống nhiễu đèn/nắng)
MOTION_BG_ALPHA = 0.05         # tốc độ cập nhật nền
AWAKE_HOLD_SECONDS = 10.0      # thức thêm bao lâu sau lần thấy chuyển động/mặt cuối
MOTION_ROI = None              # vd. (0.2, 0.0, 0.8, 1.0) = chỉ xét vùng cửa (tỉ lệ 0..1)


# ============ CẤU HÌNH QUẢN LÝ THÀNH VIÊN ============

OWNER_CONFIRM_TIMEOUT = 30.0           # Chủ nhà phải xác nhận bằng mặt trong 30s
STATUS_PUBLISH_INTERVAL = 2.0          # Chu kỳ gửi trạng thái (đếm ngược / tiến độ) lên app

ENROLL_START_DELAY = 8.0               # Sau khi chủ nhà xác nhận: chờ người mới vào vị trí rồi mới chụp
ENROLL_SAMPLES = 8                     # Số mẫu mục tiêu
ENROLL_MIN_SAMPLES = 5                 # Hết giờ mà có >= số này thì vẫn lưu
ENROLL_INTERVAL = 1.5                  # Giây giữa 2 lần lấy mẫu
ENROLL_TIMEOUT = 90.0                  # Tổng thời gian thu mẫu (tính từ lúc bắt đầu chụp)
ENROLL_CONSISTENCY_MIN = 0.35          # Mẫu mới phải giống ít nhất 1 mẫu trước (chặn 2 người lẫn vào nhau)
ENROLL_DUP_ABORT = 3                   # Số mẫu liên tiếp khớp người đã có thì huỷ (duplicate_person)
ENROLL_PER_HOUR = 3                    # Tối đa số lần BẮT ĐẦU enroll mỗi giờ
CMDS_PER_MINUTE = 5                    # Tối đa số lệnh quản lý mỗi phút
RELOAD_CHECK_SECONDS = 5.0             # Kiểm tra database/members.json bị sửa từ bên ngoài
TRASH_PURGE_SECONDS = 3600.0

ENROLL_HINTS = [
    "Nhin thang vao camera",
    "Quay dau nhe sang TRAI",
    "Quay dau nhe sang PHAI",
    "Ngang dau nhe",
    "Cui dau nhe",
    "Nhin thang, cuoi nhe",
    "Quay nhe sang TRAI",
    "Quay nhe sang PHAI",
]

COLOR_OK = (0, 200, 0)
COLOR_BAD = (0, 0, 255)
COLOR_INFO = (0, 200, 200)
COLOR_WARN = (0, 200, 255)

FAIL_BANNERS = {
    "IDENTITY_MISMATCH": "MISMATCH!",
    "NO_BLINK": "NO BLINK!",
    "GAZE_TIMEOUT": "GAZE TIMEOUT!",
}


# ============ LIVE STREAM (Flask MJPEG) ============

app_web = Flask(__name__)
latest_jpeg = None
jpeg_lock = threading.Lock()
_last_publish = 0.0


def _check_stream_auth():
    if not STREAM_PASSWORD:
        return True
    auth = request.authorization
    return bool(auth and auth.username == STREAM_USER and auth.password == STREAM_PASSWORD)


@app_web.route("/")
def index():
    if not _check_stream_auth():
        return Response("Yêu cầu đăng nhập", 401, {"WWW-Authenticate": 'Basic realm="DoorCam"'})
    return '<h3>Door camera - live</h3><img src="/video" style="max-width:100%">'


@app_web.route("/video")
def video():
    if not _check_stream_auth():
        return Response("Yêu cầu đăng nhập", 401, {"WWW-Authenticate": 'Basic realm="DoorCam"'})

    def gen():
        while True:
            with jpeg_lock:
                data = latest_jpeg
            if data is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            time.sleep(1.0 / STREAM_FPS)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def publish(frame):
    global latest_jpeg, _last_publish
    now = time.time()
    if now - _last_publish < 1.0 / STREAM_FPS:
        return
    _last_publish = now
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
    if ok:
        with jpeg_lock:
            latest_jpeg = buf.tobytes()


def start_stream_server():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    threading.Thread(
        target=lambda: app_web.run(host=STREAM_HOST, port=STREAM_PORT, threaded=True),
        daemon=True).start()
    print(f"[Stream] Mở trình duyệt: http://{STREAM_HOST}:{STREAM_PORT}")


# ============ MQTT ============

def _on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT] Đã kết nối broker, rc={rc}")
    client.subscribe([(TOPIC_ENROLL_CMD, 1), (TOPIC_MEMBERS_CMD, 1)])
    try:
        cmd_queue.put_nowait(("_connected", {}))   # vòng lặp chính sẽ gửi lại danh sách thành viên (retained)
    except queue.Full:
        pass


def _on_message(client, userdata, msg):
    """Chạy trong thread của paho: KHÔNG đổi trạng thái ở đây, chỉ phân tích rồi đưa vào hàng đợi."""
    if msg.retain:
        # Lệnh quản lý không bao giờ hợp lệ nếu là tin retained (tránh bị phát lại khi Pi khởi động)
        print(f"[MQTT] Bỏ qua tin retained trên {msg.topic}")
        return
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        print(f"[MQTT] Payload không phải JSON trên {msg.topic}, bỏ qua.")
        return
    if not isinstance(data, dict):
        return
    try:
        cmd_queue.put_nowait((msg.topic, data))
    except queue.Full:
        print("[MQTT] Hàng đợi lệnh đầy, bỏ lệnh.")


def setup_mqtt():
    global mqtt_client

    if not MQTT_PASSWORD:
        print("[WARN] MQTT_PASSWORD chưa được đặt (export MQTT_PASSWORD='...'). "
              "Sẽ không mở khoá được qua MQTT.")

    if hasattr(mqtt, "CallbackAPIVersion"):
        mqtt_client = mqtt.Client(
            client_id=MQTT_CLIENT_ID,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
    else:
        mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)

    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    mqtt_client.tls_set()
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
    mqtt_client.on_connect = _on_connect
    mqtt_client.on_message = _on_message
    mqtt_client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()
    print("[MQTT] Đang kết nối tới broker (async)...")


def cleanup_mqtt():
    global mqtt_client
    if mqtt_client is None:
        return
    try:
        mqtt_client.disconnect()
        mqtt_client.loop_stop()
        print("[MQTT] Đã ngắt kết nối.")
    except Exception:
        pass
    mqtt_client = None


def pub_json(topic, payload, retain=False):
    if mqtt_client is None:
        return
    try:
        mqtt_client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1, retain=retain)
    except Exception as e:
        print(f"[WARN] MQTT publish lỗi: {e}")


def _clean(d):
    return {k: v for k, v in d.items() if v is not None}


def enroll_status(state, name=None, **extra):
    pub_json(TOPIC_ENROLL_STATUS, _clean({"state": state, "name": name, **extra}))


def member_status(req_id, action, name, state, **extra):
    pub_json(TOPIC_MEMBERS_STATUS, _clean({"req_id": req_id, "action": action, "name": name,
                                           "state": state, **extra}))


# ============ 1. MODEL (INSIGHTFACE + MEDIAPIPE IRIS) ============

def load_model():
    print(f"[Model] Đang load '{MODEL_NAME}' (chỉ lấy detection + recognition)...")
    app = FaceAnalysis(
        name=MODEL_NAME,
        allowed_modules=["detection", "recognition"]
    )
    app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)
    print("[Model] InsightFace sẵn sàng.")
    return app


class FaceLandmarkerWrapper:
    """Trích xuất 478 điểm mốc khuôn mặt & mống mắt (điểm 468, 473) bằng MediaPipe."""

    def __init__(self, model_path):
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.mesh = mp_vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        res = self.mesh.detect_for_video(mp_image, ts_ms)
        if not res.face_landmarks:
            return None
        return res.face_landmarks[0]

    def close(self):
        self.mesh.close()


def _ensure_face_model(path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[FaceMesh] Chưa có sẵn file model tại '{path}', đang tải về (cần Internet lần đầu)...")
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, path)
        print("[FaceMesh] Tải xong.")
    except Exception as e:
        raise RuntimeError(
            f"Không tải được '{path}'. Hãy tải sẵn file face_landmarker.task bỏ vào thư mục models/. Lỗi: {e}"
        )


def load_face_mesh():
    _ensure_face_model(FACE_MODEL_PATH)
    landmarker = FaceLandmarkerWrapper(FACE_MODEL_PATH)
    print("[FaceMesh] Sẵn sàng (có điểm mốc mống mắt 468 & 473).")
    return landmarker


# ============ 2. HÀM SO KHỚP DANH TÍNH ============

def recognize(query_emb: np.ndarray, db: dict, threshold: float):
    best_name = None
    best_score = -1.0

    for name, emb_matrix in db.items():
        scores = emb_matrix @ query_emb
        top_score = float(np.max(scores))
        if top_score > best_score:
            best_score = top_score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


# ============ 2b. THEO DÕI MỐNG MẮT & ĐỒNG TỬ (CHUẨN HÓA THEO EYE_WIDTH) ============

# Chỉ số các điểm mốc vùng mắt của MediaPipe (468 = tâm mống mắt phải, 473 = tâm mống mắt trái)
EYE_RIGHT = dict(left=33, right=133, up=159, low=145, iris=468)
EYE_LEFT = dict(left=362, right=263, up=386, low=374, iris=473)
NOSE_TIP, CHEEK_L, CHEEK_R = 1, 234, 454


def _get_xy(lm, idx, w, h):
    """Đổi tọa độ chuẩn hóa (0..1) sang tọa độ pixel (x, y) trên ảnh."""
    return lm[idx].x * w, lm[idx].y * h


def _single_eye_gaze(lm, w, h, eye_idx):
    """
    Tính độ lệch giữa Tâm đồng tử P_pupil(x_p, y_p) và Tâm hốc mắt C_eye(x_c, y_c)
    theo công thức g = [x_p - x_c, y_p - y_c]^T, chuẩn hóa toàn bộ theo eye_width.
    """
    # 1. Tọa độ 2 khóe mắt (cố định, ít bị ảnh hưởng khi chớp mắt) và 2 mí mắt
    x_left, y_left = _get_xy(lm, eye_idx["left"], w, h)
    x_right, y_right = _get_xy(lm, eye_idx["right"], w, h)
    x_up, y_up = _get_xy(lm, eye_idx["up"], w, h)
    x_low, y_low = _get_xy(lm, eye_idx["low"], w, h)

    # 2. Tọa độ tâm đồng tử / mống mắt P_pupil(x_p, y_p) (điểm 468 hoặc 473)
    x_p, y_p = _get_xy(lm, eye_idx["iris"], w, h)

    # Chiều rộng mắt (khoảng cách giữa 2 khóe mắt) và chiều cao mở mắt
    eye_width = float(np.hypot(x_right - x_left, y_right - y_left))
    eye_height = float(np.hypot(x_low - x_up, y_low - y_up))
    if eye_width < 5.0:
        return None

    # 3. Tâm hốc mắt C_eye(x_c, y_c) tính từ trung điểm 2 khóe mắt để không bị trôi khi mí mắt nheo
    x_c = (x_left + x_right) / 2.0
    y_c = (y_left + y_right) / 2.0

    # 4. Vector độ lệch hướng nhìn đồng tử chuẩn hóa theo eye_width
    g_x = (x_p - x_c) / eye_width
    g_y = (y_p - y_c) / eye_width

    # 5. Tỷ lệ độ mở mắt EAR (Cao / Rộng) để phát hiện chớp mắt
    ear = eye_height / eye_width

    return g_x, g_y, ear, (int(x_p), int(y_p))


def analyze_gaze(face_mesh, frame):
    """Phân tích đồng thời cả 2 mắt và trả về trạng thái nhìn thẳng."""
    h, w = frame.shape[:2]
    lm = face_mesh.process(frame)
    if lm is None:
        return {"ok": False, "reason": "no_face"}

    right_res = _single_eye_gaze(lm, w, h, EYE_RIGHT)
    left_res = _single_eye_gaze(lm, w, h, EYE_LEFT)
    if right_res is None or left_res is None:
        return {"ok": False, "reason": "bad_eye"}

    # Trung bình cộng độ lệch đồng tử của 2 mắt (có trừ đi hiệu chỉnh góc đặt camera)
    gx = ((right_res[0] + left_res[0]) / 2.0) - GAZE_X_OFFSET
    gy = ((right_res[1] + left_res[1]) / 2.0) - GAZE_Y_OFFSET
    ear = (right_res[2] + left_res[2]) / 2.0
    iris_pts = [right_res[3], left_res[3]]

    # Kiểm tra đầu có quay ngang quá nhiều không (so sánh mũi với trung điểm 2 gò má)
    x_nose, y_nose = _get_xy(lm, NOSE_TIP, w, h)
    x_cheek_l, _ = _get_xy(lm, CHEEK_L, w, h)
    x_cheek_r, _ = _get_xy(lm, CHEEK_R, w, h)
    face_width = abs(x_cheek_r - x_cheek_l)
    head_offset = (x_nose - (x_cheek_l + x_cheek_r) / 2.0) / max(face_width, 10.0)

    is_eyes_closed = (ear < EYE_OPEN_MIN_EAR)

    if is_eyes_closed:
        reason = "eyes_closed"
    elif abs(head_offset) > MAX_HEAD_OFFSET_X:
        reason = "head_turned"
    elif abs(gx) > MAX_IRIS_OFFSET_X or abs(gy) > MAX_IRIS_OFFSET_Y:
        reason = "looking_away"
    else:
        reason = "ok"

    return {
        "ok": reason == "ok",
        "reason": reason,
        "is_eyes_closed": is_eyes_closed,
        "gx": gx,
        "gy": gy,
        "ear": ear,
        "head_off": head_offset,
        "iris_pts": iris_pts,
        "nose_xy": (x_nose, y_nose),  # Dùng để đối chiếu cùng 1 khuôn mặt với InsightFace
    }


def confirm_identity(app, frame, expected_name, db, expected_point=None):
    """
    Kiểm tra lại danh tính bằng InsightFace ở cuối phiên 2 giây.
    Nếu có truyền expected_point (tọa độ mũi từ MediaPipe), bắt buộc mặt được chọn
    phải chứa điểm mũi đó để đảm bảo MediaPipe và InsightFace đang xét cùng 1 người.
    """
    faces = app.get(frame)
    if not faces:
        return False, -1.0

    target_face = None
    if expected_point is not None:
        px, py = expected_point
        containing_faces = [
            f for f in faces
            if f.bbox[0] <= px <= f.bbox[2] and f.bbox[1] <= py <= f.bbox[3]
        ]
        if not containing_faces:
            # Khuôn mặt đang theo dõi đồng tử không khớp vị trí với khuôn mặt của InsightFace
            return False, -1.0
        target_face = max(containing_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    else:
        target_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    name, score = recognize(target_face.normed_embedding, db, THRESHOLD)
    return (name == expected_name), score


# ============ 3. LOG + SNAPSHOT + NHẬT KÝ KIỂM TOÁN ============

def init_logging():
    if SAVE_SNAPSHOT and not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "name", "score", "snapshot_path", "status"])


def log_event(name, score: float, frame=None, status: str = ""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_path = ""
    if SAVE_SNAPSHOT and frame is not None:
        safe_name = name if name else "unknown"
        snapshot_path = os.path.join(
            SNAPSHOT_DIR, f"{safe_name}_{timestamp.replace(':', '-')}.jpg"
        )
        cv2.imwrite(snapshot_path, frame)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name or "unknown", f"{score:.3f}", snapshot_path, status])


def audit(status, name="", req_id="", by=""):
    """Ghi MỌI lệnh quản lý (kể cả bị từ chối) vào recognition_log.csv, không phụ thuộc cờ nào."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note = status + (f"|req={req_id}" if req_id else "") + (f"|by={by}" if by else "")
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timestamp, name or "-", "", "", note])
    except Exception as e:
        print(f"[WARN] Ghi nhật ký kiểm toán lỗi: {e}")


def purge_snapshots(name):
    """Xoá ảnh snapshot của người bị xoá (tên file: <tên>_YYYY-MM-DD HH-MM-SS.jpg)."""
    if not os.path.isdir(SNAPSHOT_DIR):
        return 0
    pat = re.compile(r"^" + re.escape(name) + r"_\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}\.jpg$")
    n = 0
    for f in os.listdir(SNAPSHOT_DIR):
        if pat.match(f):
            try:
                os.remove(os.path.join(SNAPSHOT_DIR, f))
                n += 1
            except OSError:
                pass
    return n


# ============ 4. HÀNH ĐỘNG (KIỂM TRA KẾT NỐI MQTT TRƯỚC KHI PUBLISH) ============

def send_unlock(name: str) -> bool:
    if mqtt_client is None or not mqtt_client.is_connected():
        print("  -> [WARN] MQTT chưa kết nối tới Broker, huỷ gửi lệnh mở khoá để tránh gửi trễ!")
        return False
    mqtt_client.publish(TOPIC_FACE_UNLOCK, "UNLOCK", qos=1)
    print(f"  -> [MQTT] Đã gửi lệnh mở khoá cho {name} lên '{TOPIC_FACE_UNLOCK}'")
    return True


# ============ 5. VẼ LỚP PHỦ (KHUNG, ĐỒNG TỬ, CHỮ) ============

def render_overlay(img, ops, mirror=False):
    """Vẽ lớp phủ (khung, đồng tử, chữ) lên ảnh.
    mirror=True: chỉ lật ảnh camera + đổi tọa độ x của khung/chấm, chữ vẫn vẽ xuôi."""
    W = img.shape[1]
    if mirror:
        img = cv2.flip(img, 1)          # tạo mảng mới, ảnh gốc không bị sửa
    for op in ops:
        kind = op[0]
        if kind == "circle":
            _, (x, y), r, color = op
            if mirror:
                x = W - 1 - x
            cv2.circle(img, (x, y), r, color, -1)
        elif kind == "box":
            _, x1, y1, x2, y2, color, label = op
            if mirror:
                x1, x2 = W - 1 - x2, W - 1 - x1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif kind == "text":
            _, text, org, scale, color = op
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return img


# ============ 6. BỘ ĐIỀU KHIỂN CỬA + QUẢN LÝ THÀNH VIÊN ============

def new_verification(name, score, purpose="unlock"):
    """purpose: 'unlock' = mở cửa; 'owner_confirm' = chủ nhà xác nhận một lệnh quản lý."""
    now = time.time()
    return {
        "name": name,
        "score": score,
        "purpose": purpose,
        "t_start": now,
        "last_t": now,
        "hold": 0.0,
        "fail_since": None,
        "closed_since": None,                      # Thời điểm bắt đầu nhắm mắt
        "blinked": not REQUIRE_BLINK_LIVENESS,     # True khi hoàn tất chu trình Nhắm -> Mở (<= 0.5s)
        "n": 0,
    }


class MotionGate:
    """Phát hiện chuyển động rẻ tiền; chỉ để quyết định có bật AI nặng hay không."""

    def __init__(self):
        self.bg = None
        self.hits = 0
        self.active_cells = 0
        self.awake_until = 0.0
        self.last_check = 0.0

    def _prep(self, frame):
        small = cv2.resize(frame, MOTION_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if MOTION_ROI:
            W, H = MOTION_SIZE
            x1, y1, x2, y2 = MOTION_ROI
            gray = gray[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)]
        g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        return g - g.mean()          # bỏ chênh lệch sáng toàn cục (đèn bật/tắt, auto-exposure)

    def check(self, frame, now):
        self.last_check = now
        g = self._prep(frame)
        if self.bg is None:
            self.bg = g
            return False

        changed = (cv2.absdiff(g, self.bg) > MOTION_PIXEL_DELTA).astype(np.float32)
        cols, rows = MOTION_GRID
        cells = cv2.resize(changed, (cols, rows), interpolation=cv2.INTER_AREA)  # mỗi ô = tỉ lệ pixel đổi
        self.active_cells = int(np.count_nonzero(cells >= MOTION_CELL_RATIO))
        total = cols * rows

        # Gần như mọi ô cùng đổi -> đèn bật/tắt, auto-exposure: không đánh thức, nền theo kịp ngay
        if self.active_cells > total * MOTION_MAX_CELLS_FRAC:
            self.bg = g
            self.hits = 0
            return False

        cv2.accumulateWeighted(g, self.bg, MOTION_BG_ALPHA)
        self.hits = self.hits + 1 if self.active_cells >= MOTION_MIN_CELLS else 0
        if self.hits >= MOTION_CONFIRM_HITS:
            self.awake_until = now + AWAKE_HOLD_SECONDS
            return True
        return False

    def extend(self, now):
        self.awake_until = max(self.awake_until, now + AWAKE_HOLD_SECONDS)


class DoorController:
    """
    Trạng thái:
      IDLE      : quét mặt; nhận ra người có quyền thì sang VERIFYING (unlock)
      VERIFYING : nhìn thẳng 2s + chớp mắt (dùng cho mở cửa HOẶC chủ nhà xác nhận lệnh)
      ENROLLING : đang thu mẫu người mới (KHÔNG mở cửa trong lúc này)

    Nguyên tắc: lệnh làm GIẢM quyền (revoke) chạy ngay; lệnh làm TĂNG quyền hoặc không đảo ngược
    (enroll, grant, remove) phải được một OWNER xác nhận bằng mặt + nhìn thẳng + chớp mắt tại cửa.
    Mọi thay đổi trạng thái chạy trong luồng chính; luồng MQTT chỉ xếp lệnh vào hàng đợi.
    """

    def __init__(self, app, face_mesh, db, members):
        self.app = app
        self.face_mesh = face_mesh
        self.db = db
        self.members = members

        self.state = "IDLE"
        self.verify = None
        self.pending = None            # {"kind","name","req_id","deadline"}: lệnh chờ chủ nhà xác nhận
        self.enroll = None

        self.cooldown_until = {}
        self.last_unknown_log_t = 0.0
        self.banner_info = None
        self.overlay_items, self.overlay_t = [], 0.0
        self.frame_count = 0

        self.seen_req = deque(maxlen=50)
        self.cmd_times = deque()
        self.enroll_starts = deque()
        self.files_sig = files_signature()
        self.last_reload_check = 0.0
        self.last_purge = 0.0
        self.last_status_pub = 0.0

    def is_busy(self):
        """Đang xác thực / enroll / chờ chủ nhà xác nhận -> không được ngủ."""
        return self.state != "IDLE" or self.pending is not None

    # ---------- tiện ích ----------
    def can_unlock(self, name):
        m = self.members.get(name)
        return bool(m and m.get("can_unlock") is True)

    def is_owner(self, name):
        m = self.members.get(name)
        return bool(m and m.get("role") == "owner")

    def _banner(self, text, color, now):
        self.banner_info = (text, color, now + BANNER_HOLD_SECONDS)

    def publish_members_list(self):
        pub_json(TOPIC_MEMBERS_LIST, {"members": public_list(self.members)}, retain=True)

    def _persist_members(self):
        save_members(self.members)
        self.files_sig = files_signature()

    def _persist_db(self):
        save_db(self.db)
        self.files_sig = files_signature()

    def _rate_ok(self):
        now = time.time()
        while self.cmd_times and now - self.cmd_times[0] > 60:
            self.cmd_times.popleft()
        if len(self.cmd_times) >= CMDS_PER_MINUTE:
            return False
        self.cmd_times.append(now)
        return True

    def _enroll_hour_ok(self):
        now = time.time()
        while self.enroll_starts and now - self.enroll_starts[0] > 3600:
            self.enroll_starts.popleft()
        return len(self.enroll_starts) < ENROLL_PER_HOUR

    def _verifying_owner_confirm(self):
        return bool(self.state == "VERIFYING" and self.verify and self.verify["purpose"] == "owner_confirm")

    def _reset_owner_verify(self):
        if self._verifying_owner_confirm():
            self.state, self.verify = "IDLE", None

    # ---------- xử lý lệnh từ app ----------
    def handle_command(self, topic, data):
        if topic == "_connected":
            self.publish_members_list()
        elif topic == TOPIC_ENROLL_CMD:
            self._cmd_enroll(data)
        elif topic == TOPIC_MEMBERS_CMD:
            self._cmd_members(data)

    def _cmd_enroll(self, data):
        action = data.get("action")
        if action == "cancel":
            self._cancel_enroll()
            return
        if action != "start":
            print(f"[Enroll] Lệnh không hợp lệ: {data}")
            return

        raw_name = data.get("name")
        shown = raw_name if isinstance(raw_name, str) else None
        name, err = validate_name(raw_name)

        # Trùng lệnh start cho đúng phiên đang chạy (MQTT QoS1 gửi lại): bỏ qua, không báo lỗi
        if name:
            if self.pending and self.pending["kind"] == "enroll" and self.pending["name"] == name:
                return
            if self.state == "ENROLLING" and self.enroll and self.enroll["name"] == name:
                return

        if not self._rate_ok():
            enroll_status("failed", shown, reason="rate_limited")
            audit("ENROLL_REJECTED:rate_limited", shown or "")
            return
        if err or name_taken(self.members, self.db, name):
            enroll_status("failed", shown, reason="name_invalid")
            audit(f"ENROLL_REJECTED:name_invalid:{err or 'taken'}", shown or "")
            return
        if self.state != "IDLE" or self.pending is not None:
            enroll_status("failed", name, reason="busy")
            audit("ENROLL_REJECTED:busy", name)
            return
        if not self._enroll_hour_ok():
            enroll_status("failed", name, reason="rate_limited")
            audit("ENROLL_REJECTED:rate_limited_hour", name)
            return
        self.enroll_starts.append(time.time())
        self._begin_pending("enroll", name, "")

    def _cmd_members(self, data):
        req_id = data.get("req_id")
        action = data.get("action")
        raw_name = data.get("name")
        valid_req = isinstance(req_id, str) and 0 < len(req_id) <= 64
        rid = req_id if valid_req else None
        act = action if isinstance(action, str) else None
        shown = raw_name if isinstance(raw_name, str) else None

        def fail(reason):
            member_status(rid, act, shown, "failed", reason=reason)
            audit(f"MEMBER_{(act or 'UNKNOWN').upper()}_REJECTED:{reason}", shown or "", rid or "")

        # Lệnh trùng req_id (QoS1 gửi lại) bị bỏ qua im lặng để không ghi đè trạng thái đang hiển thị trên app
        if valid_req and req_id in self.seen_req:
            print(f"[Members] Bỏ qua lệnh trùng req_id={req_id}")
            return
        if not self._rate_ok():
            fail("rate_limited")
            return
        if not valid_req or act not in ("revoke", "grant", "remove") or shown is None:
            fail("invalid_request")
            return
        self.seen_req.append(req_id)

        name = normalize_name(raw_name)
        m = self.members.get(name)
        if m is None:
            fail("not_found")
            return
        if m["role"] == "owner":
            fail("owner_protected")      # chủ nhà không bị thu hồi/xoá qua MQTT, chỉ làm trên Pi
            return

        if act == "revoke":
            self._do_revoke(req_id, name)
            return
        if act == "grant" and m["can_unlock"]:
            member_status(req_id, act, name, "done")   # đã có quyền sẵn: idempotent
            return
        if self.state != "IDLE" or self.pending is not None:
            fail("busy")
            return
        self._begin_pending(act, name, req_id)

    # ---------- lệnh chờ chủ nhà xác nhận ----------
    def _begin_pending(self, kind, name, req_id):
        now = time.time()
        self.pending = {"kind": kind, "name": name, "req_id": req_id,
                        "deadline": now + OWNER_CONFIRM_TIMEOUT}
        print(f"[Members] Lệnh '{kind}' cho '{name}' đang chờ chủ nhà xác nhận ({OWNER_CONFIRM_TIMEOUT:.0f}s).")
        audit(f"MEMBER_{kind.upper()}_REQUEST" if kind != "enroll" else "ENROLL_REQUEST", name, req_id)
        self._publish_pending_status(self.pending, now)

    def _publish_pending_status(self, p, now):
        self.last_status_pub = now
        remaining = max(0, int(round(p["deadline"] - now)))
        if p["kind"] == "enroll":
            enroll_status("waiting_owner", p["name"], remaining_s=remaining)
        else:
            member_status(p["req_id"], p["kind"], p["name"], "waiting_owner", remaining_s=remaining)

    def _pending_fail(self, p, reason):
        if p["kind"] == "enroll":
            self._fail_enroll(p["name"], reason)
        else:
            member_status(p["req_id"], p["kind"], p["name"], "failed", reason=reason)
            audit(f"MEMBER_{p['kind'].upper()}_FAIL:{reason}", p["name"], p["req_id"])

    def _pending_accepts(self, name):
        p = self.pending
        if p is None or not self.is_owner(name):
            return False
        if p["kind"] in ("grant", "remove") and p["name"] == name:
            return False                 # người xác nhận không được là đối tượng của lệnh
        return True

    def _expire_pending(self, now):
        p = self.pending
        if p is None:
            return
        if now > p["deadline"] and not self._verifying_owner_confirm():
            self.pending = None
            print(f"[Members] Lệnh '{p['kind']}' hết hạn chờ chủ nhà.")
            self._pending_fail(p, "owner_timeout")
        elif now - self.last_status_pub >= STATUS_PUBLISH_INTERVAL:
            self._publish_pending_status(p, now)

    def _complete_pending(self, verifier, now):
        p, self.pending = self.pending, None
        if p is None:
            return
        if p["kind"] == "enroll":
            self._start_enrolling(p["name"], verifier, now)
        elif p["kind"] == "grant":
            self._do_grant(p, verifier)
        elif p["kind"] == "remove":
            self._do_remove(p, verifier)

    # ---------- revoke / grant / remove ----------
    def _do_revoke(self, req_id, name):
        m = self.members[name]
        m["can_unlock"] = False          # hiệu lực NGAY trong bộ nhớ, kể cả khi ghi đĩa lỗi (fail-safe)
        persist_ok = True
        try:
            self._persist_members()
        except Exception as e:
            persist_ok = False
            print(f"[CẢNH BÁO] Đã thu hồi '{name}' trong RAM nhưng KHÔNG ghi được đĩa: {e}")

        now = time.time()
        # Huỷ phiên mở cửa đang chạy của chính người này
        if (self.state == "VERIFYING" and self.verify and self.verify["name"] == name
                and self.verify["purpose"] == "unlock"):
            self.state, self.verify = "IDLE", None
            self._banner(f"{ascii_name(name)}: REVOKED", COLOR_BAD, now)
        # Huỷ lệnh grant/remove đang chờ cho chính người này
        p = self.pending
        if p and p["kind"] in ("grant", "remove") and p["name"] == name:
            self.pending = None
            self._reset_owner_verify()
            self._pending_fail(p, "cancelled")

        audit("MEMBER_REVOKE_OK" if persist_ok else "MEMBER_REVOKE_OK_NOT_PERSISTED", name, req_id)
        self.publish_members_list()
        if persist_ok:
            member_status(req_id, "revoke", name, "done")
        else:
            member_status(req_id, "revoke", name, "failed", reason="unknown_error")
        print(f"[Members] Đã thu hồi quyền mở cửa của '{name}'.")

    def _do_grant(self, p, by):
        name, req_id = p["name"], p["req_id"]
        m = self.members.get(name)
        if m is None:
            self._pending_fail(p, "not_found")
            return
        if m["role"] == "owner":
            self._pending_fail(p, "owner_protected")
            return
        m["can_unlock"] = True
        try:
            self._persist_members()
        except Exception as e:
            m["can_unlock"] = False
            print(f"[Members] Ghi members.json lỗi, huỷ cấp quyền: {e}")
            self._pending_fail(p, "unknown_error")
            return
        audit("MEMBER_GRANT_OK", name, req_id, by)
        self.publish_members_list()
        member_status(req_id, "grant", name, "done")
        print(f"[Members] '{by}' đã xác nhận cấp quyền mở cửa cho '{name}'.")

    def _do_remove(self, p, by):
        name, req_id = p["name"], p["req_id"]
        m = self.members.get(name)
        if m is None:
            self._pending_fail(p, "not_found")
            return
        if m["role"] == "owner":
            self._pending_fail(p, "owner_protected")
            return

        emb = self.db.get(name)
        try:
            if TRASH_KEEP_DAYS > 0 and emb is not None:
                path = move_to_trash(name, emb, m)       # xoá mềm: giữ bản phục hồi trên Pi
                print(f"[Members] Đã chuyển embedding của '{name}' vào {path}")
            del self.members[name]
            self.db.pop(name, None)
            self._persist_db()
        except Exception as e:
            self.members[name] = m                       # hoàn tác
            if emb is not None:
                self.db[name] = emb
            print(f"[Members] Xoá '{name}' lỗi, đã hoàn tác: {e}")
            self._pending_fail(p, "unknown_error")
            return
        try:
            self._persist_members()
        except Exception as e:
            # Khuôn mặt đã bị xoá khỏi database; hồ sơ thừa sẽ được dọn khi tải lại
            print(f"[WARN] Ghi members.json lỗi sau khi xoá '{name}': {e}")

        self.cooldown_until.pop(name, None)
        n_snap = purge_snapshots(name)
        audit("MEMBER_REMOVE_OK", name, req_id, by)
        self.publish_members_list()
        member_status(req_id, "remove", name, "done")
        print(f"[Members] '{by}' đã xác nhận xoá '{name}' (xoá {n_snap} ảnh snapshot).")

    # ---------- enroll ----------
    def _fail_enroll(self, name, reason):
        enroll_status("failed", name, reason=reason)
        audit(f"ENROLL_FAIL:{reason}", name or "")
        print(f"[Enroll] Thất bại ({reason}) cho '{name}'.")

    def _cancel_enroll(self):
        if self.state == "ENROLLING" and self.enroll:
            name = self.enroll["name"]
            self.enroll, self.state = None, "IDLE"
            self._fail_enroll(name, "cancelled")
        elif self.pending and self.pending["kind"] == "enroll":
            p, self.pending = self.pending, None
            self._reset_owner_verify()
            self._fail_enroll(p["name"], "cancelled")
        else:
            enroll_status("idle")

    def _start_enrolling(self, name, verifier, now):
        self.state = "ENROLLING"
        self.enroll = {
            "name": name, "samples": [], "dups": 0, "last_try": 0.0, "last_box": None,
            "t_capture": now + ENROLL_START_DELAY,
            "deadline": now + ENROLL_START_DELAY + ENROLL_TIMEOUT,
            "last_progress": -1, "last_pub": 0.0,
        }
        audit("ENROLL_STARTED", name, "", verifier)
        print(f"[Enroll] '{verifier}' đã xác nhận. Người mới '{name}' vào vị trí, chụp sau {ENROLL_START_DELAY:.0f}s...")

    def _try_sample(self, e, faces):
        """Trả về (đã_nhận_mẫu, thông_báo, lý_do_huỷ)."""
        e["last_box"] = None
        if len(faces) == 0:
            return False, "khong thay mat", None
        if len(faces) > 1:
            return False, "co nhieu hon 1 mat", None
        face = faces[0]
        e["last_box"] = face.bbox.astype(int)
        w = face.bbox[2] - face.bbox[0]
        h = face.bbox[3] - face.bbox[1]
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            return False, f"mat qua nho ({int(w)}x{int(h)}px)", None

        emb = face.normed_embedding.astype(np.float32)
        who, _ = recognize(emb, self.db, THRESHOLD)
        if who is not None:
            e["dups"] += 1
            if e["dups"] >= ENROLL_DUP_ABORT:
                return False, "da co trong he thong", "duplicate_person"
            return False, "khuon mat da co trong he thong", None
        e["dups"] = 0

        if e["samples"]:
            sims = np.array(e["samples"]) @ emb
            if float(sims.max()) < ENROLL_CONSISTENCY_MIN:
                return False, "khac nguoi truoc, chi 1 nguoi truoc camera", None
        e["samples"].append(emb)
        return True, None, None

    def _finish_enroll(self, now):
        e = self.enroll
        name, samples = e["name"], e["samples"]
        n = len(samples)
        self.enroll, self.state = None, "IDLE"

        if n < ENROLL_MIN_SAMPLES:
            self._fail_enroll(name, "capture_timeout" if n == 0 else "too_few_samples")
            self._banner("ENROLL FAILED", COLOR_BAD, now)
            return

        self.db[name] = np.array(samples, dtype=np.float32)
        self.members[name] = {"role": "member", "can_unlock": False, "created": now_iso()}
        try:
            self._persist_db()
            self._persist_members()
        except Exception as ex:
            self.db.pop(name, None)
            self.members.pop(name, None)
            print(f"[Enroll] Lưu lỗi, đã hoàn tác: {ex}")
            self._fail_enroll(name, "unknown_error")
            return

        audit("ENROLL_OK", name)
        enroll_status("done", name)
        self.publish_members_list()
        self._banner(f"{ascii_name(name)}: ENROLLED", COLOR_OK, now)
        print(f"[Enroll] Đã lưu '{name}' ({n} mẫu). Mặc định CHƯA được mở cửa; chủ nhà cấp quyền trong app.")

    def _tick_enrolling(self, frame, now, ops):
        e = self.enroll
        name, disp = e["name"], ascii_name(e["name"])
        n = len(e["samples"])

        if now < e["t_capture"]:
            hint = f"Nguoi moi dung truoc camera ({int(e['t_capture'] - now) + 1}s)"
        else:
            if now >= e["deadline"]:
                self._finish_enroll(now)
                return
            hint = ENROLL_HINTS[n % len(ENROLL_HINTS)]
            if now - e["last_try"] >= ENROLL_INTERVAL:
                e["last_try"] = now
                faces = self.app.get(frame)
                ok, msg, abort = self._try_sample(e, faces)
                if abort:
                    self.enroll, self.state = None, "IDLE"
                    self._fail_enroll(name, abort)
                    self._banner("DA CO TRONG HE THONG", COLOR_BAD, now)
                    return
                if ok:
                    n = len(e["samples"])
                    print(f"[Enroll] Đã lấy mẫu {n}/{ENROLL_SAMPLES}")
                    if n >= ENROLL_SAMPLES:
                        self._finish_enroll(now)
                        return
                else:
                    hint = f"Bo qua: {msg}"

        if n != e["last_progress"] or now - e["last_pub"] >= STATUS_PUBLISH_INTERVAL:
            e["last_progress"], e["last_pub"] = n, now
            enroll_status("capturing", name, progress=n, total=ENROLL_SAMPLES, hint=hint)

        if e["last_box"] is not None:
            x1, y1, x2, y2 = e["last_box"]
            ops.append(("box", x1, y1, x2, y2, COLOR_OK, ""))
        ops.append(("text", f"Them: {disp}  {n}/{ENROLL_SAMPLES}", (10, 30), 0.65, COLOR_INFO))
        ops.append(("text", hint, (10, 65), 0.6, COLOR_WARN))

    # ---------- đồng bộ file khi bị sửa từ bên ngoài ----------
    def _housekeeping(self, now):
        if now - self.last_purge >= TRASH_PURGE_SECONDS:
            self.last_purge = now
            try:
                if TRASH_KEEP_DAYS > 0:
                    purge_trash(TRASH_KEEP_DAYS)
            except Exception as e:
                print(f"[WARN] Dọn thùng rác lỗi: {e}")
        if now - self.last_reload_check < RELOAD_CHECK_SECONDS:
            return
        self.last_reload_check = now
        sig = files_signature()
        if sig == self.files_sig:
            return
        try:
            new_db = load_db(DB_PATH)
            new_members = load_members(new_db)
        except Exception as e:
            print(f"[Reload] Đọc lại database/members lỗi, giữ bản cũ: {e}")
            self.files_sig = sig
            return
        self.db, self.members = new_db, new_members
        self.files_sig = files_signature()
        print(f"[Reload] Đã tải lại: {len(self.db)} người.")
        self.publish_members_list()

    # ---------- VERIFYING ----------
    def _abort_verify(self, name, status, now, color=COLOR_BAD):
        self.state, self.verify = "IDLE", None
        self._banner(f"{ascii_name(name)}: {status}", color, now)

    def _tick_verifying(self, frame, now, ops):
        v = self.verify
        name, disp = v["name"], ascii_name(v["name"])

        # Quyền có thể đổi giữa phiên (thu hồi, sửa file) -> kiểm tra lại mỗi frame
        if v["purpose"] == "unlock" and not self.can_unlock(name):
            print(f"  -> [Huỷ] '{name}' không còn quyền mở cửa, huỷ phiên.")
            self._abort_verify(name, "REVOKED", now)
            return
        if v["purpose"] == "owner_confirm" and (self.pending is None or not self.is_owner(name)):
            self.state, self.verify = "IDLE", None
            return

        dt = min(now - v["last_t"], 0.2)
        v["last_t"] = now
        v["n"] += 1

        g = analyze_gaze(self.face_mesh, frame)

        # Vẽ 2 chấm tròn tại tâm đồng tử (468 & 473) lên màn hình Live Stream / TFT
        for pt in g.get("iris_pts", []):
            ops.append(("circle", pt, 3, (0, 255, 255)))

        # Kiểm tra chu trình chớp mắt chuẩn: Nhắm (ear < ngưỡng) rồi Mở lại trong khoảng 0.04s - 0.5s
        if "ear" in g:
            if g["is_eyes_closed"]:
                if v["closed_since"] is None:
                    v["closed_since"] = now
            elif v["closed_since"] is not None:
                blink_dur = now - v["closed_since"]
                if BLINK_MIN_DURATION <= blink_dur <= BLINK_MAX_DURATION:
                    if not v["blinked"]:
                        v["blinked"] = True
                        print(f"  -> [Liveness] Phát hiện chớp mắt hợp lệ ({blink_dur:.2f}s)!")
                v["closed_since"] = None

        # Tích lũy thời gian nhìn thẳng (2.0 giây)
        if g["ok"]:
            v["fail_since"] = None
            v["hold"] += dt
        else:
            if v["fail_since"] is None:
                v["fail_since"] = now
            elif now - v["fail_since"] > GAZE_GRACE_SECONDS:
                if v["hold"] > 0:
                    print(f"  [Gaze] Lệch ánh nhìn đồng tử ({g['reason']}) -> reset T=0")
                v["hold"] = 0.0

        if DEBUG_GAZE and v["n"] % 5 == 0:
            if "gx" in g:
                print(f"[Gaze] gx={g['gx']:+.3f} gy={g['gy']:+.3f} "
                      f"ear={g['ear']:.2f} blink={v['blinked']} "
                      f"-> {g['reason']} | T={v['hold']:.1f}s")
            else:
                print(f"[Gaze] {g['reason']} | T={v['hold']:.1f}s")

        blink_str = "OK" if v["blinked"] else "Chop mat 1 lan!"
        prefix = "Chu nha " if v["purpose"] == "owner_confirm" else ""
        ops.append(("text",
                    f"{prefix}{disp}: Gaze {v['hold']:.1f}/{GAZE_HOLD_SECONDS:.0f}s | Blink: {blink_str}",
                    (10, 30), 0.65, COLOR_INFO))

        finished = None

        # Khi đã nhìn thẳng đủ 2s VÀ đã hoàn tất chớp mắt
        if v["hold"] >= GAZE_HOLD_SECONDS and v["blinked"]:
            if RECHECK_IDENTITY_AT_END:
                same, sc = confirm_identity(
                    self.app, frame, name, self.db, expected_point=g.get("nose_xy")
                )
                if same:
                    finished = "UNLOCKED"
                    v["score"] = sc
                else:
                    finished = "IDENTITY_MISMATCH"
            else:
                finished = "UNLOCKED"
        elif now - v["t_start"] > VERIFY_TIMEOUT_SECONDS:
            # Phân biệt rõ lỗi do không nhìn đủ 2s (GAZE_TIMEOUT) hay do chưa chớp mắt (NO_BLINK)
            if v["hold"] >= GAZE_HOLD_SECONDS and not v["blinked"]:
                finished = "NO_BLINK"
            else:
                finished = "GAZE_TIMEOUT"

        if not finished:
            return

        purpose, score = v["purpose"], v["score"]
        self.state, self.verify = "IDLE", None

        if finished == "UNLOCKED":
            if purpose == "owner_confirm":
                if not self._pending_accepts(name):
                    print(f"  -> [FAIL] '{name}' không còn đủ điều kiện xác nhận lệnh này.")
                    self.cooldown_until[name] = now + FAIL_COOLDOWN_SECONDS
                    log_event(name, score, frame, status="OWNER_NOT_ALLOWED")
                    return
                print(f"  -> [OK] Chủ nhà '{name}' xác nhận thành công.")
                log_event(name, score, frame, status="OWNER_CONFIRMED")
                self.cooldown_until[name] = now + FAIL_COOLDOWN_SECONDS
                self._complete_pending(name, now)
                return

            # Kiểm tra quyền LẦN CUỐI ngay trước khi gửi lệnh mở khoá
            if not self.can_unlock(name):
                print(f"  -> [FAIL] '{name}' vừa bị thu hồi quyền, KHÔNG mở khoá.")
                self.cooldown_until[name] = now + FAIL_COOLDOWN_SECONDS
                self._banner(f"{disp}: NO PERMISSION", COLOR_BAD, now)
                log_event(name, score, frame, status="NO_PERMISSION")
                return
            print(f"  -> [OK] {name} xác thực thành công (Mặt + Đồng tử {GAZE_HOLD_SECONDS:.0f}s + Chớp mắt)")
            send_unlock(name)
            self.cooldown_until[name] = now + UNLOCK_COOLDOWN_SECONDS
            self._banner(f"{disp}: UNLOCKED!", COLOR_OK, now)
            log_event(name, score, frame, status="UNLOCKED")
            return

        # Các trường hợp thất bại
        if finished == "IDENTITY_MISMATCH":
            print("  -> [FAIL] Kiểm tra lại danh tính cuối phiên không khớp, KHÔNG xác nhận.")
        elif finished == "NO_BLINK":
            print("  -> [FAIL] Đã nhìn thẳng đủ 2s nhưng không phát hiện chớp mắt (Nghi ngờ ảnh tĩnh).")
        else:
            print(f"  -> [FAIL] Quá {VERIFY_TIMEOUT_SECONDS:.0f}s không duy trì ánh nhìn 2s, huỷ phiên.")
        self.cooldown_until[name] = now + FAIL_COOLDOWN_SECONDS
        self._banner(f"{disp}: {FAIL_BANNERS[finished]}", COLOR_BAD, now)
        log_event(name, score, frame, status=finished if purpose == "unlock" else f"OWNER_{finished}")

    # ---------- IDLE ----------
    def _tick_idle(self, frame, now, ops):
        if self.pending is None:
            self._housekeeping(now)

        # ---------------- IDLE (QUÉT NHẬN DIỆN MẶT MỖI FRAME_SKIP) ----------------
        if self.frame_count % FRAME_SKIP != 0:
            return

        t0 = time.time()
        faces = self.app.get(frame)
        if DEBUG_FRAMES:
            print(f"[Debug] Frame {self.frame_count}: phát hiện {len(faces)} mặt, "
                  f"xử lý mất {time.time() - t0:.2f}s")

        self.overlay_items, self.overlay_t = [], now

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            name, score = recognize(face.normed_embedding, self.db, THRESHOLD)
            if DEBUG_FRAMES:
                print(f"[Debug]   Kết quả: {name or 'unknown'} (score={score:.3f})")

            color = COLOR_OK if name else (0, 0, 200)
            label = f"{ascii_name(name)} ({score:.2f})" if name else f"unknown ({score:.2f})"
            self.overlay_items.append((x1, y1, x2, y2, color, label))

            if name:
                if now < self.cooldown_until.get(name, 0.0):
                    continue
                print(f"  -> [Hành động] Xin chào {name}!")

                # Có lệnh đang chờ: chủ nhà đứng trước camera = bắt đầu xác nhận lệnh
                if self._pending_accepts(name):
                    print(f"  -> [Verify] Chủ nhà '{name}': hãy nhìn thẳng {GAZE_HOLD_SECONDS:.0f}s và chớp mắt 1 lần "
                          f"để xác nhận lệnh '{self.pending['kind']}'...")
                    self.state = "VERIFYING"
                    self.verify = new_verification(name, score, purpose="owner_confirm")
                    break

                if self.can_unlock(name):
                    print(f"  -> [Verify] Hãy nhìn thẳng vào camera {GAZE_HOLD_SECONDS:.0f}s và chớp mắt 1 lần...")
                    self.state = "VERIFYING"
                    self.verify = new_verification(name, score, purpose="unlock")
                    break
                else:
                    print(f"  -> [Info] {name} không có quyền mở khoá tự động, chỉ ghi log.")
                    self.cooldown_until[name] = now + FAIL_COOLDOWN_SECONDS
                    log_event(name, score, frame, status="NO_PERMISSION")
            else:
                if now - self.last_unknown_log_t >= UNKNOWN_LOG_COOLDOWN_SECONDS:
                    self.last_unknown_log_t = now
                    log_event(None, score, frame, status="UNKNOWN")

    # ---------- mỗi khung hình ----------
    def process_frame(self, frame):
        """Trả về danh sách thứ cần vẽ lên khung hình (khung, chấm đồng tử, chữ)."""
        self.frame_count += 1
        ops = []
        now = time.time()

        self._expire_pending(now)

        if self.state == "VERIFYING":
            self._tick_verifying(frame, now, ops)
        elif self.state == "ENROLLING":
            self._tick_enrolling(frame, now, ops)
        else:
            self._tick_idle(frame, now, ops)

        # Khung giữ lại khi ở trạng thái IDLE
        if self.state == "IDLE" and (now - self.overlay_t < OVERLAY_HOLD_SECONDS):
            for (x1, y1, x2, y2, color, label) in self.overlay_items:
                ops.append(("box", x1, y1, x2, y2, color, label))

        # Nhắc chủ nhà khi có lệnh đang chờ xác nhận
        if self.pending is not None and self.state != "ENROLLING" and not self._verifying_owner_confirm():
            remaining = max(0, int(self.pending["deadline"] - now))
            ops.append(("text", f"Chu nha xac nhan: {remaining}s", (10, 95), 0.6, COLOR_WARN))

        # Giữ Banner kết quả 2 giây trên màn hình
        if self.banner_info is not None:
            b_text, b_color, b_expire = self.banner_info
            if now < b_expire:
                ops.append(("text", b_text, (10, 65), 0.8, b_color))
            else:
                self.banner_info = None
        return ops


# ============ 7. VÒNG LẶP CHÍNH ============

def main():
    db = load_db(DB_PATH)
    if not db:
        raise RuntimeError(
            "Database khuôn mặt đang trống. Dừng service và chạy "
            "'python enroll.py <tên>' để thêm chủ nhà đầu tiên (xem INITIAL_OWNERS trong face_store.py)."
        )
    members = load_members(db)
    print(f"[Database] Đã load {len(db)} người: {list(db.keys())}")
    owners = [n for n, m in members.items() if m["role"] == "owner"]
    if not owners:
        print("[WARN] Chưa có owner nào: không ai xác nhận được enroll/grant/remove. "
              "Dùng 'python face_store.py role <tên> owner'.")
    if TRASH_KEEP_DAYS > 0:
        purge_trash(TRASH_KEEP_DAYS)

    app = load_model()
    face_mesh = load_face_mesh()
    init_logging()
    setup_mqtt()
    if STREAM_ENABLED:
        start_stream_server()

    # Màn TFT: việc lật gương do render_overlay() đảm nhiệm (chỉ lật ảnh, không lật chữ),
    # nên TftPreview luôn để mirror=False để tránh lật 2 lần.
    tft = None
    if TFT_ENABLED:
        try:
            from TFT_lcd import TftPreview
            tft = TftPreview(rotate=TFT_ROTATE, fps=TFT_FPS,
                             speed=TFT_SPEED_HZ, mirror=False)
            print("[TFT] Đã khởi tạo màn hình.")
        except Exception as e:
            print(f"[WARN] Không khởi tạo được TFT, bỏ qua: {e}")

    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Không mở được camera. Kiểm tra lại CAMERA_SOURCE hoặc kết nối camera.")

    print("[Camera] Đã kết nối. Nhấn 'q' để thoát." if SHOW_WINDOW else "[Camera] Đã kết nối (headless).")

    ctrl = DoorController(app, face_mesh, db, members)
    gate = MotionGate() if EVENT_DRIVEN else None

    try:
        while True:
            # Lệnh từ app xử lý trước, vì lệnh mới làm ctrl.pending != None (phải thức)
            while True:
                try:
                    topic, data = cmd_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    ctrl.handle_command(topic, data)
                except Exception:
                    print("[Members] Lỗi khi xử lý lệnh:")
                    traceback.print_exc()

            if not cap.grab():
                print("[Camera] Không đọc được frame, thử lại...")
                time.sleep(0.5)
                continue

            now = time.time()
            awake = True
            due = False
            if gate:
                awake = ctrl.is_busy() or now < gate.awake_until
                due = now - gate.last_check >= 1.0 / STANDBY_CHECK_FPS
                if not awake and not due:
                    continue                      # bỏ frame, không decode
            ret, frame = cap.retrieve()
            if not ret:
                continue

            if gate and due:
                if gate.check(frame, now) and not awake:
                    print(f"[Motion] Có chuyển động (active_cells={gate.active_cells}) -> đánh thức AI")
                awake = awake or now < gate.awake_until

            if awake:
                ops = ctrl.process_frame(frame)
                # Đang thấy mặt thì chưa cho ngủ
                if gate and ctrl.overlay_items and now - ctrl.overlay_t < 1.0:
                    gate.extend(now)
            else:
                ctrl._housekeeping(now)           # vẫn tải lại file / dọn rác khi đang ngủ
                ops = [("text", "STANDBY", (10, 30), 0.6, COLOR_INFO)]

            display_frame = render_overlay(frame.copy(), ops, mirror=False)
            if STREAM_ENABLED:
                publish(display_frame)
            if tft is not None:
                tft.push(render_overlay(frame, ops, mirror=True) if TFT_MIRROR else display_frame)
            if SHOW_WINDOW:
                cv2.imshow("Face Recognition - Smart Home", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[Dừng] Nhận Ctrl+C, đang thoát...")

    finally:
        if tft is not None:
            try:
                tft.close()
            except Exception as e:
                print(f"[WARN] Lỗi khi đóng TFT: {e}")
        cap.release()
        face_mesh.close()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        cleanup_mqtt()
        print("[Kết thúc] Đã giải phóng camera.")


if __name__ == "__main__":
    main()