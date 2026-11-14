"""
face_store.py - Lưu trữ DÙNG CHUNG cho recognition_faceID.py, enroll.py, enroll_live.py.

Quản lý:
  - models/face_database.pkl : embedding khuôn mặt của từng người
  - models/members.json      : vai trò (owner/member) và quyền mở cửa (can_unlock)
  - models/trash/            : embedding của người bị xoá (giữ TRASH_KEEP_DAYS ngày)

Công cụ dòng lệnh (chạy trên Pi, đây là nơi DUY NHẤT đổi được vai trò owner):
  python face_store.py members              # xem danh sách
  python face_store.py role <tên> owner     # đổi vai trò (owner | member)
  python face_store.py trash                # xem thùng rác
  python face_store.py restore <file.pkl>   # phục hồi người đã xoá (quyền mở cửa = TẮT)
  python face_store.py purge                # dọn thùng rác quá hạn
"""
import glob
import json
import os
import pickle
import re
import shutil
import sys
import time
import unicodedata
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(MODELS_DIR, "face_database.pkl")
MEMBERS_PATH = os.path.join(MODELS_DIR, "members.json")
TRASH_DIR = os.path.join(MODELS_DIR, "trash")

BACKUP_KEEP = 5            # số bản backup có đánh dấu thời gian được giữ lại cho mỗi file
TRASH_KEEP_DAYS = 7        # xoá mềm: giữ embedding bao nhiêu ngày (0 = xoá hẳn ngay, không giữ)

# Khi members.json chưa có (lần đầu) hoặc bị hỏng, những tên này được coi là owner + được mở cửa.
# Tất cả người còn lại trong database mặc định là member và KHÔNG được mở cửa (an toàn).
INITIAL_OWNERS = {"hvt", "Nhu"}

NAME_MAX_LEN = 30
_BAD_CHARS = set('/\\",')
RESERVED_NAMES = {"unknown"}
ROLES = ("owner", "member")


# ============ TÊN ============

def normalize_name(name):
    """Chuẩn hoá Unicode NFC (chữ có dấu từ app/bàn phím có thể ở dạng NFC hoặc NFD) và gọn khoảng trắng."""
    if not isinstance(name, str):
        return None
    name = unicodedata.normalize("NFC", name).strip()
    return re.sub(r"\s+", " ", name)


def validate_name(name):
    """Trả về (tên_đã_chuẩn_hoá, None) nếu hợp lệ, ngược lại (None, mã_lỗi)."""
    n = normalize_name(name)
    if not n:
        return None, "empty"
    if len(n) > NAME_MAX_LEN:
        return None, "too_long"
    if any(ch in _BAD_CHARS for ch in n):
        return None, "bad_char"
    if any(unicodedata.category(ch).startswith("C") for ch in n):
        return None, "control_char"
    if n.casefold() in RESERVED_NAMES:
        return None, "reserved"
    return n, None


def ascii_name(text):
    """Bỏ dấu để vẽ lên TFT/stream (cv2.putText font Hershey không vẽ được chữ có dấu)."""
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.encode("ascii", "replace").decode("ascii")


def file_safe(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_name(name)).strip("_") or "member"


def name_taken(members, db, name):
    """Tên đã tồn tại chưa (không phân biệt hoa/thường)."""
    key = name.casefold()
    return any(n.casefold() == key for n in set(members) | set(db))


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


# ============ GHI FILE AN TOÀN ============

def _atomic_write(path, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _backup(path, keep=BACKUP_KEEP):
    """Backup có đánh dấu thời gian, giữ `keep` bản gần nhất (bản cũ không bị ghi đè như '.bak' đơn)."""
    if not os.path.exists(path):
        return
    ts = time.strftime("%Y%m%d-%H%M%S")
    dst, i = f"{path}.{ts}.bak", 1
    while os.path.exists(dst):
        dst = f"{path}.{ts}-{i}.bak"
        i += 1
    shutil.copy2(path, dst)
    baks = sorted(glob.glob(path + ".*.bak"), key=os.path.getmtime)
    for old in baks[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass


# ============ DATABASE KHUÔN MẶT ============

def load_db(path=DB_PATH):
    """Trả về dict {tên: ndarray (N, 512) float32}. File chưa có -> {}."""
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        raw = pickle.load(f)
    db = {}
    for name, embs in raw.items():
        if len(embs) == 0:
            continue
        db[name] = np.asarray(embs, dtype=np.float32)
    return db


def save_db(db, path=DB_PATH):
    raw = {n: [np.asarray(e, dtype=np.float32) for e in embs] for n, embs in db.items() if len(embs)}
    _backup(path)
    _atomic_write(path, pickle.dumps(raw))


# ============ THÀNH VIÊN (members.json) ============

def _new_member_record(name):
    owner = name in INITIAL_OWNERS
    return {"role": "owner" if owner else "member", "can_unlock": owner, "created": now_iso()}


def save_members(members):
    _backup(MEMBERS_PATH)
    data = json.dumps({"version": 1, "members": members}, ensure_ascii=False, indent=2)
    _atomic_write(MEMBERS_PATH, data.encode("utf-8"))


def _clean_member(m):
    """Dữ liệu sai định dạng -> hạ về quyền thấp nhất (member, không mở cửa)."""
    if not isinstance(m, dict):
        return {"role": "member", "can_unlock": False, "created": now_iso()}
    role = m.get("role") if m.get("role") in ROLES else "member"
    return {"role": role, "can_unlock": m.get("can_unlock") is True,
            "created": m.get("created") or now_iso()}


def load_members(db):
    """
    Đọc members.json và đối chiếu với database:
      - người có trong database mà chưa có hồ sơ -> thêm (INITIAL_OWNERS là owner, còn lại member + KHÔNG mở cửa)
      - hồ sơ không còn khuôn mặt trong database -> bỏ
    File thiếu/hỏng -> dựng lại từ database (fail-safe), file hỏng được đổi tên để giữ lại.
    """
    members, changed = None, False
    if os.path.exists(MEMBERS_PATH):
        try:
            with open(MEMBERS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data["members"] if isinstance(data, dict) and "members" in data else data
            if not isinstance(raw, dict):
                raise ValueError("members.json không phải dict")
            members = {n: _clean_member(m) for n, m in raw.items()}
        except Exception as e:
            bad = f"{MEMBERS_PATH}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
            print(f"[Members] CẢNH BÁO: members.json hỏng ({e}). Đổi tên thành {bad}, dựng lại từ database.")
            os.replace(MEMBERS_PATH, bad)
    if members is None:
        members, changed = {}, True

    for n in db:
        if n not in members:
            members[n] = _new_member_record(n)
            changed = True
            print(f"[Members] Thêm hồ sơ cho '{n}': role={members[n]['role']}, can_unlock={members[n]['can_unlock']}")
    for n in list(members):
        if n not in db:
            del members[n]
            changed = True
            print(f"[Members] Bỏ hồ sơ '{n}' vì không còn khuôn mặt trong database.")
    if changed:
        save_members(members)
    return members


def public_list(members):
    """Danh sách gửi lên app (chỉ tên, vai trò, quyền; KHÔNG có embedding)."""
    items = sorted(members.items(), key=lambda kv: (kv[1]["role"] != "owner", kv[0].casefold()))
    return [{"name": n, "role": m["role"], "can_unlock": bool(m["can_unlock"])} for n, m in items]


# ============ THÙNG RÁC (XOÁ MỀM) ============

def _unique_path(path):
    base, ext = os.path.splitext(path)
    p, i = path, 1
    while os.path.exists(p):
        p = f"{base}-{i}{ext}"
        i += 1
    return p


def move_to_trash(name, embeddings, member):
    path = _unique_path(os.path.join(TRASH_DIR, f"{file_safe(name)}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"))
    payload = {"name": name, "member": member, "deleted_at": now_iso(),
               "embeddings": [np.asarray(e, dtype=np.float32) for e in embeddings]}
    _atomic_write(path, pickle.dumps(payload))
    return path


def purge_trash(days=TRASH_KEEP_DAYS):
    """Xoá file trong thùng rác quá `days` ngày (days <= 0: xoá tất cả). Trả về số file đã xoá."""
    n = 0
    cutoff = time.time() - days * 86400
    for p in glob.glob(os.path.join(TRASH_DIR, "*.pkl")):
        try:
            if days <= 0 or os.path.getmtime(p) < cutoff:
                os.remove(p)
                n += 1
        except OSError:
            pass
    return n


def files_signature():
    """Dấu vết (mtime, size) của database + members để phát hiện file bị sửa từ bên ngoài."""
    sig = []
    for p in (DB_PATH, MEMBERS_PATH):
        try:
            st = os.stat(p)
            sig.append((st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            sig.append(None)
    return tuple(sig)


# ============ CLI QUẢN TRỊ ============

def _cli(argv):
    cmd = argv[0] if argv else "members"
    db = load_db()
    members = load_members(db)

    if cmd == "members":
        for m in public_list(members):
            print(f"{m['name']:<30} {m['role']:<7} can_unlock={m['can_unlock']}")
    elif cmd == "role" and len(argv) == 3 and argv[2] in ROLES:
        name, role = normalize_name(argv[1]), argv[2]
        if name not in members:
            sys.exit(f"Không có '{name}'.")
        members[name]["role"] = role
        save_members(members)
        print(f"Đã đặt '{name}' = {role}. Quyền mở cửa đổi trong app (grant/revoke).")
    elif cmd == "trash":
        files = sorted(glob.glob(os.path.join(TRASH_DIR, "*.pkl")))
        for p in files:
            print(os.path.basename(p))
        print(f"({len(files)} file, tự xoá sau {TRASH_KEEP_DAYS} ngày)")
    elif cmd == "restore" and len(argv) == 2:
        p = argv[1] if os.path.isabs(argv[1]) else os.path.join(TRASH_DIR, argv[1])
        with open(p, "rb") as f:
            payload = pickle.load(f)
        name = payload["name"]
        if name_taken(members, db, name):
            sys.exit(f"Tên '{name}' đã tồn tại, không phục hồi.")
        db[name] = np.asarray(payload["embeddings"], dtype=np.float32)
        save_db(db)
        members[name] = {"role": "member", "can_unlock": False, "created": now_iso()}
        save_members(members)
        print(f"Đã phục hồi '{name}' (member, KHÔNG mở cửa). Cấp quyền trong app khi cần.")
    elif cmd == "purge":
        print(f"Đã xoá {purge_trash()} file quá hạn.")
    else:
        print(__doc__)


if __name__ == "__main__":
    _cli(sys.argv[1:])
