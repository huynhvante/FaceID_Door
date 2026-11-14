import threading
import time

import cv2
import numpy as np
import spidev
from gpiozero import DigitalOutputDevice

# Cac chan BCM da dung trong living_room.py (khong duoc dung cho TFT)
RESERVED_PINS = {
    4: "DHT22", 17: "REED (cua)", 18: "SERVO SG90", 22: "LED",
    23: "QUAT INA", 24: "QUAT INB", 25: "BUZZER", 27: "PIR",
    2: "I2C SDA (BH1750)", 3: "I2C SCL (BH1750)",
    5: "NUT DEN SIG2", 6: "NUT DEN SIG1",
    13: "NUT QUAT SIG2", 26: "NUT QUAT SIG1",
}


# các chan vật lý TFT tương ứng với Raspberry Pi 5:
TFT_DC = 16    # chân vật lý 36
TFT_RST = 20   # chân vật lý 38
TFT_MOSI = 10  # chân vật lý 19
TFT_SCLK = 11  # chân vật lý 23
TFT_CS = 8     # chân vật lý 24
TFT_LED = 12   # chân vật lý 32 (3.3V) - bật đèn nền TFT


# Lenh ST7789
SWRESET = 0x01
SLPOUT = 0x11
INVOFF = 0x20
DISPON = 0x29
CASET = 0x2A
RASET = 0x2B
RAMWR = 0x2C
MADCTL = 0x36
COLMOD = 0x3A

# MADCTL, thu tu mau RGB (bit 3 = 0): xoay 0 / 90 / 180 / 270 do
MADCTL_BY_ROTATE = {0: 0x00, 90: 0x60, 180: 0xC0, 270: 0xA0}


class St7789:
    """Driver toi gian cho ST7789V 240x320 qua SPI."""

    def __init__(self, dc=16, rst=20, bus=0, dev=0, speed=24_000_000, rotate=90):
        if rotate not in MADCTL_BY_ROTATE:
            raise ValueError("rotate phai la 0, 90, 180 hoac 270")
        bad = [f"GPIO{p} ({RESERVED_PINS[p]})" for p in (dc, rst) if p in RESERVED_PINS]
        if bad:
            raise ValueError("Trung chan voi living_room.py: " + ", ".join(bad))

        self.rotate = rotate
        self.width, self.height = (320, 240) if rotate in (90, 270) else (240, 320)

        self.dc = DigitalOutputDevice(dc)
        self.rst = DigitalOutputDevice(rst, initial_value=True)
        self.spi = spidev.SpiDev()
        self.spi.open(bus, dev)
        self.spi.max_speed_hz = speed
        self.spi.mode = 0

        self._init_display()

    # ---- muc thap ----
    def _cmd(self, c):
        self.dc.off()
        self.spi.writebytes([c])

    def _data(self, buf):
        self.dc.on()
        buf = bytes(buf)
        for i in range(0, len(buf), 4096):
            self.spi.writebytes2(buf[i:i + 4096])

    def _write(self, c, data=()):
        self._cmd(c)
        if len(data):
            self._data(data)

    def _init_display(self):
        self.rst.on()
        time.sleep(0.05)
        self.rst.off()
        time.sleep(0.05)
        self.rst.on()
        time.sleep(0.15)

        self._write(SWRESET)
        time.sleep(0.15)
        self._write(MADCTL, [MADCTL_BY_ROTATE[self.rotate]])
        self._write(COLMOD, [0x55])            # RGB565
        self._write(0xB2, [0x0C, 0x0C, 0x00, 0x33, 0x33])
        self._write(0xB7, [0x35])
        self._write(0xBB, [0x1F])
        self._write(0xC0, [0x2C])
        self._write(0xC2, [0x01])
        self._write(0xC3, [0x12])
        self._write(0xC4, [0x20])
        self._write(0xC6, [0x0F])
        self._write(0xD0, [0xA4, 0xA1])
        self._write(0xE0, [0xD0, 0x08, 0x11, 0x08, 0x0C, 0x15, 0x39, 0x33,
                           0x50, 0x36, 0x13, 0x14, 0x29, 0x2D])
        self._write(0xE1, [0xD0, 0x08, 0x10, 0x08, 0x06, 0x06, 0x39, 0x44,
                           0x51, 0x0B, 0x16, 0x14, 0x2F, 0x31])
        self._write(INVOFF)
        self._write(SLPOUT)
        time.sleep(0.15)
        self._write(DISPON)
        time.sleep(0.05)

    # ---- ve ----
    def _set_window(self):
        x1, y1 = self.width - 1, self.height - 1
        self._write(CASET, [0, 0, x1 >> 8, x1 & 0xFF])
        self._write(RASET, [0, 0, y1 >> 8, y1 & 0xFF])
        self._cmd(RAMWR)

    def _letterbox(self, frame):
        """Thu nho giu nguyen ti le, them vien den neu can (640x480 -> 320x240 la vua khit)."""
        h, w = frame.shape[:2]
        if (w, h) == (self.width, self.height):
            return frame
        scale = min(self.width / w, self.height / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        x0, y0 = (self.width - nw) // 2, (self.height - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = small
        return canvas

    def show_bgr(self, frame, mirror=False):
        """Hien mot khung hinh BGR cua OpenCV len man."""
        if mirror:
            frame = cv2.flip(frame, 1)
        frame = self._letterbox(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint16)
        rgb565 = ((rgb[..., 0] & 0xF8) << 8) | ((rgb[..., 1] & 0xFC) << 3) | (rgb[..., 2] >> 3)
        self._set_window()
        self._data(rgb565.astype(">u2").tobytes())

    def close(self):
        try:
            self.show_bgr(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        except Exception:
            pass
        self.spi.close()
        self.dc.close()
        self.rst.close()


class TftPreview:
    """Chay man TFT trong thread rieng; vong lap chinh chi can goi push(frame)."""

    def __init__(self, dc=16, rst=20, speed=24_000_000, rotate=90, fps=12, mirror=False):
        self.lcd = St7789(dc=dc, rst=rst, speed=speed, rotate=rotate)
        self.mirror = mirror
        self.period = 1.0 / max(fps, 1)
        self.shown = 0                      # so khung da hien (de do FPS)
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, bgr_frame):
        """Giao khung hinh moi nhat; khung cu chua kip hien se bi bo qua."""
        with self._lock:
            self._latest = bgr_frame

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                frame, self._latest = self._latest, None
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.time()
            try:
                self.lcd.show_bgr(frame, mirror=self.mirror)
            except Exception as e:
                print(f"[TFT] Loi khi ve, dung thread TFT: {e}")
                return
            self.shown += 1
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.lcd.close()


# ---------------- test doc lap: camera -> TFT ----------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test camera -> TFT")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--rotate", type=int, default=90, choices=[0, 90, 180, 270])
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--speed", type=int, default=24_000_000)
    ap.add_argument("--mirror", action="store_true", help="Lat guong ngang")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Khong mo duoc camera.")

    tft = TftPreview(rotate=args.rotate, fps=args.fps, speed=args.speed, mirror=args.mirror)
    print("Dang hien camera len TFT. Nhan Ctrl+C de thoat.")
    last_t, last_n = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            tft.push(frame)
            now = time.time()
            if now - last_t >= 2.0:
                print(f"TFT: {(tft.shown - last_n) / (now - last_t):.1f} FPS")
                last_t, last_n = now, tft.shown
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        tft.close()