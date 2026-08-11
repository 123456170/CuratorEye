import json
import math
import os
import random
import socket
import threading
import time
import uuid
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

try:
    import redis
except Exception:
    redis = None

try:
    from skimage.metrics import structural_similarity as ssim_compare
except Exception:
    ssim_compare = None

try:
    import requests
except Exception:
    requests = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from fastapi import FastAPI
    from fastapi.responses import Response, StreamingResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False


# ----------------------------------------------------------------------------
# CuratorEye configuration
# ----------------------------------------------------------------------------

GALLERY_W, GALLERY_H = 640, 420
ARTIFACT_W, ARTIFACT_H = 640, 480
FPS = 15

# Demo security-console PIN. This is not an API key; it only locks the UI.
SECURITY_PIN = os.getenv("CURATOREYE_PIN", "2026")


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def random_token():
    return uuid.uuid4().hex[:8]


def to_rgb(frame):
    if frame is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def compare_images(img_a, img_b):
    """
    SSIM-style similarity score in [0, 1].
    Falls back to a lightweight luminance/structure approximation if scikit-image
    is unavailable.
    """
    if img_a is None or img_b is None:
        return 1.0

    a = cv2.resize(img_a, (128, 128)).astype(np.uint8)
    b = cv2.resize(img_b, (128, 128)).astype(np.uint8)

    if ssim_compare is not None:
        try:
            return float(ssim_compare(a, b, data_range=255))
        except Exception:
            pass

    # Fallback SSIM-like score.
    af = a.astype(np.float32)
    bf = b.astype(np.float32)

    mu_a = float(af.mean())
    mu_b = float(bf.mean())
    sig_a = float(af.std())
    sig_b = float(bf.std())
    sig_ab = float(((af - mu_a) * (bf - mu_b)).mean())

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    numerator = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sig_a ** 2 + sig_b ** 2 + c2)

    if denominator <= 1e-6:
        return 1.0

    return float(clamp(numerator / denominator, 0.0, 1.0))


def jpeg_bytes(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return b""
    return buf.tobytes()


# ----------------------------------------------------------------------------
# Redis fallback: if a real Redis server is not available, use an in-memory
# privacy-safe simulator with the same tiny subset of commands.
# ----------------------------------------------------------------------------

class MemoryRedis:
    def __init__(self):
        self.lock = threading.Lock()
        self.kv = {}
        self.lists = {}

    def _purge(self):
        now = time.time()
        expired = [k for k, v in self.kv.items() if v[0] < now]
        for k in expired:
            del self.kv[k]

    def ping(self):
        return True

    def setex(self, key, ttl, value):
        with self.lock:
            self._purge()
            self.kv[key] = (time.time() + float(ttl), str(value))

    def get(self, key):
        with self.lock:
            self._purge()
            item = self.kv.get(key)
            return item[1] if item else None

    def lpush(self, key, *values):
        with self.lock:
            dq = self.lists.setdefault(key, deque(maxlen=200))
            for value in values:
                dq.appendleft(str(value))

    def lrange(self, key, start, end):
        with self.lock:
            self._purge()
            items = list(self.lists.get(key, deque()))
            if end == -1:
                end = len(items) - 1
            elif end < 0:
                end = len(items) + end
            return items[start:end + 1]


def make_redis():
    """
    Try real Redis first. If unavailable, fall back to in-memory simulator.
    This keeps the demo runnable without any service setup.
    """
    if redis is not None:
        try:
            client = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=1,
            )
            client.ping()
            return client, True
        except Exception:
            pass

    return MemoryRedis(), False


# ----------------------------------------------------------------------------
# CuratorEye engine
# ----------------------------------------------------------------------------

class CuratorEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.redis, self.redis_real = make_redis()

        self.t = 0.0
        self.frame_count = 0
        self.running = True
        self.last_error = None

        self.gallery_frame = np.zeros((GALLERY_H, GALLERY_W, 3), dtype=np.uint8)
        self.artifact_frame = np.zeros((ARTIFACT_H, ARTIFACT_W, 3), dtype=np.uint8)

        # Exhibits: wall artwork rectangles, visitor standing points, gaze points.
        self.exhibits = [
            {
                "id": "EX-01",
                "room": "A",
                "rect": (55, 55, 95, 110),
                "color": (178, 120, 62),
                "point": (102, 255),
                "gaze_point": (102, 110),
            },
            {
                "id": "EX-02",
                "room": "A",
                "rect": (195, 48, 95, 112),
                "color": (78, 140, 168),
                "point": (242, 250),
                "gaze_point": (242, 104),
            },
            {
                "id": "EX-03",
                "room": "B",
                "rect": (355, 55, 95, 110),
                "color": (88, 160, 102),
                "point": (402, 255),
                "gaze_point": (402, 110),
            },
            {
                "id": "EX-04",
                "room": "B",
                "rect": (495, 48, 95, 112),
                "color": (142, 96, 168),
                "point": (542, 250),
                "gaze_point": (542, 104),
            },
        ]

        for exhibit in self.exhibits:
            exhibit.update(
                {
                    "engagement": 0.0,
                    "total_dwell": 0.0,
                    "visits": 0,
                    "current_dwellers": 0,
                    "avg_dwell": 0.0,
                }
            )

        self.tracks = []
        self.heat = np.zeros((GALLERY_H, GALLERY_W), dtype=np.float32)

        # Heat stamp kernel.
        self.heat_kernel = np.zeros((33, 33), dtype=np.float32)
        cv2.circle(self.heat_kernel, (16, 16), 16, 1.0, -1)
        self.heat_kernel = cv2.GaussianBlur(self.heat_kernel, (0, 0), 5)
        self.heat_kernel /= max(1e-6, float(self.heat_kernel.max()))

        self.flow = {"A->B": 0, "B->A": 0}

        self.alerts = deque(maxlen=100)
        self.alert_seq = 0

        # Protected artifact zone.
        self.zone = np.array(
            [
                [210, 130],
                [430, 130],
                [465, 360],
                [175, 360],
            ],
            dtype=np.int32,
        )
        self.zone_contour = self.zone.reshape((-1, 1, 2)).astype(np.int32)

        self.artifact_center = (320, 205)
        self.sprite_top_left = (255, 110)

        self.artifact_sprite = self.make_artifact_sprite()
        self.artifact_mask = cv2.cvtColor(self.artifact_sprite, cv2.COLOR_BGR2GRAY)
        _, self.artifact_mask = cv2.threshold(self.artifact_mask, 12, 255, cv2.THRESH_BINARY)

        self.artifact_pose = (0.0, 0.0, 0.0)
        self.artifact_baseline = self.render_artifact_crop(0.0, 0.0, 0.0)
        self.latest_artifact_crop = self.artifact_baseline.copy()
        self.prev_artifact_gray = None

        self.ssim_score = 1.0
        self.drift_score = 0.0
        self.micro_motion = 0.0

        self.breach = False
        self.warning = False

        self.hand_pos = (ARTIFACT_W + 120, 260)
        self.hand_active = False
        self.hand_progress = 0.0
        self.force_start = -100.0
        self.force_until = -100.0

        self.last_drift_alert = 0.0
        self.last_status_push = 0.0

        self.gallery_bg = self.render_gallery_background()
        self.artifact_bg = self.render_artifact_background()
        self.gallery_bg_gray = cv2.GaussianBlur(
            cv2.cvtColor(self.gallery_bg, cv2.COLOR_BGR2GRAY),
            (21, 21),
            0,
        )

        self.init_tracks(4)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        last = time.time()
        while self.running:
            try:
                now = time.time()
                dt = min(0.08, max(0.001, now - last))
                last = now

                with self.lock:
                    self.update(dt)
                    self.render()
            except Exception as exc:
                self.last_error = str(exc)

            time.sleep(1.0 / FPS)

    def update(self, dt):
        self.t += dt
        self.frame_count += 1

        self.update_tracks(dt)
        self.update_heat(dt)
        self.update_security(dt)
        self.publish_metrics()

    def render(self):
        self.render_gallery()
        self.render_artifact()

    # ------------------------------------------------------------------
    # Visitor simulation and analytics
    # ------------------------------------------------------------------

    def init_tracks(self, count):
        self.tracks = [self.new_track() for _ in range(count)]

    def new_track(self):
        track = {
            "id": random_token(),
            "x": float(random.randint(70, GALLERY_W - 70)),
            "y": float(random.randint(235, GALLERY_H - 45)),
            "vx": 0.0,
            "vy": 0.0,
            "room": "A",
            "state": "walk",
            "target_idx": None,
            "target": (0.0, 0.0),
            "gaze": (0.0, 0.0),
            "dwell_timer": 0.0,
            "current_dwell": 0.0,
            "ttl": random.uniform(14.0, 30.0),
            "phase": random.uniform(0.0, 2.0 * math.pi),
            "yaw": 0.0,
            "pitch": 0.0,
            "box": None,
            "detected": False,
        }
        track["room"] = "A" if track["x"] < GALLERY_W // 2 else "B"
        self.choose_target(track)
        return track

    def choose_target(self, track):
        if random.random() < 0.82:
            idx = random.randrange(len(self.exhibits))
            exhibit = self.exhibits[idx]
            px, py = exhibit["point"]
            track["target_idx"] = idx
            track["target"] = (
                float(px + random.randint(-16, 16)),
                float(py + random.randint(-6, 22)),
            )
        else:
            track["target_idx"] = None
            track["target"] = (
                float(random.randint(60, GALLERY_W - 60)),
                float(random.randint(235, GALLERY_H - 45)),
            )

        track["state"] = "walk"
        track["dwell_timer"] = 0.0
        track["current_dwell"] = 0.0

    def retire_track(self, track):
        """
        Privacy rule: tokens expire and are replaced by new random tokens.
        No persistent identity or biometric linkage is kept.
        """
        track["id"] = random_token()
        track["ttl"] = random.uniform(15.0, 30.0)
        track["current_dwell"] = 0.0
        self.choose_target(track)

    def update_tracks(self, dt):
        for exhibit in self.exhibits:
            exhibit["current_dwellers"] = 0
            exhibit["engagement"] *= math.exp(-dt / 35.0)

        for track in self.tracks:
            track["ttl"] -= dt
            if track["ttl"] <= 0:
                self.retire_track(track)

            if track["state"] == "dwell":
                track["current_dwell"] += dt
                idx = track.get("target_idx")

                if idx is not None:
                    exhibit = self.exhibits[idx]
                    exhibit["current_dwellers"] += 1
                    exhibit["total_dwell"] += dt
                    exhibit["engagement"] += dt * 2.5

                    gx, gy = exhibit["gaze_point"]
                    track["gaze"] = (
                        gx + math.sin(self.t * 1.7 + track["phase"]) * 4.0,
                        gy + math.cos(self.t * 1.3 + track["phase"]) * 3.0,
                    )

                    self.add_heat(track["gaze"][0], track["gaze"][1], dt * 2.4)
                    self.add_heat(track["x"], track["y"] - 35, dt * 0.5)

                track["dwell_timer"] -= dt
                track["x"] += math.sin(self.t * 2.0 + track["phase"]) * dt * 2.0
                track["y"] += math.cos(self.t * 1.6 + track["phase"]) * dt * 1.1

                if track["dwell_timer"] <= 0:
                    self.choose_target(track)

            else:
                track["current_dwell"] = max(0.0, track["current_dwell"] - dt * 2.0)

                tx, ty = track["target"]
                dx = tx - track["x"]
                dy = ty - track["y"]
                dist = math.hypot(dx, dy)

                if dist < 14:
                    if track.get("target_idx") is not None:
                        track["state"] = "dwell"
                        track["dwell_timer"] = random.uniform(2.5, 8.0)
                        exhibit = self.exhibits[track["target_idx"]]
                        exhibit["visits"] += 1
                        track["gaze"] = exhibit["gaze_point"]
                    else:
                        self.choose_target(track)
                else:
                    speed = 46.0
                    desired_vx = dx / dist * speed + random.uniform(-8, 8)
                    desired_vy = dy / dist * speed + random.uniform(-5, 5)

                    track["vx"] = 0.82 * track["vx"] + 0.18 * desired_vx
                    track["vy"] = 0.82 * track["vy"] + 0.18 * desired_vy

                    track["x"] += track["vx"] * dt
                    track["y"] += track["vy"] * dt
                    track["gaze"] = (tx, ty)

            track["x"] = clamp(track["x"], 28, GALLERY_W - 28)
            track["y"] = clamp(track["y"], 225, GALLERY_H - 38)

            new_room = "A" if track["x"] < GALLERY_W // 2 else "B"
            if new_room != track["room"]:
                key = f"{track['room']}->{new_room}"
                if key in self.flow:
                    self.flow[key] += 1
                track["room"] = new_room
                track["ttl"] = max(track["ttl"], random.uniform(12.0, 22.0))

            hx = track["x"]
            hy = track["y"] - 48
            gx, gy = track["gaze"]

            yaw = math.degrees(math.atan2(gy - hy, gx - hx))
            track["yaw"] = clamp(yaw, -85, 85)
            track["pitch"] = clamp((gy - hy) * 0.12, -22, 22)

        for exhibit in self.exhibits:
            exhibit["avg_dwell"] = exhibit["total_dwell"] / max(1, exhibit["visits"])

    def add_heat(self, x, y, amount):
        x = int(clamp(x, 0, GALLERY_W - 1))
        y = int(clamp(y, 0, GALLERY_H - 1))
        r = 16

        x0 = max(0, x - r)
        y0 = max(0, y - r)
        x1 = min(GALLERY_W, x + r + 1)
        y1 = min(GALLERY_H, y + r + 1)

        kx0 = x0 - (x - r)
        ky0 = y0 - (y - r)
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)

        if x1 <= x0 or y1 <= y0 or kx1 <= kx0 or ky1 <= ky0:
            return

        self.heat[y0:y1, x0:x1] += self.heat_kernel[ky0:ky1, kx0:kx1] * amount

    def update_heat(self, dt):
        self.heat *= math.exp(-dt / 30.0)
        np.clip(self.heat, 0.0, 6.0, out=self.heat)

    # ------------------------------------------------------------------
    # Protected artifact security simulation
    # ------------------------------------------------------------------

    def make_artifact_sprite(self):
        sprite = np.zeros((190, 130, 3), dtype=np.uint8)

        # Base.
        cv2.ellipse(sprite, (65, 145), (42, 32), 0, 0, 360, (120, 150, 180), -1)

        # Body.
        cv2.ellipse(sprite, (65, 100), (36, 50), 0, 0, 360, (160, 180, 205), -1)

        # Neck.
        cv2.rectangle(sprite, (52, 35), (78, 72), (150, 170, 195), -1)

        # Lip.
        cv2.ellipse(sprite, (65, 30), (22, 12), 0, 0, 360, (185, 200, 220), -1)

        # Decorative bands.
        cv2.line(sprite, (30, 100), (100, 100), (80, 100, 130), 3)
        cv2.line(sprite, (34, 116), (96, 116), (80, 100, 130), 2)

        return sprite

    def rotate_sprite(self, angle, dx, dy):
        h, w = self.artifact_sprite.shape[:2]
        center = (w / 2.0, h / 2.0)

        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        m[0, 2] += dx
        m[1, 2] += dy

        rotated = cv2.warpAffine(
            self.artifact_sprite,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpAffine(
            self.artifact_mask,
            m,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )

        return rotated, mask

    def paste_sprite(self, frame, sprite, mask):
        x, y = self.sprite_top_left
        h, w = sprite.shape[:2]

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(ARTIFACT_W, x + w)
        y1 = min(ARTIFACT_H, y + h)

        if x1 <= x0 or y1 <= y0:
            return

        sx0 = x0 - x
        sy0 = y0 - y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        roi = frame[y0:y1, x0:x1]
        m = mask[sy0:sy1, sx0:sx1]
        sp = sprite[sy0:sy1, sx0:sx1]

        roi[m > 0] = sp[m > 0]

    def render_artifact_crop(self, angle, dx, dy):
        rotated, mask = self.rotate_sprite(angle, dx, dy)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        gray[mask == 0] = 0
        return gray

    def update_security(self, dt):
        # Simulated hand trajectory. It automatically enters periodically,
        # and can also be forced from the security console.
        forced = self.t < self.force_until
        p = 0.0

        if forced:
            elapsed = self.t - self.force_start
            if elapsed < 0.65:
                p = elapsed / 0.65
            elif elapsed < 4.35:
                p = 1.0
            else:
                p = max(0.0, 1.0 - (elapsed - 4.35) / 1.35)
        else:
            cycle = self.t % 18.0
            if 9.5 <= cycle < 10.7:
                p = (cycle - 9.5) / 1.2
            elif 10.7 <= cycle < 13.3:
                p = 1.0
            elif 13.3 <= cycle < 14.5:
                p = 1.0 - (cycle - 13.3) / 1.2

        p = clamp(p, 0.0, 1.0)
        self.hand_progress = p
        self.hand_active = p > 0.02

        start = np.array([ARTIFACT_W + 90.0, 300.0], dtype=np.float32)
        target = np.array(
            [self.artifact_center[0] + 25.0, self.artifact_center[1] + 35.0],
            dtype=np.float32,
        )

        pos = start + (target - start) * p
        pos[1] -= math.sin(p * math.pi) * 45.0
        self.hand_pos = (float(pos[0]), float(pos[1]))

        # Proximity check.
        inside = False
        dist = -100.0

        if self.hand_active:
            dist = float(cv2.pointPolygonTest(self.zone_contour, self.hand_pos, True))
            inside = dist >= 0.0

        self.warning = self.hand_active and (not inside) and dist > -42.0

        was_breach = self.breach
        self.breach = bool(inside)

        if self.breach and not was_breach:
            self.add_alert(
                "PROXIMITY_BREACH",
                "Protection zone breached by simulated hand. Depth proximity + micro-motion alert fired.",
                "critical",
            )

        if not self.breach and was_breach:
            self.add_alert(
                "ZONE_CLEAR",
                "Protection zone cleared.",
                "info",
            )

        # Artifact micro-motion pose.
        angle = (
            1.1 * math.sin(self.t * 0.7)
            + 0.35 * math.sin(self.t * 2.6)
            + random.uniform(-0.12, 0.12)
        )
        dx = 0.9 * math.sin(self.t * 1.4) + random.uniform(-0.15, 0.15)
        dy = 0.7 * math.sin(self.t * 1.9) + random.uniform(-0.12, 0.12)

        if self.breach:
            angle += 4.5 * math.sin(self.t * 18.0)
            dx += 5.0 * math.sin(self.t * 14.0)
            dy += 2.0 * math.cos(self.t * 16.0)

        self.artifact_pose = (angle, dx, dy)

        # SSIM drift scoring against registered baseline.
        if self.frame_count % 2 == 0:
            live_crop = self.render_artifact_crop(angle, dx, dy)
            self.latest_artifact_crop = live_crop

            self.ssim_score = compare_images(self.artifact_baseline, live_crop)
            raw_drift = clamp(1.0 - self.ssim_score, 0.0, 1.0)
            self.drift_score = 0.72 * self.drift_score + 0.28 * raw_drift

            if self.prev_artifact_gray is not None:
                motion = float(
                    cv2.mean(cv2.absdiff(live_crop, self.prev_artifact_gray))[0]
                ) / 255.0
                self.micro_motion = 0.7 * self.micro_motion + 0.3 * motion

            self.prev_artifact_gray = live_crop.copy()

        if self.drift_score > 0.30 and self.t - self.last_drift_alert > 4.0:
            self.add_alert(
                "TAMPER_DRIFT",
                f"Baseline silhouette SSIM drift {self.drift_score:.3f}; possible artifact movement/tamper.",
                "warning",
            )
            self.last_drift_alert = self.t

    def force_breach(self):
        with self.lock:
            self.force_start = self.t
            self.force_until = self.t + 6.0
            self.add_alert(
                "SIMULATION",
                "Manual simulated hand breach triggered.",
                "info",
            )

    def reset_baseline(self):
        with self.lock:
            self.artifact_baseline = self.latest_artifact_crop.copy()
            self.drift_score = 0.0
            self.ssim_score = 1.0
            self.add_alert(
                "BASELINE_RESET",
                "Artifact baseline re-registered from live silhouette.",
                "info",
            )

    def clear_alerts(self):
        with self.lock:
            self.alerts.clear()

    def add_alert(self, kind, message, severity="warning"):
        self.alert_seq += 1
        alert = {
            "id": self.alert_seq,
            "ts": now_iso(),
            "kind": kind,
            "message": message,
            "severity": severity,
        }
        self.alerts.appendleft(alert)

        try:
            self.redis.lpush("curatoreye:alerts", json.dumps(alert))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Redis/API status payloads
    # ------------------------------------------------------------------

    def get_status_payload(self):
        room_a = sum(1 for tr in self.tracks if tr["room"] == "A")
        room_b = len(self.tracks) - room_a

        return {
            "time": now_iso(),
            "active_visitors": len(self.tracks),
            "room_A": room_a,
            "room_B": room_b,
            "flow": dict(self.flow),
            "zone_status": "BREACH" if self.breach else ("WARNING" if self.warning else "SECURE"),
            "breach": bool(self.breach),
            "warning": bool(self.warning),
            "ssim": round(float(self.ssim_score), 4),
            "drift_score": round(float(self.drift_score), 4),
            "micro_motion": round(float(self.micro_motion), 4),
            "alerts": len(self.alerts),
            "redis_real": bool(self.redis_real),
            "privacy": "Ephemeral anonymous tokens only. No identity, face template, or cross-exhibit re-identification stored.",
        }

    def get_engagement_payload(self):
        room_a = sum(1 for tr in self.tracks if tr["room"] == "A")
        room_b = len(self.tracks) - room_a

        exhibits = []
        for exhibit in self.exhibits:
            exhibits.append(
                {
                    "exhibit": exhibit["id"],
                    "room": exhibit["room"],
                    "current_dwellers": int(exhibit["current_dwellers"]),
                    "engagement": round(float(exhibit["engagement"]), 2),
                    "avg_dwell_sec": round(float(exhibit["avg_dwell"]), 2),
                    "total_dwell_sec": round(float(exhibit["total_dwell"]), 1),
                }
            )

        return {
            "active_visitors": len(self.tracks),
            "room_A": room_a,
            "room_B": room_b,
            "flow": dict(self.flow),
            "exhibits": exhibits,
        }

    def publish_metrics(self):
        if self.t - self.last_status_push < 1.0:
            return

        self.last_status_push = self.t

        try:
            self.redis.setex(
                "curatoreye:status",
                15,
                json.dumps(self.get_status_payload()),
            )
            self.redis.setex(
                "curatoreye:analytics",
                15,
                json.dumps(self.get_engagement_payload()),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Gallery rendering
    # ------------------------------------------------------------------

    def render_gallery_background(self):
        img = np.full((GALLERY_H, GALLERY_W, 3), (238, 238, 238), dtype=np.uint8)

        # Wall and floor.
        cv2.rectangle(img, (0, 0), (GALLERY_W, 215), (247, 247, 247), -1)
        cv2.rectangle(img, (0, 215), (GALLERY_W, GALLERY_H), (216, 216, 216), -1)

        # Floor perspective lines.
        center_x = GALLERY_W / 2.0
        for i in range(0, GALLERY_W + 1, 80):
            x_top = i
            x_bottom = int(center_x + (i - center_x) * 1.3)
            cv2.line(img, (x_top, 215), (x_bottom, GALLERY_H), (205, 205, 205), 1)

        # Room divider.
        cv2.rectangle(
            img,
            (GALLERY_W // 2 - 2, 0),
            (GALLERY_W // 2 + 2, 215),
            (226, 226, 226),
            -1,
        )

        cv2.putText(
            img,
            "ROOM A",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "ROOM B",
            (GALLERY_W // 2 + 15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

        # Exhibit frames.
        for exhibit in self.exhibits:
            x, y, w, h = exhibit["rect"]

            cv2.rectangle(img, (x - 6, y - 6), (x + w + 6, y + h + 6), (190, 190, 190), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), exhibit["color"], -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 2)

            cv2.putText(
                img,
                exhibit["id"],
                (x, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )

        return img

    def draw_visitor(self, img, track):
        sway = math.sin(self.t * 2.4 + track["phase"]) * 1.7
        x = int(track["x"] + sway)
        y = int(track["y"])

        # Shadow.
        cv2.ellipse(img, (x, y + 34), (20, 7), 0, 0, 360, (190, 190, 190), -1)

        # Body silhouette.
        cv2.ellipse(img, (x, y), (16, 32), 0, 0, 360, (35, 35, 42), -1)

        # Anonymized head block: no facial details.
        cv2.rectangle(img, (x - 12, y - 58), (x + 12, y - 32), (25, 25, 28), -1)

    def detect_persons(self, gray):
        diff = cv2.absdiff(self.gallery_bg_gray, gray)
        _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        thresh = cv2.dilate(thresh, np.ones((9, 9), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 700:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h >= 55 and w >= 18 and h <= 190 and w / max(1.0, float(h)) <= 1.8:
                boxes.append((x, y, w, h))

        return boxes

    def match_detections(self, boxes):
        for track in self.tracks:
            track["box"] = None
            track["detected"] = False

        for box in boxes:
            x, y, w, h = box
            cx = x + w / 2.0
            cy = y + h / 2.0

            best = None
            best_d = 1e18

            for track in self.tracks:
                dx = track["x"] - cx
                dy = (track["y"] - 25) - cy
                d = dx * dx + dy * dy

                if d < best_d:
                    best = track
                    best_d = d

            if best is not None and best_d < 130 ** 2 and not best["detected"]:
                best["detected"] = True
                best["box"] = box

    def render_gallery(self):
        raw = self.gallery_bg.copy()

        for track in self.tracks:
            self.draw_visitor(raw, track)

        gray = cv2.GaussianBlur(
            cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY),
            (21, 21),
            0,
        )

        boxes = self.detect_persons(gray)
        self.match_detections(boxes)

        frame = raw.copy()

        # Heat-map overlay.
        heat_norm = cv2.normalize(self.heat, None, 0, 255, cv2.NORM_MINMAX)
        heat_norm = heat_norm.astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        mask = (self.heat > 0.04).astype(np.uint8) * 255
        heat_color = cv2.bitwise_and(heat_color, heat_color, mask=mask)

        frame = cv2.addWeighted(frame, 1.0, heat_color, 0.35, 0)

        # Exhibit live labels.
        for exhibit in self.exhibits:
            x, y, w, h = exhibit["rect"]
            label = f"{exhibit['id']} dwellers={exhibit['current_dwellers']}"
            cv2.putText(
                frame,
                label,
                (x, y + h + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )

        # Raw motion/person boxes.
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 195, 0), 1)

        # Anonymous visitor overlays.
        for track in self.tracks:
            if track["box"] is not None:
                x, y, w, h = track["box"]
            else:
                x = int(track["x"] - 24)
                y = int(track["y"] - 66)
                w = 48
                h = 104

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

            label = f"ANON-{track['id'][:5]} dwell {track['current_dwell']:.1f}s"
            cv2.putText(
                frame,
                label,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 110, 0),
                1,
                cv2.LINE_AA,
            )

            hx = int(track["x"])
            hy = int(track["y"] - 48)

            gx, gy = track["gaze"]
            gx = int(gx)
            gy = int(gy)

            if math.hypot(gx - hx, gy - hy) > 12:
                cv2.arrowedLine(
                    frame,
                    (hx, hy),
                    (gx, gy),
                    (255, 80, 80),
                    2,
                    tipLength=0.18,
                )

            angle = math.radians(track["yaw"])
            ex = int(hx + 30 * math.cos(angle))
            ey = int(hy + 30 * math.sin(angle) + track["pitch"] * 0.4)

            cv2.arrowedLine(
                frame,
                (hx, hy),
                (ex, ey),
                (0, 220, 255),
                2,
                tipLength=0.35,
            )

        # Header and flow.
        cv2.putText(
            frame,
            f"Gallery camera | anonymized detection | visitors={len(self.tracks)}",
            (10, GALLERY_H - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

        flow_text = (
            f"Flow A->B {self.flow['A->B']} | B->A {self.flow['B->A']} | "
            "short-horizon tokens only"
        )
        cv2.putText(
            frame,
            flow_text,
            (10, GALLERY_H - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

        self.gallery_frame = frame

    # ------------------------------------------------------------------
    # Artifact rendering
    # ------------------------------------------------------------------

    def render_artifact_background(self):
        img = np.zeros((ARTIFACT_H, ARTIFACT_W, 3), dtype=np.uint8)

        # Wall and floor.
        img[:] = (28, 26, 24)
        cv2.rectangle(img, (0, 0), (ARTIFACT_W, 320), (38, 34, 32), -1)
        cv2.rectangle(img, (0, 320), (ARTIFACT_W, ARTIFACT_H), (22, 22, 24), -1)

        # Spotlight.
        cv2.ellipse(img, (320, 180), (190, 140), 0, 0, 360, (52, 48, 42), -1)

        # Pedestal.
        cv2.rectangle(img, (250, 300), (390, 400), (70, 70, 76), -1)
        cv2.rectangle(img, (240, 290), (400, 310), (90, 90, 98), -1)

        # Glass case outline.
        cv2.rectangle(img, (235, 105), (405, 310), (110, 110, 110), 2)

        cv2.putText(
            img,
            "PROTECTED ARTIFACT - DEPTH CAMERA 04",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        return img

    def draw_hand(self, img):
        if not self.hand_active:
            return

        x = int(self.hand_pos[0])
        y = int(self.hand_pos[1])

        # Arm.
        cv2.line(img, (ARTIFACT_W + 70, y + 80), (x + 28, y + 14), (95, 75, 65), 22)

        # Palm.
        cv2.circle(img, (x, y), 26, (130, 100, 85), -1)

        # Fingers.
        for dy in (-18, -6, 6, 18):
            cv2.line(img, (x - 6, y + dy), (x - 38, y + dy - 6), (130, 100, 85), 7)

        cv2.putText(
            img,
            "simulated hand",
            (max(10, x - 85), max(20, y - 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    def draw_depth_pip(self, frame):
        depth = np.zeros((90, 120), dtype=np.uint8)
        depth[:] = 22

        # Artifact depth blob.
        cv2.ellipse(depth, (60, 45), (20, 30), 0, 0, 360, 110, -1)

        if self.hand_active:
            hx = int(self.hand_pos[0] * 120 / ARTIFACT_W)
            hy = int(self.hand_pos[1] * 90 / ARTIFACT_H)
            cv2.circle(depth, (hx, hy), 12, 240, -1)

        color = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)

        x0 = ARTIFACT_W - 140
        y0 = 15

        frame[y0:y0 + 90, x0:x0 + 120] = color
        cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + 121, y0 + 91), (255, 255, 255), 1)

        cv2.putText(
            frame,
            "DEPTH SIM",
            (x0, y0 + 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def render_artifact(self):
        frame = self.artifact_bg.copy()

        angle, dx, dy = self.artifact_pose
        sprite, mask = self.rotate_sprite(angle, dx, dy)
        self.paste_sprite(frame, sprite, mask)

        self.draw_hand(frame)

        # Protection zone overlay.
        if self.breach:
            zone_color = (0, 0, 255) if int(self.t * 6) % 2 == 0 else (70, 70, 255)
            alpha = 0.32
        elif self.warning:
            zone_color = (0, 190, 255)
            alpha = 0.22
        else:
            zone_color = (0, 210, 0)
            alpha = 0.16

        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.zone], zone_color)
        frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

        cv2.polylines(frame, [self.zone], True, zone_color, 2)

        status = "BREACH" if self.breach else ("WARNING" if self.warning else "SECURE")
        status_color = (0, 0, 255) if self.breach else (
            (0, 190, 255) if self.warning else (0, 220, 0)
        )

        cv2.putText(
            frame,
            f"ZONE {status}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"SSIM drift {self.drift_score:.3f} | SSIM {self.ssim_score:.3f}",
            (15, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"micro-motion {self.micro_motion:.3f}",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        if self.breach and int(self.t * 5) % 2 == 0:
            cv2.putText(
                frame,
                "PROXIMITY ALERT",
                (135, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.05,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        self.draw_depth_pip(frame)

        self.artifact_frame = frame


# ----------------------------------------------------------------------------
# FastAPI backend
# ----------------------------------------------------------------------------

def is_port_free(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def create_api(engine: CuratorEngine):
    app = FastAPI(title="CuratorEye Backend", version="1.0")

    @app.get("/")
    def root():
        html = """
        <!doctype html>
        <html>
          <head>
            <title>CuratorEye Live</title>
          </head>
          <body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
            <h2 style="padding:12px;margin:0">CuratorEye Live Camera Feeds</h2>

            <div style="display:flex;gap:12px;flex-wrap:wrap;padding:12px">
              <div style="flex:1;min-width:360px">
                <h3>Gallery camera</h3>
                <img
                  src="/stream/gallery"
                  alt="Gallery live feed"
                  style="width:100%;border:1px solid #444;border-radius:8px"
                >
              </div>

              <div style="flex:1;min-width:360px">
                <h3>Protected artifact camera</h3>
                <img
                  src="/stream/artifact"
                  alt="Artifact live feed"
                  style="width:100%;border:1px solid #444;border-radius:8px"
                >
              </div>
            </div>

            <p style="padding:0 12px">
              <a href="/docs" style="color:#7dd3fc">Open API docs</a>
            </p>
          </body>
        </html>
        """
        return Response(content=html, media_type="text/html")

    @app.get("/health")
    def health():
        with engine.lock:
            return engine.get_status_payload()

    @app.get("/analytics/engagement")
    def analytics_engagement():
        with engine.lock:
            return engine.get_engagement_payload()

    @app.get("/security/status")
    def security_status():
        with engine.lock:
            return engine.get_status_payload()

    @app.get("/security/alerts")
    def security_alerts():
        with engine.lock:
            return {"alerts": list(engine.alerts)[:50]}

    @app.get("/frames/gallery.jpeg")
    def gallery_jpeg():
        with engine.lock:
            frame = engine.gallery_frame.copy()
        return Response(content=jpeg_bytes(frame), media_type="image/jpeg")

    @app.get("/frames/artifact.jpeg")
    def artifact_jpeg():
        with engine.lock:
            frame = engine.artifact_frame.copy()
        return Response(content=jpeg_bytes(frame), media_type="image/jpeg")

    def mjpeg_generator(frame_getter):
        while True:
            with engine.lock:
                frame = frame_getter(engine).copy()

            jpg = jpeg_bytes(frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(1.0 / FPS)

    @app.get("/stream/gallery")
    def stream_gallery():
        return StreamingResponse(
            mjpeg_generator(lambda e: e.gallery_frame),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/artifact")
    def stream_artifact():
        return StreamingResponse(
            mjpeg_generator(lambda e: e.artifact_frame),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


_RUNNING_SERVERS = []


def start_fastapi_thread(engine: CuratorEngine):
    if not FASTAPI_AVAILABLE:
        return None

    app = create_api(engine)

    for port in range(8765, 8770):
        if not is_port_free(port):
            continue

        try:
            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="error",
            )
            server = uvicorn.Server(config)
            _RUNNING_SERVERS.append(server)

            threading.Thread(target=server.run, daemon=True).start()
            time.sleep(0.35)
            return port
        except Exception:
            continue

    return None


# ----------------------------------------------------------------------------
# ASGI entrypoint
# Fixes: Error loading ASGI app. Attribute "app" not found in module "app".
# ----------------------------------------------------------------------------

_CURATOREYE_IN_STREAMLIT = False

if os.getenv("STREAMLIT_SERVER_PORT") or os.getenv("STREAMLIT_RUNTIME_SEGMENT"):
    _CURATOREYE_IN_STREAMLIT = True

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        _CURATOREYE_IN_STREAMLIT = True
except Exception:
    pass

_CURATOREYE_WANT_ASGI = (__name__ != "__main__") and (not _CURATOREYE_IN_STREAMLIT)

if _CURATOREYE_WANT_ASGI and FASTAPI_AVAILABLE:
    _ASGI_ENGINE = CuratorEngine()
    _ASGI_ENGINE.start()
    app = create_api(_ASGI_ENGINE)
    application = app

elif _CURATOREYE_WANT_ASGI:
    async def app(scope, receive, send):
        if scope["type"] == "http":
            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"FastAPI is not installed. Run: pip install -r requirements.txt",
                }
            )

    application = app

else:
    app = None
    application = None


# ----------------------------------------------------------------------------
# Streamlit runtime
# ----------------------------------------------------------------------------

@st.cache_resource
def init_runtime():
    engine = CuratorEngine()
    engine.start()

    api_port = start_fastapi_thread(engine)

    return {
        "engine": engine,
        "api_port": api_port,
    }


def render_live_demo(engine: CuratorEngine):
    st.subheader("Live demo — gallery camera + protected artifact camera")

    col1, col2 = st.columns(2)

    with engine.lock:
        gallery_frame = engine.gallery_frame.copy()
        artifact_frame = engine.artifact_frame.copy()
        status = engine.get_status_payload()

    with col1:
        st.image(
            to_rgb(gallery_frame),
            caption="Live gallery camera: anonymized visitors, gaze/head-pose, dwell heat-map",
            use_container_width=True,
        )

    with col2:
        st.image(
            to_rgb(artifact_frame),
            caption="Live protected artifact camera: proximity zone, depth sim, SSIM drift",
            use_container_width=True,
        )

    if status["breach"]:
        st.error("PROXIMITY ZONE BREACH — simulated hand entered the protected artifact zone.")
    elif status["warning"]:
        st.warning("Approach warning: simulated hand is near the protected artifact zone.")
    else:
        st.caption("Looping simulated live feeds. A simulated hand enters periodically.")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Active anonymous visitors", status["active_visitors"])
    m2.metric("Room A / Room B", f"{status['room_A']} / {status['room_B']}")
    m3.metric("Artifact zone", status["zone_status"])
    m4.metric("SSIM drift", f"{status['drift_score']:.3f}")


def render_curatorial(engine: CuratorEngine):
    st.subheader("Curatorial analytics dashboard")
    st.caption(
        "Anonymous per-exhibit engagement heat-map and short-horizon room flow. "
        "No identity, face template, or persistent visitor profile is stored."
    )

    with engine.lock:
        gallery_frame = engine.gallery_frame.copy()
        engagement = engine.get_engagement_payload()

    st.image(
        to_rgb(gallery_frame),
        caption="Live gallery camera with gaze/dwell overlays",
        use_container_width=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Active anonymous visitors", engagement["active_visitors"])
    c2.metric("Room A", engagement["room_A"])
    c3.metric("Room B", engagement["room_B"])
    c4.metric("A->B / B->A", f"{engagement['flow']['A->B']} / {engagement['flow']['B->A']}")

    st.write("### Exhibit engagement")

    if pd is not None:
        df = pd.DataFrame(engagement["exhibits"])
        st.dataframe(df, use_container_width=True)

        st.write("### Engagement score")
        st.bar_chart(df.set_index("exhibit")["engagement"])
    else:
        st.json(engagement)

    st.write("**Visitor flow:**", engagement["flow"])


def render_security(engine: CuratorEngine):
    st.subheader("Security console")

    with engine.lock:
        artifact_frame = engine.artifact_frame.copy()
        status = engine.get_status_payload()
        alerts = list(engine.alerts)[:15]

    if not st.session_state.security_unlocked:
        st.warning(
            "Locked-down security console. Live preview is blurred. "
            f"Unlock with the demo PIN in the sidebar. Demo PIN: {SECURITY_PIN}"
        )

        blurred = cv2.GaussianBlur(artifact_frame, (31, 31), 0)
        st.image(
            to_rgb(blurred),
            caption="Locked live artifact feed",
            use_container_width=True,
        )
        return

    if status["breach"]:
        st.error("PROXIMITY ZONE BREACH — instant alert fired in security console.")
    elif status["warning"]:
        st.warning("Approach warning: object near protection zone.")
    else:
        st.success("Protection zone secure.")

    st.image(
        to_rgb(artifact_frame),
        caption="Live protected-artifact depth camera with proximity and tamper overlays",
        use_container_width=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Zone", status["zone_status"])
    c2.metric("SSIM", f"{status['ssim']:.3f}")
    c3.metric("Drift", f"{status['drift_score']:.3f}")
    c4.metric("Micro-motion", f"{status['micro_motion']:.3f}")

    st.write("### Console controls")

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("Simulate hand breach now"):
            engine.force_breach()

    with b2:
        if st.button("Reset baseline"):
            engine.reset_baseline()

    with b3:
        if st.button("Clear alerts"):
            engine.clear_alerts()

    st.write("### Alert feed")

    if alerts:
        for alert in alerts:
            text = f"{alert['ts']} | {alert['kind']} | {alert['message']}"

            if alert["severity"] == "critical":
                st.error(text)
            elif alert["severity"] == "warning":
                st.warning(text)
            else:
                st.info(text)
    else:
        st.caption("No alerts.")


def main():
    st.set_page_config(
        page_title="CuratorEye",
        page_icon="🖼️",
        layout="wide",
    )

    runtime = init_runtime()
    engine: CuratorEngine = runtime["engine"]
    api_port = runtime["api_port"]

    if "security_unlocked" not in st.session_state:
        st.session_state.security_unlocked = False

    if "live" not in st.session_state:
        st.session_state.live = True

    if "refresh" not in st.session_state:
        st.session_state.refresh = 0.12

    st.title("CuratorEye")
    st.caption(
        "Museum/gallery visitor engagement + artifact security demo. "
        "Live simulated camera feeds, anonymized analytics, Redis-backed status, FastAPI endpoints."
    )

    if engine.last_error:
        st.error(f"Engine error: {engine.last_error}")

    with st.sidebar:
        st.header("Control panel")

        st.session_state.live = st.checkbox("Live auto-refresh", value=True)
        st.session_state.refresh = st.slider(
            "Refresh interval seconds",
            min_value=0.05,
            max_value=1.0,
            value=float(st.session_state.refresh),
            step=0.01,
        )

        view = st.radio(
            "Frontend",
            [
                "Live Demo",
                "Curatorial Analytics",
                "Security Console",
            ],
        )

        st.divider()

        if not st.session_state.security_unlocked:
            pin = st.text_input(
                "Security console PIN",
                type="password",
                placeholder=SECURITY_PIN,
            )

            if st.button("Unlock security console"):
                if pin == SECURITY_PIN:
                    st.session_state.security_unlocked = True
                    st.rerun()
                else:
                    st.error("Invalid PIN.")

            st.caption(f"Demo PIN: {SECURITY_PIN}")
        else:
            st.success("Security console unlocked.")

            if st.button("Lock console"):
                st.session_state.security_unlocked = False
                st.rerun()

        st.divider()

        st.caption("Privacy mode: ephemeral tokens only. No identity storage.")
        st.caption(f"Redis: {'real Redis connected' if engine.redis_real else 'in-memory simulated Redis'}")

        if api_port:
            st.caption(f"FastAPI backend: http://127.0.0.1:{api_port}/docs")
            st.caption(
                "Endpoints: /health, /analytics/engagement, /security/status, "
                "/frames/gallery.jpeg, /frames/artifact.jpeg, /stream/gallery, /stream/artifact"
            )
        else:
            st.caption("FastAPI backend not started.")

    if view == "Live Demo":
        render_live_demo(engine)
    elif view == "Curatorial Analytics":
        render_curatorial(engine)
    else:
        render_security(engine)

    if st.session_state.live:
        time.sleep(st.session_state.refresh)
        st.rerun()


if _CURATOREYE_IN_STREAMLIT or __name__ == "__main__":
    main()