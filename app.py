import math
import time
from collections import deque

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="CuratorEye",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helper utilities
# -----------------------------
W, H = 960, 540
GALLERY_AREAS = [
    {"name": "Impressionism", "x1": 70, "y1": 120, "x2": 250, "y2": 250},
    {"name": "Sculpture", "x1": 340, "y1": 95, "x2": 540, "y2": 270},
    {"name": "Modern Wing", "x1": 650, "y1": 130, "x2": 860, "y2": 280},
]

ROOMS = ["Lobby", "Impressionism", "Sculpture", "Modern Wing", "Exit"]

if "t0" not in st.session_state:
    st.session_state.t0 = time.time()
if "visitor_tracks" not in st.session_state:
    st.session_state.visitor_tracks = []
if "alerts" not in st.session_state:
    st.session_state.alerts = deque(maxlen=12)
if "room_flow" not in st.session_state:
    st.session_state.room_flow = {r: 0 for r in ROOMS}
if "dwell" not in st.session_state:
    st.session_state.dwell = {a["name"]: 0 for a in GALLERY_AREAS}
if "security_breach" not in st.session_state:
    st.session_state.security_breach = False
if "tamper_score" not in st.session_state:
    st.session_state.tamper_score = 0.0
if "last_frame_time" not in st.session_state:
    st.session_state.last_frame_time = 0.0


def background_frame(kind="gallery", t=0.0):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if kind == "gallery":
        img[:] = (18, 20, 24)
        # floor
        cv2.rectangle(img, (0, 380), (W, H), (28, 30, 35), -1)
        # walls / frames
        for a in GALLERY_AREAS:
            cv2.rectangle(img, (a["x1"], a["y1"]), (a["x2"], a["y2"]), (80, 62, 35), 3)
            cv2.rectangle(img, (a["x1"] + 8, a["y1"] + 8), (a["x2"] - 8, a["y2"] - 8), (40, 40, 48), -1)
        # light spots
        for i in range(7):
            x = int(80 + i * 125 + 18 * math.sin(t * 0.8 + i))
            y = int(60 + 10 * math.cos(t * 0.9 + i))
            cv2.circle(img, (x, y), 30, (50, 48, 40), -1)
    else:
        img[:] = (12, 12, 14)
        cv2.rectangle(img, (120, 100), (840, 470), (35, 35, 40), 2)
        cv2.rectangle(img, (290, 180), (670, 420), (55, 55, 65), 2)
        cv2.rectangle(img, (360, 240), (600, 360), (80, 80, 95), -1)
        cv2.rectangle(img, (390, 270), (570, 335), (20, 24, 28), -1)
        # stand / pedestal
        cv2.rectangle(img, (450, 360), (510, 455), (90, 90, 100), -1)
    return img


def add_text(img, text, org, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def alpha_heatmap_overlay(frame, center_points, weights, color=(0, 180, 255)):
    overlay = frame.copy()
    hm = np.zeros((H, W), dtype=np.float32)
    for (x, y), w in zip(center_points, weights):
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(hm, (x, y), 55, float(w), -1)
    hm = cv2.GaussianBlur(hm, (0, 0), 18)
    if hm.max() > 0:
        hm = hm / hm.max()
    c = np.zeros_like(frame, dtype=np.uint8)
    c[:, :, 0] = int(color[0])
    c[:, :, 1] = int(color[1])
    c[:, :, 2] = int(color[2])

    heat = (hm[..., None] * c).astype(np.uint8)
    return cv2.addWeighted(frame, 0.80, heat, 0.55, 0.0)


def draw_person(frame, x, y, gaze_angle, active=False):
    # anonymized stick-person silhouette
    body_color = (220, 220, 230) if active else (170, 170, 180)
    head_color = (240, 240, 245) if active else (185, 185, 195)
    cv2.circle(frame, (x, y - 32), 11, head_color, -1)
    cv2.line(frame, (x, y - 20), (x, y + 28), body_color, 3)
    cv2.line(frame, (x, y - 5), (x - 15, y + 12), body_color, 3)
    cv2.line(frame, (x, y - 5), (x + 15, y + 12), body_color, 3)
    cv2.line(frame, (x, y + 28), (x - 14, y + 46), body_color, 3)
    cv2.line(frame, (x, y + 28), (x + 14, y + 46), body_color, 3)

    gx = int(x + 30 * math.cos(gaze_angle))
    gy = int(y - 32 + 30 * math.sin(gaze_angle))
    cv2.arrowedLine(frame, (x, y - 32), (gx, gy), (0, 255, 255), 2, tipLength=0.25)


def generate_gallery_scene(t):
    frame = background_frame("gallery", t)
    np.random.seed(int(t * 10) % 9999)

    visitors = []
    heat_points = []
    weights = []

    for i in range(4):
        phase = t * 0.75 + i * 1.1
        x = int(120 + i * 190 + 42 * math.sin(phase))
        y = int(335 + 18 * math.cos(phase * 1.1))
        gaze = -0.6 + 1.1 * math.sin(phase * 0.7)

        active = (i % 2 == 0) or (math.sin(phase) > 0.3)
        draw_person(frame, x, y, gaze, active=active)

        # simulated gaze endpoint
        ex = int(x + 130 * math.cos(gaze))
        ey = int((y - 32) + 130 * math.sin(gaze))
        cv2.circle(frame, (ex, ey), 5, (0, 255, 255), -1)
        cv2.line(frame, (x, y - 32), (ex, ey), (0, 220, 255), 2)

        visitors.append((x, y, gaze))
        heat_points.append((ex, ey))
        weights.append(1.0 + 0.2 * (i % 3))

        # dwell accumulation by nearest exhibit
        for a in GALLERY_AREAS:
            if a["x1"] <= ex <= a["x2"] and a["y1"] <= ey <= a["y2"]:
                st.session_state.dwell[a["name"]] += 1

    # heat-map overlay
    frame = alpha_heatmap_overlay(frame, heat_points, weights, color=(0, 200, 255))

    # exhibit labels and anonymized analytics
    for a in GALLERY_AREAS:
        cv2.rectangle(frame, (a["x1"], a["y1"]), (a["x2"], a["y2"]), (130, 110, 70), 2)
        add_text(frame, a["name"], (a["x1"], a["y1"] - 10), 0.6, (220, 220, 180), 2)

    # simulate room flow tracking without identity retention
    room_idx = int((t * 0.7) % len(ROOMS))
    st.session_state.room_flow[ROOMS[room_idx]] = int(2 + 3 * abs(math.sin(t * 0.9 + 1.5)))

    add_text(frame, "ANONYMIZED GALLERY ENGAGEMENT FEED", (24, 34), 0.8, (255, 255, 255), 2)
    add_text(frame, "No face IDs stored | short-horizon flow only", (24, 62), 0.6, (180, 200, 220), 2)

    return frame


def generate_artifact_scene(t):
    frame = background_frame("artifact", t)
    hand_x = int(145 + 650 * (0.5 + 0.5 * math.sin(t * 0.9)))
    hand_y = int(235 + 70 * math.sin(t * 1.8))
    # Base artifact silhouette jitter to simulate live motion
    drift = 3 * math.sin(t * 0.8)
    cv2.ellipse(frame, (480, 300), (78 + int(abs(drift)), 48), int(4 * drift), 0, 360, (100, 100, 115), -1)
    cv2.ellipse(frame, (480, 300), (72, 42), 0, 0, 360, (170, 170, 185), 2)
    cv2.rectangle(frame, (460, 260), (500, 340), (90, 90, 100), -1)

    # protection zone
    zone_center = (480, 300)
    zone_axes = (170, 120)
    breach = (abs(hand_x - zone_center[0]) < zone_axes[0] and abs(hand_y - zone_center[1]) < zone_axes[1])

    zone_color = (0, 0, 255) if breach else (0, 200, 255)
    thickness = 4 if breach else 2
    cv2.ellipse(frame, zone_center, zone_axes, 0, 0, 360, zone_color, thickness)

    # moving hand
    cv2.circle(frame, (hand_x, hand_y), 16, (120, 95, 70), -1)
    cv2.rectangle(frame, (hand_x - 10, hand_y + 10), (hand_x + 10, hand_y + 38), (100, 80, 60), -1)
    cv2.line(frame, (hand_x - 12, hand_y + 20), (hand_x - 34, hand_y + 12), (120, 95, 70), 7)

    # SSIM-like drift proxy (baseline silhouette score)
    # Higher when hand intrudes or artifact changes more strongly.
    tamper_score = 0.08 + (0.75 if breach else 0.0) + 0.05 * abs(math.sin(t * 2.3))
    st.session_state.tamper_score = min(1.0, 0.82 * st.session_state.tamper_score + 0.18 * tamper_score)

    if breach:
        st.session_state.security_breach = True
        st.session_state.alerts.appendleft(("ZONE BREACH", "Instant proximity alert: hand entered protection zone"))
    elif st.session_state.security_breach and not breach:
        st.session_state.security_breach = False

    if st.session_state.tamper_score > 0.42:
        st.session_state.alerts.appendleft(("TAMPER DRIFT", f"SSIM drift elevated: {st.session_state.tamper_score:.2f}"))

    # live overlays
    add_text(frame, "PROTECTED ARTIFACT LIVE FEED", (24, 34), 0.8, (255, 255, 255), 2)
    add_text(frame, f"Tamper drift score: {st.session_state.tamper_score:.2f}", (24, 62), 0.65, (255, 215, 90), 2)
    add_text(frame, "Locked-down security console stream", (24, 90), 0.55, (180, 190, 210), 2)
    if breach:
        cv2.rectangle(frame, (300, 20), (660, 105), (0, 0, 255), -1)
        add_text(frame, "ALERT: PROTECTION ZONE BREACHED", (322, 72), 0.8, (255, 255, 255), 2)

    return frame


def to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# -----------------------------
# UI
# -----------------------------
st.title("CuratorEye")
st.caption("Museum visitor engagement analytics and artifact security demo")

with st.sidebar:
    st.header("Controls")
    mode = st.radio(
        "View",
        ["Analytics Dashboard", "Security Console", "Both"],
        index=2,
    )
    st.markdown("---")
    st.write("This demo uses simulated looping video and anonymized overlays.")
    st.write("Identity is never stored or re-used across exhibits.")

# live refresh
placeholder = st.empty()
t = time.time() - st.session_state.t0

gallery = generate_gallery_scene(t)
artifact = generate_artifact_scene(t)

if mode == "Analytics Dashboard":
    cols = st.columns([1.55, 1.0])
    with cols[0]:
        st.image(to_rgb(gallery), use_container_width=True)
    with cols[1]:
        st.subheader("Curatorial Analytics")
        st.metric("Current dwell pressure", f"{sum(st.session_state.dwell.values())}")
        for a in GALLERY_AREAS:
            st.progress(min(1.0, st.session_state.dwell[a["name"]] / 60.0))
            st.write(f"**{a['name']}** — dwell score: {st.session_state.dwell[a['name']]}")
        st.markdown("### Short-horizon room flow")
        for room, count in st.session_state.room_flow.items():
            st.write(f"- {room}: {count}")
        st.markdown("### Latest privacy-preserving events")
        if st.session_state.alerts:
            for kind, msg in list(st.session_state.alerts)[:6]:
                st.write(f"- {kind}: {msg}")
        else:
            st.write("No active alerts.")

elif mode == "Security Console":
    cols = st.columns([1.45, 1.0])
    with cols[0]:
        st.image(to_rgb(artifact), use_container_width=True)
    with cols[1]:
        st.subheader("Security Console")
        st.metric("Live tamper drift", f"{st.session_state.tamper_score:.2f}")
        st.metric("Breach state", "ACTIVE" if st.session_state.security_breach else "CLEAR")
        st.markdown("### Alerts")
        if st.session_state.alerts:
            for kind, msg in list(st.session_state.alerts)[:8]:
                st.error(f"{kind}: {msg}")
        else:
            st.success("No active security alerts.")
        st.markdown("### Protection logic")
        st.write("- Proximity zone rendered around artifact")
        st.write("- Micro-motion / drift scoring simulated live")
        st.write("- Alert fires instantly on zone intrusion")

else:
    left, right = st.columns(2)
    with left:
        st.subheader("Curatorial Analytics Dashboard")
        st.image(to_rgb(gallery), use_container_width=True)
    with right:
        st.subheader("Locked-down Security Console")
        st.image(to_rgb(artifact), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Dwell total", f"{sum(st.session_state.dwell.values())}")
    c2.metric("Tamper drift", f"{st.session_state.tamper_score:.2f}")
    c3.metric("Security", "BREACH" if st.session_state.security_breach else "MONITORING")

    c4, c5 = st.columns(2)
    with c4:
        st.markdown("### Exhibit dwell")
        for a in GALLERY_AREAS:
            st.write(f"{a['name']}: {st.session_state.dwell[a['name']]}")
    with c5:
        st.markdown("### Live alerts")
        if st.session_state.alerts:
            for kind, msg in list(st.session_state.alerts)[:8]:
                st.error(f"{kind}: {msg}")
        else:
            st.info("No active alerts.")

# auto-refresh for live demo feel
time.sleep(0.05)
st.rerun()