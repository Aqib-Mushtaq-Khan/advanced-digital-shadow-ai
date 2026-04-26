"""
Advanced Digital Shadow System v4.0 — Clean Holographic AI Interface
Real-time face tracking + hand/finger tracking + clean JARVIS-style hologram workspace

Main design goal:
- One clean central AI object at a time
- One fixed AI analysis panel on the right
- No overlapping annotation cards
- Minimal particles, controlled glow, and clean geometry
- Pinch / palm / dual-hand gestures create and manipulate advanced holograms

Install:
    pip install opencv-python mediapipe==0.10.14 pygame numpy

Run:
    python advanced_digital_shadow_system.py

Gestures:
    Hold pinch thumb + index finger  -> create polygon hologram
    Hold open palm                   -> create cube hologram
    Show both hands open             -> create AI mesh hologram
    Show both hands after object     -> scale + rotate active object
    Index finger                     -> clean targeting laser

Controls:
    Q / ESC  -> quit
    R        -> reset active object and particles
    H        -> hide/show HUD
    M        -> mirror camera on/off
    C        -> toggle clean/full glow mode
    1        -> force polygon
    2        -> force cube
    3        -> force AI mesh
"""

import math
import random
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
import pygame


# ============================================================
# CONFIGURATION
# ============================================================
WIDTH = 1280
HEIGHT = 720
FPS_TARGET = 60
CAMERA_ID = 0

CAMERA_PANEL_WIDTH = 315
RIGHT_PANEL_WIDTH = 330
WORKSPACE_LEFT = CAMERA_PANEL_WIDTH
WORKSPACE_RIGHT = WIDTH - RIGHT_PANEL_WIDTH
WORKSPACE_CENTER = np.array([
    (WORKSPACE_LEFT + WORKSPACE_RIGHT) / 2,
    HEIGHT * 0.52,
], dtype=np.float32)

# Clean render settings
MAX_PARTICLES = 650
PARTICLE_FADE = 4.0
BACKGROUND_ALPHA = 72
BASE_HAND_PARTICLES = 1
BURST_PARTICLES = 26

HUD_ENABLED = True
MIRROR_CAMERA = True
CLEAN_MODE = True

# Gesture thresholds
PINCH_THRESHOLD = 38
OPEN_PALM_SPREAD = 132
GESTURE_HOLD_TIME = 0.55
GESTURE_COOLDOWN = 1.05
TWO_HAND_MESH_HOLD = 0.80

# Colors
BLACK = (0, 0, 0)
CYAN = (80, 220, 255)
BLUE = (40, 120, 255)
WHITE = (235, 245, 255)
GOLD = (255, 190, 85)
GREEN = (75, 255, 150)
RED = (255, 80, 80)
PURPLE = (170, 95, 255)
PINK = (255, 90, 185)
PANEL_BG = (5, 14, 22)
GRID = (4, 22, 32)
SOFT_GRID = (3, 14, 22)


# ============================================================
# DATA MODELS
# ============================================================
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: tuple
    glow: bool = False

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.982
        self.vy *= 0.982
        self.life -= PARTICLE_FADE * dt

    @property
    def alive(self):
        return self.life > 0

    @property
    def alpha(self):
        return max(0, min(255, int(255 * (self.life / self.max_life))))


@dataclass
class HologramObject:
    kind: str
    x: float
    y: float
    size: float
    rotation: float = 0.0
    scale: float = 1.0
    color: tuple = CYAN
    sides: int = 6
    object_id: str = "AI_OBJ_001"
    gesture: str = "UNKNOWN"
    confidence: int = 95
    state: str = "STABLE"
    created_at: float = field(default_factory=time.time)
    pulse: float = 0.0

    def update(self, dt: float):
        self.pulse += dt


@dataclass
class TrackedHand:
    label: str
    points: list
    wrist: np.ndarray
    index_tip: np.ndarray
    thumb_tip: np.ndarray
    middle_tip: np.ndarray
    palm_center: np.ndarray
    velocity: np.ndarray
    pinch: bool
    open_palm: bool
    spread: float


# ============================================================
# MAIN SYSTEM
# ============================================================
class AdvancedDigitalShadow:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Advanced Digital Shadow v4 — Clean Holographic AI Interface")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_tiny = pygame.font.SysFont("consolas", 13)
        self.font_small = pygame.font.SysFont("consolas", 16)
        self.font_medium = pygame.font.SysFont("consolas", 22)
        self.font_large = pygame.font.SysFont("consolas", 34)

        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.55,
        )

        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.62,
            min_tracking_confidence=0.62,
        )

        self.particles = []
        self.active_object = None
        self.object_counter = 0

        self.face_found = False
        self.smooth_face_center = WORKSPACE_CENTER.copy()
        self.prev_face_center = self.smooth_face_center.copy()
        self.face_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.face_size = 130

        self.prev_hands = {}
        self.tracked_hands = []
        self.last_object_time = 0
        self.last_burst_time = 0
        self.last_gesture = "IDLE"
        self.gesture_start = {}
        self.preview_kind = None
        self.preview_progress = 0.0
        self.clean_mode = CLEAN_MODE

        self.show_hud = HUD_ENABLED
        self.mirror = MIRROR_CAMERA
        self.running = True
        self.start_time = time.time()
        self.last_frame = time.time()

    # --------------------------------------------------------
    # BASIC HELPERS
    # --------------------------------------------------------
    def text(self, text, x, y, font=None, color=WHITE):
        if font is None:
            font = self.font_small
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))

    def cam_to_screen(self, x, y):
        sx = WORKSPACE_LEFT + 45 + (x / 640.0) * (WORKSPACE_RIGHT - WORKSPACE_LEFT - 90)
        sy = 45 + (y / 480.0) * (HEIGHT - 90)
        return np.array([sx, sy], dtype=np.float32)

    def draw_transparent_circle(self, color, x, y, radius, alpha, width=0):
        if radius <= 0:
            return
        s = pygame.Surface((radius * 2 + 6, radius * 2 + 6), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (radius + 3, radius + 3), radius, width)
        self.screen.blit(s, (x - radius - 3, y - radius - 3))

    def draw_glow_line(self, a, b, color, width=1, alpha=105):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        if not self.clean_mode:
            pygame.draw.line(overlay, (*color, int(alpha * 0.16)), (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), width + 8)
        pygame.draw.line(overlay, (*color, int(alpha * 0.30)), (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), width + 3)
        pygame.draw.line(overlay, (*color, alpha), (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), width)
        self.screen.blit(overlay, (0, 0))

    def polygon_points(self, x, y, radius, sides, rotation=0):
        return [
            (
                int(x + math.cos(rotation + i * math.tau / sides) * radius),
                int(y + math.sin(rotation + i * math.tau / sides) * radius),
            )
            for i in range(sides)
        ]

    # --------------------------------------------------------
    # CAMERA + TRACKING
    # --------------------------------------------------------
    def read_tracking(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, False, None, self.face_size, []

        if self.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detector.process(rgb)
        hand_results = self.hands_detector.process(rgb)

        detected_face = False
        face_center_cam = None
        face_size = self.face_size

        if face_results.detections:
            detection = face_results.detections[0]
            box = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = box.xmin * w
            y = box.ymin * h
            bw = box.width * w
            bh = box.height * h
            cx = x + bw / 2
            cy = y + bh / 2
            detected_face = True
            face_center_cam = np.array([cx, cy], dtype=np.float32)
            face_size = max(60, min(220, int((bw + bh) * 0.5)))

            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (80, 220, 255), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (80, 255, 160), -1)
            cv2.putText(frame, "FACE LOCK", (int(x), max(20, int(y) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 255), 1)

        hands = []
        if hand_results.multi_hand_landmarks:
            handedness = hand_results.multi_handedness or []
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                label = "Hand"
                if idx < len(handedness):
                    label = handedness[idx].classification[0].label

                points_cam = []
                points_screen = []
                for lm in hand_landmarks.landmark:
                    px = lm.x * 640
                    py = lm.y * 480
                    points_cam.append((px, py))
                    points_screen.append(self.cam_to_screen(px, py))

                # Draw clean mini-hand in camera preview only
                for a, b in self.mp_hands.HAND_CONNECTIONS:
                    ax, ay = points_cam[a]
                    bx, by = points_cam[b]
                    cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (70, 210, 255), 1)
                for px, py in points_cam:
                    cv2.circle(frame, (int(px), int(py)), 2, (255, 190, 85), -1)

                wrist = points_screen[0]
                thumb_tip = points_screen[4]
                index_tip = points_screen[8]
                middle_tip = points_screen[12]
                palm_center = np.mean([points_screen[i] for i in [0, 5, 9, 13, 17]], axis=0)

                prev = self.prev_hands.get(label, palm_center)
                velocity = palm_center - prev
                self.prev_hands[label] = palm_center.copy()

                pinch_dist = float(np.linalg.norm(index_tip - thumb_tip))
                pinch = pinch_dist < PINCH_THRESHOLD

                fingertip_ids = [4, 8, 12, 16, 20]
                spread = float(np.mean([np.linalg.norm(points_screen[i] - palm_center) for i in fingertip_ids]))
                open_palm = spread > OPEN_PALM_SPREAD and not pinch

                hands.append(
                    TrackedHand(
                        label=label,
                        points=points_screen,
                        wrist=wrist,
                        index_tip=index_tip,
                        thumb_tip=thumb_tip,
                        middle_tip=middle_tip,
                        palm_center=palm_center,
                        velocity=velocity,
                        pinch=pinch,
                        open_palm=open_palm,
                        spread=spread,
                    )
                )

        return frame, detected_face, face_center_cam, face_size, hands

    def update_face(self, detected, face_center_cam, face_size):
        self.face_found = detected
        self.prev_face_center = self.smooth_face_center.copy()

        if detected and face_center_cam is not None:
            target = self.cam_to_screen(face_center_cam[0], face_center_cam[1])
            self.smooth_face_center = self.smooth_face_center * 0.90 + target * 0.10
            self.face_size = int(self.face_size * 0.92 + face_size * 0.08)
        else:
            self.smooth_face_center = self.smooth_face_center * 0.98 + WORKSPACE_CENTER * 0.02

        self.face_velocity = self.smooth_face_center - self.prev_face_center

    # --------------------------------------------------------
    # PARTICLES — SUBTLE ONLY
    # --------------------------------------------------------
    def spawn_particle(self, x, y, color=CYAN, power=1.0, burst=False):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(18, 95) * power
        life = random.uniform(0.45, 1.20) if burst else random.uniform(0.55, 1.65)
        size = random.uniform(1.0, 2.8) if burst else random.uniform(0.7, 1.8)
        self.particles.append(
            Particle(
                x=x,
                y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=life,
                max_life=life,
                size=size,
                color=color,
                glow=False,
            )
        )

    def burst_at(self, x, y, color=GOLD):
        for _ in range(BURST_PARTICLES):
            self.spawn_particle(x + random.gauss(0, 4), y + random.gauss(0, 4), color=color, power=1.05, burst=True)

    def update_particles(self, dt):
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive]
        if len(self.particles) > MAX_PARTICLES:
            self.particles = self.particles[-MAX_PARTICLES:]

    # --------------------------------------------------------
    # ACTIVE OBJECT MANAGEMENT
    # --------------------------------------------------------
    def set_active_object(self, kind, gesture="MANUAL"):
        now = time.time()
        if now - self.last_object_time < GESTURE_COOLDOWN:
            return
        self.last_object_time = now
        self.object_counter += 1

        if kind == "polygon":
            obj = HologramObject(
                kind="polygon",
                x=float(WORKSPACE_CENTER[0]),
                y=float(WORKSPACE_CENTER[1]),
                size=92,
                sides=random.choice([5, 6, 8]),
                color=CYAN,
                object_id=f"AI_POLYGON_{self.object_counter:03d}",
                gesture=gesture,
                confidence=random.randint(91, 98),
            )
        elif kind == "box":
            obj = HologramObject(
                kind="box",
                x=float(WORKSPACE_CENTER[0]),
                y=float(WORKSPACE_CENTER[1]),
                size=118,
                color=CYAN,
                object_id=f"AI_CUBE_{self.object_counter:03d}",
                gesture=gesture,
                confidence=random.randint(90, 98),
            )
        else:
            obj = HologramObject(
                kind="mesh",
                x=float(WORKSPACE_CENTER[0]),
                y=float(WORKSPACE_CENTER[1]),
                size=126,
                color=GOLD,
                object_id=f"AI_MESH_{self.object_counter:03d}",
                gesture=gesture,
                confidence=random.randint(93, 99),
            )

        self.active_object = obj
        self.burst_at(obj.x, obj.y, color=obj.color)

    # --------------------------------------------------------
    # GESTURE LOGIC
    # --------------------------------------------------------
    def gesture_hold(self, key):
        now = time.time()
        if key not in self.gesture_start:
            self.gesture_start[key] = now
            return 0.0, False
        progress = min(1.0, (now - self.gesture_start[key]) / GESTURE_HOLD_TIME)
        return progress, progress >= 1.0

    def reset_gesture_keys_except(self, active_keys):
        for key in list(self.gesture_start.keys()):
            if key not in active_keys:
                del self.gesture_start[key]

    def update_gestures(self):
        self.preview_kind = None
        self.preview_progress = 0.0
        self.last_gesture = "IDLE"
        active_keys = set()

        # Hand-created object gestures
        for hand in self.tracked_hands:
            # Small fingertip sparkle only
            for tip_id in [4, 8]:
                tip = hand.points[tip_id]
                if random.random() < 0.35:
                    self.spawn_particle(tip[0], tip[1], color=GOLD if hand.pinch else CYAN, power=0.45)

            if hand.pinch:
                key = f"{hand.label}_pinch"
                active_keys.add(key)
                progress, ready = self.gesture_hold(key)
                self.preview_kind = "polygon"
                self.preview_progress = progress
                self.last_gesture = "PINCH HOLD → POLYGON"
                if ready:
                    self.set_active_object("polygon", gesture="PINCH CREATE")
                    if key in self.gesture_start:
                        del self.gesture_start[key]

            elif hand.open_palm:
                key = f"{hand.label}_palm"
                active_keys.add(key)
                progress, ready = self.gesture_hold(key)
                self.preview_kind = "box"
                self.preview_progress = progress
                self.last_gesture = "PALM HOLD → CUBE"
                if ready:
                    self.set_active_object("box", gesture="PALM CREATE")
                    if key in self.gesture_start:
                        del self.gesture_start[key]

        # Dual-hand mesh creation if both palms open
        if len(self.tracked_hands) >= 2 and all(h.open_palm for h in self.tracked_hands[:2]):
            key = "dual_open_mesh"
            active_keys.add(key)
            progress, ready = self.gesture_hold(key)
            self.preview_kind = "mesh"
            self.preview_progress = min(1.0, progress * (GESTURE_HOLD_TIME / TWO_HAND_MESH_HOLD))
            self.last_gesture = "DUAL PALM HOLD → AI MESH"
            if time.time() - self.gesture_start.get(key, time.time()) >= TWO_HAND_MESH_HOLD:
                self.set_active_object("mesh", gesture="DUAL HAND MESH")
                if key in self.gesture_start:
                    del self.gesture_start[key]

        self.reset_gesture_keys_except(active_keys)
        self.manipulate_active_object()

    def manipulate_active_object(self):
        if self.active_object is None or len(self.tracked_hands) < 2:
            return

        h1, h2 = self.tracked_hands[0], self.tracked_hands[1]
        dist = float(np.linalg.norm(h1.palm_center - h2.palm_center))
        angle = math.atan2(
            h2.palm_center[1] - h1.palm_center[1],
            h2.palm_center[0] - h1.palm_center[0],
        )

        # Keep object centered for professional clean presentation, use hands only for transform.
        self.active_object.x = float(WORKSPACE_CENTER[0])
        self.active_object.y = float(WORKSPACE_CENTER[1])
        self.active_object.scale = max(0.65, min(1.75, dist / 245.0))
        self.active_object.rotation = angle
        self.active_object.state = "MANIPULATING"
        self.last_gesture = "TWO-HAND SCALE / ROTATE"

    # --------------------------------------------------------
    # DRAWING: BACKGROUND + PANELS
    # --------------------------------------------------------
    def draw_background(self):
        fade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        fade.fill((0, 0, 0, BACKGROUND_ALPHA if self.clean_mode else 48))
        self.screen.blit(fade, (0, 0))

        # Workspace boundary
        pygame.draw.rect(self.screen, (2, 10, 16), (WORKSPACE_LEFT, 0, WORKSPACE_RIGHT - WORKSPACE_LEFT, HEIGHT))

        # Clean grid only in workspace
        t = time.time() - self.start_time
        for x in range(WORKSPACE_LEFT, WORKSPACE_RIGHT, 80):
            pygame.draw.line(self.screen, SOFT_GRID, (x, 0), (x, HEIGHT), 1)
        for y in range(35, HEIGHT, 80):
            pygame.draw.line(self.screen, SOFT_GRID, (WORKSPACE_LEFT, y), (WORKSPACE_RIGHT, y), 1)

        # Subtle scan line
        scan_y = int((math.sin(t * 0.55) * 0.5 + 0.5) * HEIGHT)
        pygame.draw.line(self.screen, (8, 55, 75), (WORKSPACE_LEFT, scan_y), (WORKSPACE_RIGHT, scan_y), 1)

        # Center targeting reticle
        cx, cy = WORKSPACE_CENTER
        pygame.draw.circle(self.screen, (8, 55, 75), (int(cx), int(cy)), 150, 1)
        pygame.draw.circle(self.screen, (8, 55, 75), (int(cx), int(cy)), 80, 1)
        pygame.draw.line(self.screen, (8, 55, 75), (int(cx - 170), int(cy)), (int(cx + 170), int(cy)), 1)
        pygame.draw.line(self.screen, (8, 55, 75), (int(cx), int(cy - 170)), (int(cx), int(cy + 170)), 1)

    def draw_camera_panel(self, frame):
        panel = pygame.Surface((CAMERA_PANEL_WIDTH, HEIGHT), pygame.SRCALPHA)
        panel.fill((*PANEL_BG, 245))
        self.screen.blit(panel, (0, 0))

        self.text("DIGITAL SHADOW", 24, 24, self.font_large, WHITE)
        self.text("CLEAN HOLOGRAPHIC AI INTERFACE", 25, 62, self.font_small, CYAN)
        self.text("ONE OBJECT • ONE ANALYSIS PANEL", 25, 82, self.font_tiny, (160, 200, 220))

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (CAMERA_PANEL_WIDTH - 30, 210))
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            self.screen.blit(frame_surface, (15, 108))
            pygame.draw.rect(self.screen, CYAN, (15, 108, CAMERA_PANEL_WIDTH - 30, 210), 1)

    def draw_left_status(self, fps):
        if not self.show_hud:
            return
        status_color = GREEN if self.face_found else RED
        self.text("SYSTEM STATUS", 25, 350, self.font_medium, CYAN)
        self.text(f"FACE     : {'LOCKED' if self.face_found else 'SEARCHING'}", 25, 382, color=status_color)
        self.text(f"HANDS    : {len(self.tracked_hands)}", 25, 406, color=GREEN if self.tracked_hands else RED)
        self.text(f"GESTURE  : {self.last_gesture[:24]}", 25, 430, color=GOLD)
        self.text(f"PREVIEW  : {int(self.preview_progress * 100):03d}%", 25, 454, color=GOLD)
        self.text(f"FPS      : {fps:.1f}", 25, 478)
        self.text(f"PARTICLE : {len(self.particles)}", 25, 502)
        self.text(f"MODE     : {'CLEAN' if self.clean_mode else 'FULL'}", 25, 526, color=CYAN)

        self.text("GESTURES", 25, 578, self.font_medium, CYAN)
        self.text("Hold pinch      -> polygon", 25, 610)
        self.text("Hold open palm  -> cube", 25, 634)
        self.text("Two palms       -> AI mesh", 25, 658)
        self.text("Two hands       -> scale/rotate", 25, 682)

    def draw_right_analysis_panel(self):
        x = WORKSPACE_RIGHT
        y = 0
        w = RIGHT_PANEL_WIDTH
        h = HEIGHT
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((*PANEL_BG, 246))
        self.screen.blit(panel, (x, y))

        # Main top card
        card_x = x + 22
        card_y = 38
        card_w = w - 44
        card_h = 180
        pygame.draw.rect(self.screen, (6, 22, 32), (card_x, card_y, card_w, card_h), 0)
        pygame.draw.rect(self.screen, CYAN, (card_x, card_y, card_w, card_h), 1)
        self.text("AI VISUAL ENGINE", card_x + 18, card_y + 22, self.font_medium, WHITE)
        self.text(f"RENDER MODE : {'CLEAN' if self.clean_mode else 'FULL'}", card_x + 18, card_y + 60, self.font_small, CYAN)
        self.text("ANNOTATION  : FIXED PANEL", card_x + 18, card_y + 86, self.font_small, GOLD)
        self.text("OBJECT MODE : SINGLE ACTIVE", card_x + 18, card_y + 112, self.font_small, WHITE)
        self.text("STATUS      : STABLE", card_x + 18, card_y + 138, self.font_small, GREEN)

        # Object analysis card
        card_y2 = 255
        card_h2 = 255
        pygame.draw.rect(self.screen, (6, 22, 32), (card_x, card_y2, card_w, card_h2), 0)
        pygame.draw.rect(self.screen, CYAN, (card_x, card_y2, card_w, card_h2), 1)
        self.text("AI OBJECT ANALYSIS", card_x + 18, card_y2 + 18, self.font_medium, WHITE)
        pygame.draw.line(self.screen, CYAN, (card_x + 16, card_y2 + 52), (card_x + card_w - 16, card_y2 + 52), 1)

        if self.active_object is None:
            self.text("NO ACTIVE OBJECT", card_x + 18, card_y2 + 78, self.font_small, GOLD)
            self.text("Create one using gesture:", card_x + 18, card_y2 + 110, self.font_small, WHITE)
            self.text("Pinch / Palm / Dual Palm", card_x + 18, card_y2 + 136, self.font_small, CYAN)
        else:
            obj = self.active_object
            obj.state = "STABLE" if len(self.tracked_hands) < 2 else obj.state
            self.text(f"ID         : {obj.object_id}", card_x + 18, card_y2 + 72, self.font_small, WHITE)
            self.text(f"TYPE       : {obj.kind.upper()}", card_x + 18, card_y2 + 100, self.font_small, WHITE)
            self.text(f"GESTURE    : {obj.gesture}", card_x + 18, card_y2 + 128, self.font_small, GOLD)
            self.text(f"CONFIDENCE : {obj.confidence}%", card_x + 18, card_y2 + 156, self.font_small, GOLD)
            self.text(f"SCALE      : {obj.scale:.2f}x", card_x + 18, card_y2 + 184, self.font_small, CYAN)
            self.text(f"STATE      : {obj.state}", card_x + 18, card_y2 + 212, self.font_small, GREEN)

        # Gesture card
        card_y3 = 545
        card_h3 = 125
        pygame.draw.rect(self.screen, (6, 22, 32), (card_x, card_y3, card_w, card_h3), 0)
        pygame.draw.rect(self.screen, (30, 90, 120), (card_x, card_y3, card_w, card_h3), 1)
        self.text("LIVE CONTROL", card_x + 18, card_y3 + 18, self.font_medium, WHITE)
        self.text(f"CURRENT : {self.last_gesture[:20]}", card_x + 18, card_y3 + 56, self.font_small, GOLD)
        self.text(f"HOLD    : {int(self.preview_progress * 100):03d}%", card_x + 18, card_y3 + 84, self.font_small, CYAN)

    # --------------------------------------------------------
    # DRAWING: OBJECTS + HANDS
    # --------------------------------------------------------
    def draw_preview(self):
        if self.preview_kind is None or self.preview_progress <= 0:
            return
        x, y = WORKSPACE_CENTER
        radius = 96
        color = GOLD if self.preview_kind in ["polygon", "mesh"] else CYAN

        self.draw_transparent_circle(color, int(x), int(y), radius + 16, 20, 1)
        self.draw_transparent_circle(color, int(x), int(y), radius, 90, 1)

        # progress arc
        end_angle = -math.pi / 2 + self.preview_progress * math.tau
        points = []
        steps = 56
        for i in range(steps):
            a = -math.pi / 2 + (end_angle + math.pi / 2) * i / max(1, steps - 1)
            points.append((int(x + math.cos(a) * radius), int(y + math.sin(a) * radius)))
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 4)

        self.text(f"GENERATING {self.preview_kind.upper()}", int(x - 96), int(y + 128), self.font_small, color)
        self.text(f"{int(self.preview_progress * 100)}%", int(x - 18), int(y - 8), self.font_medium, WHITE)

    def draw_active_object(self, dt):
        if self.active_object is None:
            # Idle mesh face-like neural symbol, similar to user's reference image but simple
            self.draw_idle_ai_face()
            return

        obj = self.active_object
        obj.update(dt)
        x, y = obj.x, obj.y
        s = obj.size * obj.scale
        color = obj.color

        # focus reticle around object
        pulse = math.sin(obj.pulse * 4.0) * 4
        self.draw_transparent_circle(color, int(x), int(y), int(s + 34 + pulse), 42, 1)
        self.draw_transparent_circle(WHITE, int(x), int(y), int(s + 54 - pulse), 20, 1)

        if obj.kind == "polygon":
            self.draw_polygon_object(obj, x, y, s, color)
        elif obj.kind == "box":
            self.draw_box_object(obj, x, y, s, color)
        else:
            self.draw_mesh_object(obj, x, y, s, color)

        # small clean label under object
        self.text(obj.object_id, int(x - 72), int(y + s + 72), self.font_small, color)

    def draw_polygon_object(self, obj, x, y, s, color):
        pts = self.polygon_points(x, y, s, obj.sides, obj.rotation)
        inner = self.polygon_points(x, y, s * 0.52, obj.sides, -obj.rotation * 0.5)
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(overlay, (*color, 30), pts, 0)
        pygame.draw.polygon(overlay, (*color, 230), pts, 2)
        pygame.draw.polygon(overlay, (*WHITE, 90), inner, 1)
        for a, b in zip(pts, inner):
            pygame.draw.line(overlay, (*color, 70), a, b, 1)
        for p in pts:
            pygame.draw.circle(overlay, (*WHITE, 210), p, 4)
            pygame.draw.circle(overlay, (*color, 80), p, 10, 1)
        self.screen.blit(overlay, (0, 0))

    def draw_box_object(self, obj, x, y, s, color):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dx = math.cos(obj.rotation) * s * 0.28
        dy = math.sin(obj.rotation) * s * 0.18 - s * 0.18
        rect1 = pygame.Rect(int(x - s / 2), int(y - s / 2), int(s), int(s))
        rect2 = pygame.Rect(int(x - s / 2 + dx), int(y - s / 2 + dy), int(s), int(s))
        pygame.draw.rect(overlay, (*color, 28), rect1, 0)
        pygame.draw.rect(overlay, (*color, 235), rect1, 2)
        pygame.draw.rect(overlay, (*WHITE, 120), rect2, 1)
        corners1 = [rect1.topleft, rect1.topright, rect1.bottomleft, rect1.bottomright]
        corners2 = [rect2.topleft, rect2.topright, rect2.bottomleft, rect2.bottomright]
        for c1, c2 in zip(corners1, corners2):
            pygame.draw.line(overlay, (*color, 110), c1, c2, 1)
        for i in range(4):
            yy = rect1.y + int(20 + i * s / 5)
            pygame.draw.line(overlay, (*WHITE, 50), (rect1.x + 12, yy), (rect1.x + rect1.w - 12, yy), 1)
        self.screen.blit(overlay, (0, 0))

    def draw_mesh_object(self, obj, x, y, s, color):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        random.seed(42)  # stable mesh, not flickering
        nodes = []
        node_count = 28
        for i in range(node_count):
            a = obj.rotation + i * math.tau / node_count
            r = s * (0.35 + 0.68 * random.random())
            px = x + math.cos(a) * r + random.uniform(-18, 18)
            py = y + math.sin(a) * r + random.uniform(-18, 18)
            nodes.append((px, py))

        # connect close nodes for neural mesh
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                d = math.dist(nodes[i], nodes[j])
                if d < s * 0.48:
                    alpha = int(max(20, 115 - d * 0.45))
                    pygame.draw.line(
                        overlay,
                        (*color, alpha),
                        (int(nodes[i][0]), int(nodes[i][1])),
                        (int(nodes[j][0]), int(nodes[j][1])),
                        1,
                    )

        # central AI face-like outline
        face_pts = []
        for i in range(34):
            a = -math.pi / 2 + i * math.tau / 34
            rx = s * 0.42
            ry = s * 0.58
            face_pts.append((int(x + math.cos(a) * rx), int(y + math.sin(a) * ry)))
        pygame.draw.lines(overlay, (*WHITE, 110), True, face_pts, 1)

        # eyes
        pygame.draw.circle(overlay, (*CYAN, 170), (int(x - s * 0.16), int(y - s * 0.08)), 4)
        pygame.draw.circle(overlay, (*CYAN, 170), (int(x + s * 0.16), int(y - s * 0.08)), 4)
        pygame.draw.line(overlay, (*WHITE, 70), (int(x - s * 0.10), int(y + s * 0.20)), (int(x + s * 0.10), int(y + s * 0.20)), 1)

        for px, py in nodes:
            pygame.draw.circle(overlay, (*WHITE, 200), (int(px), int(py)), 3)
            pygame.draw.circle(overlay, (*color, 60), (int(px), int(py)), 7, 1)
        self.screen.blit(overlay, (0, 0))

    def draw_idle_ai_face(self):
        x, y = WORKSPACE_CENTER
        t = time.time() - self.start_time
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        nodes = []
        random.seed(7)
        for i in range(34):
            a = i * math.tau / 34
            r = 105 + random.uniform(-28, 34)
            px = x + math.cos(a) * r
            py = y + math.sin(a) * r * 1.16
            nodes.append((px, py))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if math.dist(nodes[i], nodes[j]) < 92:
                    pygame.draw.line(
                        overlay,
                        (*CYAN, 35),
                        (int(nodes[i][0]), int(nodes[i][1])),
                        (int(nodes[j][0]), int(nodes[j][1])),
                        1,
                    )
        for px, py in nodes:
            pygame.draw.circle(overlay, (*WHITE, 120), (int(px), int(py)), 2)
        pygame.draw.circle(overlay, (*CYAN, 90), (int(x - 32), int(y - 10)), 4)
        pygame.draw.circle(overlay, (*CYAN, 90), (int(x + 32), int(y - 10)), 4)
        pygame.draw.line(overlay, (*WHITE, 50), (int(x - 28), int(y + 45)), (int(x + 28), int(y + 45)), 1)
        self.screen.blit(overlay, (0, 0))
        self.text("WAITING FOR GESTURE", int(x - 92), int(y + 165 + math.sin(t * 2) * 4), self.font_small, CYAN)

    def draw_hands_overlay(self):
        for hand in self.tracked_hands:
            pts = hand.points
            color = GOLD if hand.pinch else (GREEN if hand.open_palm else CYAN)

            # Essential clean hand geometry only
            essential = [
                (0, 5), (5, 8),
                (0, 9), (9, 12),
                (0, 13), (13, 16),
                (0, 17), (17, 20),
                (0, 4),
                (5, 9), (9, 13), (13, 17),
            ]
            for a, b in essential:
                self.draw_glow_line(pts[a], pts[b], color, width=1, alpha=72)

            for i in [4, 8, 12, 16, 20]:
                p = pts[i]
                r = 7 if i in [4, 8] else 4
                self.draw_transparent_circle(color, int(p[0]), int(p[1]), r + 7, 22, 0)
                self.draw_transparent_circle(WHITE if i in [4, 8] else color, int(p[0]), int(p[1]), r, 180, 0)

            self.draw_transparent_circle(color, int(hand.palm_center[0]), int(hand.palm_center[1]), 22, 22, 1)
            self.draw_transparent_circle(color, int(hand.palm_center[0]), int(hand.palm_center[1]), 6, 190, 0)

            # index laser pointer
            direction = hand.index_tip - hand.wrist
            norm = np.linalg.norm(direction)
            if norm > 1:
                direction = direction / norm
                end = hand.index_tip + direction * 190
                self.draw_glow_line(hand.index_tip, end, color, width=1, alpha=75)

        # clean two-hand bridge
        if len(self.tracked_hands) >= 2:
            a = self.tracked_hands[0].palm_center
            b = self.tracked_hands[1].palm_center
            self.draw_glow_line(a, b, GOLD, width=2, alpha=105)
            mid = (a + b) / 2
            self.draw_transparent_circle(GOLD, int(mid[0]), int(mid[1]), 32, 18, 1)

    def draw_particles(self):
        for p in self.particles:
            a = min(p.alpha, 150)
            if a <= 0:
                continue
            self.draw_transparent_circle(p.color, int(p.x), int(p.y), max(1, int(p.size)), a, 0)

    # --------------------------------------------------------
    # EVENTS + LOOP
    # --------------------------------------------------------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.particles.clear()
                    self.active_object = None
                    self.gesture_start.clear()
                elif event.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                elif event.key == pygame.K_m:
                    self.mirror = not self.mirror
                elif event.key == pygame.K_c:
                    self.clean_mode = not self.clean_mode
                elif event.key == pygame.K_1:
                    self.set_active_object("polygon", gesture="KEYBOARD FORCE")
                elif event.key == pygame.K_2:
                    self.set_active_object("box", gesture="KEYBOARD FORCE")
                elif event.key == pygame.K_3:
                    self.set_active_object("mesh", gesture="KEYBOARD FORCE")

    def run(self):
        self.screen.fill(BLACK)

        while self.running:
            now = time.time()
            dt = min(0.05, now - self.last_frame)
            self.last_frame = now

            self.handle_events()

            frame_result = self.read_tracking()
            if frame_result[0] is None:
                frame, detected, face_center_cam, face_size, hands = None, False, None, self.face_size, []
            else:
                frame, detected, face_center_cam, face_size, hands = frame_result

            self.tracked_hands = hands
            self.update_face(detected, face_center_cam, face_size)
            self.update_gestures()
            self.update_particles(dt)

            fps = self.clock.get_fps()

            self.draw_background()
            self.draw_preview()
            self.draw_active_object(dt)
            self.draw_hands_overlay()
            self.draw_particles()
            self.draw_camera_panel(frame)
            self.draw_left_status(fps)
            self.draw_right_analysis_panel()

            pygame.display.flip()
            self.clock.tick(FPS_TARGET)

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.face_detector.close()
        self.hands_detector.close()
        pygame.quit()


if __name__ == "__main__":
    app = AdvancedDigitalShadow()
    app.run()
