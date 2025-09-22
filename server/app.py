# ========================= #
# run this file for backend #
# ========================= #

import os
import cv2
import mediapipe as mp
import time, random, math
import numpy as np
import sys

W, H = 960, 540
GRAVITY = 1200.0  # px/s^2
SPAWN_EVERY = 0.9  # seconds
SLICE_SPEED_THRESH = 1400.0  # px/s (tweak)
FRUIT_RADIUS = 28
MAX_FRUITS = 6
MULT_DURATION = 5.0  # seconds of 2x multiplier

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "assets")

def load_png(name):
    """Load a PNG with alpha, raise clear error if missing, and ensure 4 channels."""
    path = os.path.join(ASSET_DIR, name)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # keep alpha channel
    if img is None:
        raise FileNotFoundError(
            f"Asset not found or unreadable: {path}\n"
            f"Working dir: {os.getcwd()}\n"
            f"Tip: ensure the file exists and the path is correct."
        )
    # If the image has only 3 channels, add an opaque alpha channel
    if img.ndim == 3 and img.shape[2] == 3:
        alpha = 255 * np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
    return img

def play_slice_sfx():
    if sys.platform.startswith("win"):
        import winsound
        winsound.PlaySound(
            os.path.join(ASSET_DIR, "slice.wav"),
            winsound.SND_FILENAME | winsound.SND_ASYNC  # non-blocking
        )
    else:
        # No-op on non-Windows (or print a warning)
        pass

def play_explosion_sfx():
    if sys.platform.startswith("win"):
        import winsound
        winsound.PlaySound(
            os.path.join(ASSET_DIR, "explosion.wav"),
            winsound.SND_FILENAME | winsound.SND_ASYNC
        )
    else:
        pass

# Load them in beforehand
fruit_imgs = {
    "red":    load_png("strawberry.png"),
    "orange": load_png("mango.png"),
    "yellow": load_png("pineapple.png"),
    "green":  load_png("watermelon.png"),
    "blue":   load_png("score_2x_banana.png"),
    "purple": load_png("plum.png"),
    "brown":  load_png("coconut.png"),
    "bomb":   load_png("bomb.png"),
}

hud_icon = load_png("watermelon.png")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def overlay_img(bg, fg, x, y):
    """
    Alpha-blend fg (with 4 channels) onto bg (3 channels), centered at (x,y).
    Handles all edge cases where the sprite is partially or fully off-screen.
    """
    fh, fw = fg.shape[:2]
    H, W = bg.shape[:2]

    # top-left of where we'd like to place the sprite
    x1 = int(round(x - fw / 2))
    y1 = int(round(y - fh / 2))
    x2 = x1 + fw
    y2 = y1 + fh

    # compute intersection with the background bounds
    bx1 = max(0, x1)
    by1 = max(0, y1)
    bx2 = min(W, x2)
    by2 = min(H, y2)

    # if completely off-screen, nothing to do
    if bx1 >= bx2 or by1 >= by2:
        return bg

    # corresponding crop on the foreground
    fx1 = bx1 - x1
    fy1 = by1 - y1
    fx2 = fx1 + (bx2 - bx1)
    fy2 = fy1 + (by2 - by1)

    fg_crop = fg[fy1:fy2, fx1:fx2]
    bg_roi  = bg[by1:by2, bx1:bx2]

    # ensure we actually have overlap
    if fg_crop.size == 0 or bg_roi.size == 0:
        return bg

    # alpha blend (assumes fg has 4 channels)
    alpha = fg_crop[:, :, 3] / 255.0
    # expand alpha to 3 channels to match BGR
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]

    # blend into the ROI in-place
    bg[by1:by2, bx1:bx2] = alpha * fg_crop[:, :, :3] + (1.0 - alpha) * bg_roi
    return bg

class Fruit:
    TYPES = ["red","orange","yellow","green","blue","purple","brown"]
    def __init__(self, now, bomb_prob=0.12):
        self.x = random.randint(int(0.15*W), int(0.85*W))
        self.y = H + FRUIT_RADIUS + 5
        self.vx = random.uniform(-220, 220)
        self.vy = -random.uniform(700, 1300)
        self.alive = True
        self.born = now
        if random.random() < bomb_prob:
            self.key = "bomb"
            self.is_bomb = True
        else:
            self.key = random.choice(Fruit.TYPES)
            self.is_bomb = False

    def update(self, dt):
        if not self.alive: return
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.y - FRUIT_RADIUS > H + 80:
            self.alive = False

def seg_circle_intersects(x1,y1,x2,y2, cx,cy,r):
    # Segment-circle intersection (closest distance <= r and projection within segment)
    dx, dy = x2-x1, y2-y1
    if dx==0 and dy==0:
        return math.hypot(cx-x1, cy-y1) <= r
    t = ((cx-x1)*dx + (cy-y1)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    px, py = x1 + t*dx, y1 + t*dy
    return (cx - px)**2 + (cy - py)**2 <= r*r

# If user moves their too fast, the detection disappears. Tries to make it a lil better.
class TipFilter:
    def __init__(self):
        self.x = self.y = None
        self.vx = self.vy = 0.0
        self.last_t = None
        self.last_seen = -1.0

    def _alpha(self, speed):  # “One Euro-ish”: more smoothing when slow, less when fast
        # speed in px/s; clamp between 0.2–0.85 adaptively
        return max(0.2, min(0.85, speed / 2000.0))

    def update_visible(self, x, y, t):
        if self.last_t is None:
            self.x, self.y = x, y
            self.vx = self.vy = 0.0
            self.last_t = t
            self.last_seen = t
            return self.x, self.y, False  # not predicted

        dt = max(1e-3, t - self.last_t)
        # instantaneous velocity
        inst_vx = (x - self.x) / dt
        inst_vy = (y - self.y) / dt
        speed = (inst_vx**2 + inst_vy**2) ** 0.5
        a = self._alpha(speed)
        # EMA smoothing on pos; keep velocity blended
        self.x = (1 - a) * self.x + a * x
        self.y = (1 - a) * self.y + a * y
        self.vx = 0.6 * self.vx + 0.4 * inst_vx
        self.vy = 0.6 * self.vy + 0.4 * inst_vy
        self.last_t = t
        self.last_seen = t
        return self.x, self.y, False

    def predict_if_missing(self, t, horizon=0.12):
        # If we lost the point recently, predict forward with constant velocity
        if self.last_seen < 0 or (t - self.last_seen) > horizon:
            return None  # too long; treat as missing
        dt = t - self.last_t if self.last_t else 0.0
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        self.last_t = t
        return self.x, self.y, True  # predicted

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        print("Could not open camera"); return

    fruits = []
    score = 0
    last_spawn = 0.0
    last_t = time.time()

    # fingertip tracking
    last_tip = None
    last_tip_time = None

    bomb_flash_until = 0.0 # will flash red screen if you hit a bomb
    score_mult_until = 0.0  # time until which 2x is active

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (W, H))
        frame = cv2.flip(frame, 1)  # mirror

        now = time.time()
        dt = now - last_t
        last_t = now

        # spawn fruits
        if now - last_spawn > SPAWN_EVERY and sum(f.alive for f in fruits) < MAX_FRUITS:
            # bomb chance grows with score, but caps out
            bomb_prob = 0.1 + min(0.3, score * 0.01)  # 10% base, up to 40% max
            fruits.append(Fruit(now, bomb_prob=bomb_prob))
            last_spawn = now

        # update fruits
        for f in fruits:
            f.update(dt)

        # detect hand + fingertip
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        slice_segment = None  # (x1,y1,x2,y2)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            tip = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * W), int(tip.y * H)

            if last_tip is not None and last_tip_time is not None:
                dt_tip = now - last_tip_time
                if dt_tip > 0:
                    speed = math.hypot(x - last_tip[0], y - last_tip[1]) / dt_tip
                    if speed > SLICE_SPEED_THRESH:
                        slice_segment = (last_tip[0], last_tip[1], x, y)

            last_tip = (x, y)
            last_tip_time = now

            # draw fingertip
            cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)

        if slice_segment:
            x1,y1,x2,y2 = slice_segment
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 2)
            for f in fruits:
                if f.alive and seg_circle_intersects(x1,y1,x2,y2, int(f.x), int(f.y), FRUIT_RADIUS):
                    f.alive = False
                    if getattr(f, "is_bomb", False):
                        score = max(0, score - 5)
                        bomb_flash_until = now + 0.20
                        play_explosion_sfx()
                    else:
                        # base points for a fruit
                        base = 1

                        # if it's the 2x banana, activate/extend multiplier
                        if f.key == "blue":
                            # stacking: extend by 5s if already active, otherwise set for 5s
                            score_mult_until = max(score_mult_until, now) + MULT_DURATION
                            play_slice_sfx()

                        # compute multiplier AFTER possibly updating the timer
                        mult = 2 if now < score_mult_until else 1
                        score += base * mult
                        play_slice_sfx()


        # draw fruits
        for f in fruits:
            if not f.alive:
                continue

            img = fruit_imgs.get(f.key, None)
            if img is None:
                # fallback: draw a simple circle so game keeps running
                cv2.circle(frame, (int(f.x), int(f.y)), FRUIT_RADIUS, (0,255,0), -1)
                continue

            # resize sprite to match fruit size
            scale = (2 * FRUIT_RADIUS) / img.shape[0]
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            fg = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # overlay PNG with alpha
            frame = overlay_img(frame, fg, int(f.x), int(f.y))


        # clean up dead fruits occasionally
        fruits = [f for f in fruits if f.alive or (now - f.born) < 8.0]

        if now < bomb_flash_until:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (W,H), (0,0,255), -1)  # red tint
            alpha = 0.25
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # HUD
        # resize icon to a fixed height
        ICON_H = 36
        scale = ICON_H / hud_icon.shape[0]
        icon_w = int(hud_icon.shape[1] * scale)
        icon_h = int(hud_icon.shape[0] * scale)
        icon_resized = cv2.resize(hud_icon, (icon_w, icon_h), interpolation=cv2.INTER_AREA)

        MARGIN = 12
        # draw icon top-left
        frame = overlay_img(frame, icon_resized, MARGIN + icon_w // 2, MARGIN + icon_h // 2)

        # draw the numeric score to the right of the icon
        text = f"{score}"
        tx = MARGIN + icon_w + 8
        ty = MARGIN + icon_h - 6

        # shadow + white for readability
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 2, cv2.LINE_AA)

        # show x2 badge while multiplier is active
        remaining = score_mult_until - now
        if remaining > 0:
            badge_text = f"x2"
            # place it a bit to the right of the score
            bx = tx + 60
            by = ty
            # subtle glow/outline
            cv2.putText(frame, badge_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, badge_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 215, 255), 2, cv2.LINE_AA)  # gold/yellow

        cv2.putText(frame, "Press 'q' to quit", (16, H-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

        cv2.imshow("Fruit Slayer (Backend)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()