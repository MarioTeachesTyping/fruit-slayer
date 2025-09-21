# ========================= #
# run this file for backend #
# ========================= #

import cv2
import mediapipe as mp
import time, random, math

W, H = 960, 540
GRAVITY = 1200.0  # px/s^2
SPAWN_EVERY = 0.9  # seconds
SLICE_SPEED_THRESH = 1400.0  # px/s (tweak)
FRUIT_RADIUS = 28
MAX_FRUITS = 6

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

class Fruit:
    __slots__ = ("x","y","vx","vy","alive","born")
    def __init__(self, now):
        self.x = random.randint(int(0.15*W), int(0.85*W))
        self.y = H + FRUIT_RADIUS + 5
        self.vx = random.uniform(-220, 220)
        self.vy = -random.uniform(700, 1000)  # toss upward
        self.alive = True
        self.born = now
    def update(self, dt):
        if not self.alive: return
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        # simple bounds
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
            fruits.append(Fruit(now))
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

        # check collisions if slicing
        if slice_segment:
            x1,y1,x2,y2 = slice_segment
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 2)
            for f in fruits:
                if f.alive and seg_circle_intersects(x1,y1,x2,y2, int(f.x), int(f.y), FRUIT_RADIUS):
                    f.alive = False
                    score += 1

        # draw fruits
        for f in fruits:
            if not f.alive: continue
            cv2.circle(frame, (int(f.x), int(f.y)), FRUIT_RADIUS, (0, 255, 0), 2)
            cv2.circle(frame, (int(f.x), int(f.y)), FRUIT_RADIUS-4, (0, 128, 0), -1)

        # clean up dead fruits occasionally
        fruits = [f for f in fruits if f.alive or (now - f.born) < 8.0]

        # HUD
        cv2.putText(frame, f"Score: {score}", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, "Press 'q' to quit", (16, H-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

        cv2.imshow("Fruit Slayer (Backend)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()