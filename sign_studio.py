# ==========================================================
#  Sign ➜ Speech  STUDIO   (camera + side info panel, no terminal)
#  Everything is controlled from the window. Arabic renders
#  correctly on the panel (no scrambled terminal text).
#  Teach signs by picking a number from the list below.
#  Requires: opencv-python mediapipe numpy pillow arabic_reshaper
#            python-bidi gTTS pygame
# ==========================================================
import os, sys, json, hashlib, threading, queue
import numpy as np

# ========== EDIT YOUR SIGNS HERE (up to 9) ==========
LABELS = ["مرحبا", "نعم", "لا", "شكراً", "أحبك", "من فضلك", "آسف", "وداعاً"]
# ====================================================

CAMERA_INDEX = 1             # 0 = built-in webcam, 1 = external camera
MODEL_PATH   = "signs.json"
K            = 5
MATCH_THRESH = 0.60
STABLE_FRAMES = 8
SAMPLES_PER_SIGN = 40
PANEL_W      = 440

# ---------------- Arabic shaping for PIL drawing ----------------
def shape_ar(text):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)

def find_font():
    for p in [
        "C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        if os.path.exists(p):
            return p
    return None

# ---------------- hand features ----------------
def landmarks_to_vector(lm):
    pts = np.asarray(lm, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]
    pts = pts / (np.linalg.norm(pts[9]) + 1e-6)
    return pts.reshape(-1)

def mirror_vector(vec):
    p = vec.reshape(21, 3).copy(); p[:, 0] *= -1
    return p.reshape(-1)

# ---------------- sign bank (KNN) ----------------
class SignBank:
    def __init__(self, path=MODEL_PATH):
        self.path = path; self.X = []; self.y = []; self.load()
    def load(self):
        if os.path.exists(self.path):
            d = json.load(open(self.path, encoding="utf-8"))
            self.X = [np.asarray(v, np.float32) for v in d.get("X", [])]
            self.y = list(d.get("y", []))
    def save(self):
        json.dump({"X": [v.tolist() for v in self.X], "y": self.y},
                  open(self.path, "w", encoding="utf-8"), ensure_ascii=False)
    def add(self, vec, label): self.X.append(np.asarray(vec, np.float32)); self.y.append(label)
    def count(self, label): return self.y.count(label)
    def remove_label(self, label):
        keep = [(x, y) for x, y in zip(self.X, self.y) if y != label]
        self.X = [x for x, _ in keep]; self.y = [y for _, y in keep]
    def clear(self): self.X.clear(); self.y.clear()
    def predict(self, vec):
        if not self.X: return None, 0.0
        X = np.stack(self.X)
        dist = np.minimum(np.linalg.norm(X - vec, axis=1),
                          np.linalg.norm(X - mirror_vector(vec), axis=1))
        idx = np.argsort(dist)[:K]
        near = [self.y[i] for i in idx]
        best = max(set(near), key=near.count)
        best_d = float(np.mean([dist[idx[i]] for i in range(len(idx)) if near[i] == best]))
        if best_d > MATCH_THRESH: return None, 0.0
        return best, max(0.0, 1.0 - best_d / MATCH_THRESH)

# ---------------- speech (gTTS + pygame, background) ----------------
def make_speaker():
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    q = queue.Queue()
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
    os.makedirs(cache, exist_ok=True)
    mixer_ok = False
    try:
        import pygame; pygame.mixer.init(); mixer_ok = True
    except Exception as e:
        print("pygame mixer:", e)
    def synth(text):
        from gtts import gTTS
        p = os.path.join(cache, hashlib.md5(text.encode()).hexdigest() + ".mp3")
        if not os.path.exists(p): gTTS(text=text, lang="ar").save(p)
        return p
    def worker():
        while True:
            t = q.get()
            if t is None: break
            try:
                p = synth(t)
                if mixer_ok:
                    import pygame
                    pygame.mixer.music.load(p); pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy(): pygame.time.wait(50)
                    try: pygame.mixer.music.unload()
                    except Exception: pass
                elif sys.platform.startswith("win"): os.startfile(p)
                else: print("speak:", t)
            except Exception as e:
                print("tts:", e)
            q.task_done()
    threading.Thread(target=worker, daemon=True).start()
    return lambda t: q.put(t)

# ---------------- UI panel ----------------
class UI:
    def __init__(self, h):
        from PIL import ImageFont
        fp = find_font()
        self.ok = fp is not None
        self.big   = ImageFont.truetype(fp, 54) if fp else ImageFont.load_default()
        self.mid   = ImageFont.truetype(fp, 30) if fp else ImageFont.load_default()
        self.small = ImageFont.truetype(fp, 24) if fp else ImageFont.load_default()
        self.tiny  = ImageFont.truetype(fp, 20) if fp else ImageFont.load_default()
        self.h = h

    def panel(self, state):
        from PIL import Image, ImageDraw
        W, H = PANEL_W, self.h
        img = Image.new("RGB", (W, H), (13, 13, 23)); d = ImageDraw.Draw(img)
        def rtext(y, text, font, fill, pad=24):
            s = shape_ar(text); w = d.textlength(s, font=font)
            d.text((W - pad - w, y), s, font=font, fill=fill); return w

        d.rectangle([0, 0, W, 70], fill=(124, 92, 255))
        rtext(16, "لغة الإشارة ← صوت", self.mid, (12, 12, 20))

        mode = state["mode"]
        y = 95
        if mode == "recognize":
            rtext(y, "الوضع: التعرّف", self.small, (150, 150, 170)); y += 46
            word = state["word"] or "—"
            d.rectangle([20, y, W - 20, y + 90], fill=(22, 22, 38),
                        outline=(124, 92, 255), width=2)
            s = shape_ar(word); w = d.textlength(s, font=self.big)
            col = (124, 240, 180) if state["word"] else (90, 90, 110)
            d.text(((W - w) / 2, y + 16), s, font=self.big, fill=col)
            y += 110
            conf = state["conf"]
            d.rectangle([24, y, W - 24, y + 16], fill=(30, 30, 48))
            d.rectangle([24, y, 24 + int((W - 48) * conf), y + 16], fill=(124, 240, 180))
            y += 40
        elif mode == "select":
            rtext(y, "اختر رقم الإشارة لتعليمها:", self.small, (250, 210, 120)); y += 50
        elif mode == "teach":
            rtext(y, f"علّم: {state['teach']}", self.mid, (250, 210, 120)); y += 50
            n = state["n"]; frac = n / SAMPLES_PER_SIGN
            d.rectangle([24, y, W - 24, y + 22], fill=(30, 30, 48))
            d.rectangle([24, y, 24 + int((W - 48) * frac), y + 22], fill=(250, 210, 120))
            d.text((W / 2 - 20, y + 2), f"{n}/{SAMPLES_PER_SIGN}", font=self.small, fill=(12, 12, 20))
            y += 50
            rtext(y, "اثبت على الإشارة أمام الكاميرا", self.tiny, (150, 150, 170)); y += 40

        # signs list
        y = max(y, 300)
        rtext(y, "الإشارات:", self.small, (150, 150, 170)); y += 40
        for i, lab in enumerate(LABELS, 1):
            n = state["counts"].get(lab, 0)
            done = n > 0
            mark = "✓" if done else "•"
            col = (124, 240, 180) if done else (120, 120, 140)
            line = f"{i}.  {lab}   {mark}"
            rtext(y, line, self.small, col); y += 34

        # controls
        yb = H - 70
        d.line([20, yb - 12, W - 20, yb - 12], fill=(40, 40, 60), width=1)
        rtext(yb, "T علّم  C مسح  S حفظ  F ملء الشاشة  Q خروج", self.tiny, (150, 150, 170))

        import cv2
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ---------------- main ----------------
def run():
    import cv2
    import mediapipe as mp
    try:
        mp_hands = mp.solutions.hands; mp_draw = mp.solutions.drawing_utils
    except AttributeError:
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions import drawing_utils as mp_draw

    bank = SignBank()
    say = make_speaker(); say("جاهز")
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # request a 16:9 size (most webcams' native ratio) to avoid a stretched face
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    TARGET_H = 720
    def fit_height(f):
        """Resize to TARGET_H keeping aspect ratio (prevents a stretched image)."""
        h, w = f.shape[:2]
        if h != TARGET_H:
            f = cv2.resize(f, (int(w * TARGET_H / h), TARGET_H))
        return f

    ok, frame = cap.read()
    if not ok:
        print(f"Camera not found (index {CAMERA_INDEX}). Try CAMERA_INDEX = 0"); return
    frame = fit_height(frame)
    H = frame.shape[0]
    ui = UI(H)

    # open full-screen (fills the whole screen for screen recording)
    WIN = "Sign Studio"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True

    mode = "recognize"          # recognize | select | teach
    teach_label = None
    last_spoken = None
    stable_label, stable_count = None, 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = fit_height(cv2.flip(frame, 1))
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        word, conf = None, 0.0

        if res.multi_hand_landmarks:
            hlm = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
            vec = landmarks_to_vector([(p.x, p.y, p.z) for p in hlm.landmark])

            if mode == "teach":
                bank.add(vec, teach_label)
                if bank.count(teach_label) >= SAMPLES_PER_SIGN:
                    bank.save(); mode = "recognize"; teach_label = None
            elif mode == "recognize":
                word, conf = bank.predict(vec)

        # speak on stable recognition
        if mode == "recognize":
            if word and word == stable_label: stable_count += 1
            else: stable_label, stable_count = word, 1
            if word and stable_count == STABLE_FRAMES and word != last_spoken:
                say(word); last_spoken = word
            if word is None: last_spoken = None

        state = {"mode": mode, "word": word, "conf": conf,
                 "teach": teach_label, "n": bank.count(teach_label) if teach_label else 0,
                 "counts": {l: bank.count(l) for l in LABELS}}
        panel = ui.panel(state)
        combined = np.hstack([frame, panel])
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord('t'): mode = "select"
        elif key == ord('s'): bank.save()
        elif key == ord('c'): bank.clear(); bank.save()
        elif mode == "select" and ord('1') <= key <= ord('9'):
            i = key - ord('1')
            if i < len(LABELS):
                teach_label = LABELS[i]
                bank.remove_label(teach_label)     # retrain this sign fresh
                mode = "teach"
            else:
                mode = "recognize"

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run()