#!/usr/bin/env python3
"""Offline crop advisory web app.

Photo -> vision model (ONNX Runtime, CPU) -> severity (OpenCV) -> verified advice card in 4 languages.
Optional extras: Llama explanation (ExecuTorch runner, guarded) and spoken advice (espeak-ng).
Everything runs on the device; nothing here talks to the internet.
"""
import base64
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from PIL import Image



BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))



def P(s):
    return Path(os.path.expandvars(os.path.expanduser(s)))


ADVISORY, CARDS = P(CFG["advisory_dir"]), P(CFG["cards_dir"])
sys.path.insert(0, str(ADVISORY))
from severity import estimate_severity  # noqa: E402
try:
    from leafcrop import leaf_crop  # noqa: E402
except ImportError:
    leaf_crop = None

SEVS = ["mild", "moderate", "severe"]
SPEAK_VOICE = {"en": "en", "hi": "hi", "bn": "bn", "as": "as"}
CARD_KEYS = ("names", "summary", "symptoms", "favourable_conditions", "actions_by_severity",
             "organic_options", "prevention", "escalate_when", "status", "reviewed_by")


def pretty(cls):
    return cls.replace("_", " ")


def read_classes_txt():
    p = P(CFG["classes_txt"])
    return p.read_text(encoding="utf-8").split() if p.exists() else []


def load_card(cls):
    p = CARDS / f"{cls}.json"
    if not p.exists():
        return None
    try:
        card = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {k: card[k] for k in CARD_KEYS if k in card}


class Vision:
    """ONNX Runtime classifier. If no trained model is present the app runs in DEMO mode and says so."""

    def __init__(self):
        self.demo, self.sess, self.temp, self.crop = True, None, 1.0, False
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        mp, cp, pp = P(CFG["model_path"]), P(CFG["classes_path"]), P(CFG["preprocess_path"])
        self.classes = read_classes_txt()
        if mp.exists() and cp.exists():
            try:
                import onnxruntime as ort
                self.sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
                self.classes = json.loads(cp.read_text(encoding="utf-8"))
                if pp.exists():
                    pre = json.loads(pp.read_text(encoding="utf-8"))
                    self.mean = np.array(pre.get("mean", self.mean), np.float32)
                    self.std = np.array(pre.get("std", self.std), np.float32)
                    self.temp = float(pre.get("temperature", 1.0))
                    self.crop = bool(pre.get("leaf_crop", False))
                self.demo = False
                print(f"[vision] loaded {mp.name}, {len(self.classes)} classes, temperature {self.temp:.2f}")
            except Exception as e:
                print("[vision] could not load the model, running in DEMO mode:", e)
        if self.demo:
            print("[vision] DEMO MODE: no trained model loaded, results are not real")

    def preprocess(self, bgr):
        if self.crop and leaf_crop is not None:        # same leaf crop as in training
            bgr, _ = leaf_crop(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)   # same resize as in training
        x = (np.asarray(im, np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def predict(self, bgr, demo_class=None):
        if self.demo:
            cls = demo_class if demo_class in self.classes else CFG["demo_default_class"]
            others = [c for c in self.classes if c != cls][:2] or ["unknown_not_a_leaf"]
            return [(cls, 0.87), (others[0], 0.08), (others[-1], 0.03)]
        logits = self.sess.run(None, {self.sess.get_inputs()[0].name: self.preprocess(bgr)})[0][0] / self.temp
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return [(self.classes[i], float(p[i])) for i in np.argsort(-p)[:3]]


VISION = Vision()
LLAMA_LOCK = threading.Lock()
app = Flask(__name__, static_folder=None)


def decide(top, th):
    cls, p = top[0]
    if cls == "unknown_not_a_leaf":
        return "retake", "not_a_leaf"
    if p < th["possible"]:
        return "retake", "low_confidence"
    if cls.endswith("_healthy"):
        return ("healthy" if p >= th["show"] else "possible"), None
    return ("show_card" if p >= th["show"] else "possible"), None


def jpeg_b64(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/")
def index():
    return send_from_directory(BASE / "templates", "index.html")


@app.get("/api/config")
def config():
    llama_ok = False
    try:
        import llama_runner
        llama_ok = Path(llama_runner.RUNNER).exists() and Path(llama_runner.MODEL).exists()
    except Exception:
        pass
    return jsonify(demo=VISION.demo, contacts=CFG["contacts"], thresholds=CFG["thresholds"],
                   classes=VISION.classes, speak=bool(shutil.which("espeak-ng")), llama=llama_ok,
                   picam=bool(shutil.which("rpicam-still") or shutil.which("libcamera-still")))


@app.post("/api/analyze")
def analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="no image"), 400
    bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="cannot read image"), 400
    h, w = bgr.shape[:2]
    if max(h, w) > 1600:
        s = 1600.0 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    t0 = time.time()
    top = VISION.predict(bgr, request.form.get("demo_class"))
    t1 = time.time()
    route, reason = decide(top, CFG["thresholds"])
    cls, conf = top[0]
    out = dict(route=route, reason=reason, cls=cls, confidence=conf, demo=VISION.demo,
               top3=[dict(cls=c, p=p, names=(load_card(c) or {}).get("names") or {"en": pretty(c)}) for c, p in top],
               severity=None, severity_used=None, severe_alert=False, overlay=None, card=None,
               ms=dict(vision=round((t1 - t0) * 1000)))

    if route in ("show_card", "possible") and not cls.endswith("_healthy"):
        ts = time.time()
        sev = estimate_severity(bgr)
        out["ms"]["severity"] = round((time.time() - ts) * 1000)
        out["severity"] = dict(pct=sev["pct"], bucket=sev["bucket"], method=sev["method"])
        out["severity_used"] = sev["bucket"] if sev["bucket"] in SEVS else "moderate"
        out["severe_alert"] = out["severity_used"] == "severe"
        out["overlay"] = jpeg_b64(sev["overlay"])
    if route in ("show_card", "possible"):
        out["card"] = load_card(cls)
        if out["card"] is None:
            out["route"] = "no_card"
            out["severe_alert"] = False
    return jsonify(out)


@app.post("/api/capture")
def capture():
    """Take a photo with the Raspberry Pi camera and return the JPEG."""
    exe = shutil.which("rpicam-still") or shutil.which("libcamera-still")
    if not exe:
        return jsonify(error="no Raspberry Pi camera tool found"), 501
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    base = [exe, "-n", "-t", "1500", "--width", "2304", "--height", "1296", "-o", path]
    af = ["--autofocus-mode", "auto", "--autofocus-on-capture"]          # Camera Module 3 has autofocus
    try:
        try:
            subprocess.run(base + af, check=True, timeout=25, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(base, check=True, timeout=25, capture_output=True)   # camera without autofocus options
        data = Path(path).read_bytes()
    except Exception as e:
        return jsonify(error=f"camera capture failed: {e}"), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/explain")
def explain():
    """Optional Llama explanation. The advice card stays the source of truth; this text is guarded and may be hidden."""
    j = request.get_json(force=True, silent=True) or {}
    try:
        import pipeline
    except Exception as e:
        return jsonify(available=False, error=f"advisory pipeline not available: {e}")
    sev = j.get("severity") if j.get("severity") in SEVS else "moderate"
    msg = dict(crop=str(j.get("cls", "")).split("_")[0], condition_id=j.get("cls", ""),
               confidence=float(j.get("confidence", 0.8)), severity=sev, language=j.get("lang", "en"),
               month=datetime.date.today().month, growth_stage="unknown")
    with LLAMA_LOCK:
        try:
            r = pipeline.advise(msg)
        except Exception as e:
            return jsonify(available=False, error=str(e))
    return jsonify(available=True, text=r.get("explanation"), lang=r.get("explanation_lang"),
                   guard=r.get("guard_reason"), seconds=r.get("seconds"))


@app.get("/api/speak")
def speak():
    exe = shutil.which("espeak-ng")
    if not exe:
        return jsonify(error="espeak-ng is not installed"), 501
    text = request.args.get("text", "").strip()[:900]
    if not text:
        return jsonify(error="no text"), 400
    voice = SPEAK_VOICE.get(request.args.get("lang", "en"), "en")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run([exe, "-v", voice, "-s", "140", "-w", path, text], check=True, timeout=30)
        data = Path(path).read_bytes()
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return Response(data, mimetype="audio/wav", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    app.run(host=CFG.get("host", "127.0.0.1"), port=int(CFG.get("port", 5000)), threaded=True)
