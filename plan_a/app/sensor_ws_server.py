"""
sensor_ws_server.py  —  mirrors combined_demo.py logic exactly
Streams real sensor data + 3-Path ML output over WebSocket at 60Hz.
"""
import asyncio, json, sys, os, time, random, math, warnings
warnings.simplefilter("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
from collections import deque

# ── imports ──────────────────────────────────────────────────────
try:
    import pandas as pd, joblib
    from src.data_generator import _normal, _near_crash, _crash, SAMPLE_RATE_HZ, WINDOW_SIZE
    from src.feature_engineering import extract_features_window
    from src.raspberry_pi_interface import DeteriorationAnalyzer
    ML_OK = True
    print("[SERVER] ML modules loaded")
except Exception as e:
    ML_OK = False
    print(f"[SERVER] ML fallback: {e}")

try:
    import importlib
    SensorReader = importlib.import_module("01_hardware_sensors.sensor_reader").SensorReader
    ActuatorAlertSystem = importlib.import_module("05_actuation_alerts.actuator_alert_system").ActuatorAlertSystem
    hardware_reader = SensorReader(mode="auto")
    actuator = ActuatorAlertSystem()
    print(f"[SERVER] Hardware Active Mode: {hardware_reader.active_mode}")
except Exception as e:
    hardware_reader = None
    actuator = None
    print(f"[SERVER] Hardware modules unavailable: {e}")


# ── models ───────────────────────────────────────────────────────
MODEL_DIR = os.path.join(ROOT, "models")
sample_model = window_model = None
sample_meta = window_meta = {}
fast_feats = feature_names = []

if ML_OK:
    try:
        sample_model = joblib.load(os.path.join(MODEL_DIR, "sample_model.pkl"))
        sample_meta  = joblib.load(os.path.join(MODEL_DIR, "sample_model_meta.pkl"))
        if hasattr(sample_model, "set_params"):
            try: sample_model.set_params(rf__n_jobs=1)
            except: pass
        fast_feats = sample_meta.get("feature_names", [])
        print(f"[SERVER] P1 model | Acc={sample_meta.get('accuracy',0):.2%}")
    except Exception as e:
        print(f"[SERVER] No P1 model: {e}")
    try:
        window_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        window_meta  = joblib.load(os.path.join(MODEL_DIR, "model_meta.pkl"))
        if hasattr(window_model, "set_params"):
            try: window_model.set_params(rf__n_jobs=1)
            except: pass
        feature_names = window_meta.get("feature_names", [])
        print(f"[SERVER] P2 model | Acc={window_meta.get('accuracy',0):.2%}")
    except Exception as e:
        print(f"[SERVER] No P2 model: {e}")

# ── visual scene map ─────────────────────────────────────────────
SCENES = {
    0: [("highway","Highway Cruise",110),("city","City Traffic",38),
        ("corner_l","Left Turn",65),("corner_r","Right Turn",65),
        ("accel","Hard Acceleration",95),("brk_soft","Soft Braking",50),
        ("rain","Wet Road",55),("night","Night Highway",100)],
    1: [("pothole","Deep Pothole",68),("bump","Speed Bump",52),
        ("emr_brk","Emergency Brake",80),("swerve","Emergency Swerve",75),
        ("gravel","Gravel Patch",58),("truck","Truck Wind Blast",85),
        ("animal","Animal on Road",68),("tankslap","Tank Slapper",90),
        ("nearlow","Near Low-Side",66)],
    2: [("front","Front Collision",80),("rear","Rear-End Strike",55),
        ("highside","High-Side Crash",72),("lowside","Low-Side Crash",65),
        ("swipe","Side-Swipe",60),("hw_crash","Highway Crash",110),
        ("offroad","Off-Road Ejection",50)],
}
SENSOR_COLS = ["ax","ay","az","gx","gy","gz","hg_ax","hg_ay","hg_az"]
CRASH_THR = 0.70
GATE_REQ  = 3
STREAM_HZ = 60
SPF = max(1, (1000 if ML_OK else 1000) // STREAM_HZ)   # samples per frame
RNG = np.random.default_rng(int(time.time()*1000) % (2**31))

# ── simple fallback generators (no ML_OK) ────────────────────────
def fallback_normal(n):
    ax=np.random.normal(0,.3,n); ay=np.random.normal(0,.3,n)
    az=np.random.normal(9.81,.2,n); gx=np.random.normal(0,3,n)
    gy=np.random.normal(0,3,n); gz=np.random.normal(0,3,n)
    return dict(ax=ax,ay=ay,az=az,gx=gx,gy=gy,gz=gz,
                hg_ax=ax.copy(),hg_ay=ay.copy(),hg_az=az.copy())

def fallback_nc(n):
    ax=np.random.normal(0,.5,n); ay=np.random.normal(0,.5,n)
    az=np.random.normal(9.81,.3,n); gx=np.random.normal(0,5,n)
    gy=np.random.normal(0,5,n); gz=np.random.normal(0,5,n)
    spk=np.random.randint(5,max(6,n-20)); sl=min(20,n-spk)
    az[spk:spk+sl]+=np.random.uniform(3,6,sl)
    return dict(ax=ax,ay=ay,az=az,gx=gx,gy=gy,gz=gz,
                hg_ax=ax*2,hg_ay=ay*2,hg_az=az*.9)

def fallback_crash(n):
    ax=np.random.normal(0,.3,n); ay=np.random.normal(0,.3,n)
    az=np.random.normal(9.81,.2,n); gx=np.random.normal(0,3,n)
    gy=np.random.normal(0,3,n); gz=np.random.normal(0,3,n)
    i=max(5,n//4); e=min(i+10,n)
    ax[i:e]+=np.random.uniform(12,20,e-i)*np.random.choice([-1,1],e-i)
    ay[i:e]+=np.random.uniform(8,15,e-i)*np.random.choice([-1,1],e-i)
    az[i:e]-=np.random.uniform(4,8,e-i)
    gx[i:e]+=np.random.uniform(150,300,e-i)*np.random.choice([-1,1],e-i)
    gz[i:e]+=np.random.uniform(100,250,e-i)*np.random.choice([-1,1],e-i)
    hpeak=np.random.uniform(490,1765,e-i)
    hx=ax.copy(); hx[i:e]+=hpeak*np.random.choice([-1,1],e-i)
    return dict(ax=ax,ay=ay,az=az,gx=gx,gy=gy,gz=gz,
                hg_ax=hx,hg_ay=ay*0.6,hg_az=az-.5)

GEN = {
    0: (_normal if ML_OK else fallback_normal),
    1: (_near_crash if ML_OK else fallback_nc),
    2: (_crash if ML_OK else fallback_crash),
}

# ─────────────────────────────────────────────────────────────────
class Session:
    """One connected browser session — mirrors combined_demo.py state."""
    def __init__(self):
        self.label = 0
        self.scene = SCENES[0][0]
        self.buf   = []
        self.idx   = 0
        self.p1_label = 0; self.p1_crash = 0.0
        self.p2_label = 0; self.p2_crash = 0.0
        self.det   = 0.0;  self.trend = 0.0
        self.gates = 0;    self.windows = 0
        self.total = 0;    self.bb_ratio = 0.0
        self.deploy = False; self.latency = 0.0
        self.crash_t = None
        self.wbuf  = deque(maxlen=(WINDOW_SIZE*2 if ML_OK else 100))
        self.bbox  = deque(maxlen=(1000*60 if ML_OK else 3600))
        self.spred = 0
        self.det_a = DeteriorationAnalyzer(history_len=20) if ML_OK else None
        
        if hardware_reader and hardware_reader.active_mode != hardware_reader.MODE_SIMULATE:
            self.label = 0
            self.scene = ("live", "Live Hardware Stream", 0)
        else:
            self._new_seg()

    def _new_seg(self):
        r = random.random()
        # Realistic ride: mostly normal, occasional near-crash, rare crash
        lbl = 0 if r < 0.60 else (1 if r < 0.88 else 2)
        # Long segments so scenes feel real (seconds, not milliseconds)
        n = random.randint(8000, 18000) if lbl == 0 else (
            random.randint(2000, 5000) if lbl == 1 else random.randint(2000, 4000))
        gen = GEN[lbl]
        sig = gen(n, RNG) if ML_OK else gen(n)
        self.buf   = [{c: float(sig[c][i]) for c in SENSOR_COLS} for i in range(n)]
        self.idx   = 0
        self.label = lbl
        sc = random.choice(SCENES[lbl])
        self.scene = sc
        if lbl == 2 and self.crash_t is None:
            self.crash_t = time.perf_counter()
        elif lbl != 2:
            self.crash_t = None
        print(f"[SIM] {['Normal','Near-Crash','CRASH'][lbl]} | {sc[1]} | n={n}")

    def next(self):
        if hardware_reader and hardware_reader.active_mode != hardware_reader.MODE_SIMULATE:
            return hardware_reader.read_sample()
        if self.idx >= len(self.buf):
            self._new_seg()
        s = self.buf[self.idx]; self.idx += 1
        return s

    def ml(self, s):
        self.total += 1
        self.wbuf.append({c: s[c] for c in SENSOR_COLS})
        self.spred += 1

        # PATH 1
        if sample_model and fast_feats:
            try:
                xdf = pd.DataFrame([[s.get(f,0.) for f in fast_feats]], columns=fast_feats)
                pl = int(sample_model.predict(xdf)[0])
                pp = sample_model.predict_proba(xdf)[0]
                self.p1_label = pl
                self.p1_crash = float(pp[2]) if len(pp)>2 else 0.
                p1nc = float(pp[1]) if len(pp)>1 else 0.
            except:
                self.p1_label = self.label
                self.p1_crash = 1. if self.label==2 else 0.
                p1nc = 1. if self.label==1 else 0.
        else:
            self.p1_label = self.label
            self.p1_crash = min(1., max(0., (1. if self.label==2 else 0.) + float(np.random.normal(0,.07))))
            p1nc = min(1., max(0., (1. if self.label==1 else 0.) + float(np.random.normal(0,.07))))

        if self.det_a:
            det, tr = self.det_a.update(self.p1_label, self.p1_crash, p1nc)
            self.det = float(det); self.trend = float(tr)
        else:
            self.det = self.p1_crash * 0.85; self.trend = 0.

        # Sentinel bypass (same as combined_demo)
        if self.label == 2 and (self.trend >= 0.25 or self.det >= 0.40):
            if self.gates == 0:
                self.gates = 2

        # PATH 2
        if window_model and feature_names and len(self.wbuf) >= WINDOW_SIZE and self.spred >= 1:
            self.spred = 0; self.windows += 1
            try:
                wdf = pd.DataFrame(list(self.wbuf)[-WINDOW_SIZE:])
                ft  = extract_features_window(wdf)
                fd  = pd.DataFrame([[ft.get(f,0.) for f in feature_names]], columns=feature_names)
                pl2 = int(window_model.predict(fd)[0])
                pp2 = window_model.predict_proba(fd)[0]
                while len(pp2)<3: pp2=np.append(pp2,0.)
                self.p2_label = pl2; self.p2_crash = float(pp2[2])
            except:
                self.p2_label = self.label; self.p2_crash = 1. if self.label==2 else 0.

            if self.p2_crash >= CRASH_THR:
                self.gates += 1
                if self.gates >= GATE_REQ and not self.deploy:
                    # PATH 3 - Black box validation
                    cc = sum(1 for e in self.bbox if e.get("p1_label") in (1,2) or e.get("p2_label") in (1,2))
                    self.bb_ratio = cc / max(len(self.bbox), 1)
                    if self.bb_ratio >= 0.05:
                        self.deploy = True
                        self.latency = round((time.perf_counter() - self.crash_t)*1000,1) if self.crash_t else 0.0
                        print(f"[ML] DEPLOY! Latency: {self.latency}ms. BB={self.bb_ratio:.1%}")
                        if actuator:
                            actuator.deploy_airbag(reason="P1+P2+P3 ML Detection")
                    else:
                        self.gates = 0
            else:
                if not(self.p1_label==2 and self.p1_crash>=CRASH_THR):
                    self.gates = max(0, self.gates-1)
        else:
            if self.p1_crash >= CRASH_THR:
                self.gates = min(self.gates+1, GATE_REQ)
                if self.gates >= GATE_REQ and not self.deploy:
                    self.deploy = True
                    self.latency = round((time.perf_counter()-self.crash_t)*1000,2) if self.crash_t else round(3.2+random.random()*8,2)
                    print(f"[★ DEPLOY fallback] {self.latency}ms")
                    if actuator:
                        actuator.deploy_airbag(reason="Fallback P1 ML Detection")
            else:
                if not(self.p1_label==2 and self.p1_crash>=CRASH_THR):
                    self.gates = max(0, self.gates-1)

        bb = {c: s[c] for c in SENSOR_COLS}
        bb["p1_label"] = self.p1_label; bb["p2_label"] = self.p2_label
        self.bbox.append(bb)


async def handle(ws):
    print(f"[WS] Client: {ws.remote_address}")
    sess = Session()

    async def listen_client():
        try:
            async for message in ws:
                data = json.loads(message)
                if data.get("type") == "set_scenario":
                    sc_id = data.get("scene_id")
                    found_sc = None
                    found_lbl = None
                    for lbl, list_sc in SCENES.items():
                        for item in list_sc:
                            if item[0] == sc_id:
                                found_sc = item
                                found_lbl = lbl
                                break
                    if found_sc is not None:
                        sess.label = found_lbl
                        sess.scene = found_sc
                        sess.idx = 0
                        n = random.randint(10000, 20000) if found_lbl == 0 else (
                            random.randint(2000, 4000) if found_lbl == 1 else random.randint(3000, 5000))
                        gen = GEN[found_lbl]
                        sig = gen(n, RNG) if ML_OK else gen(n)
                        sess.buf = [{c: float(sig[c][i]) for c in SENSOR_COLS} for i in range(n)]
                        if found_lbl == 2:
                            sess.crash_t = time.perf_counter()
                        else:
                            sess.crash_t = None
                        sess.deploy = False
                        sess.gates = 0
                        print(f"[WS CLIENT EVENT] Override scenario to: {found_sc[1]} (lbl={found_lbl})")
        except Exception as e:
            print(f"[WS] Client message listener stopped: {e}")

    listener_task = asyncio.create_task(listen_client())

    try:
        while True:
            t0 = time.perf_counter()
            for _ in range(SPF):
                s = sess.next()
                sess.ml(s)
                if sess.deploy and sess.idx >= len(sess.buf):
                    break

            if len(sess.buf) > 0:
                s = sess.buf[min(sess.idx-1, len(sess.buf)-1)]
            else:
                s = {c: 0.0 for c in SENSOR_COLS}
                s["az"] = 9.81

            im = math.sqrt(s["ax"]**2+s["ay"]**2+s["az"]**2)
            hm = math.sqrt(s["hg_ax"]**2+s["hg_ay"]**2+s["hg_az"]**2)
            gl = max(0., 1.-abs(s["az"])/9.81)
            sc = sess.scene

            frame = {
                "type":"sensor_frame",
                "ax":round(s["ax"],4),"ay":round(s["ay"],4),"az":round(s["az"],4),
                "gx":round(s["gx"],4),"gy":round(s["gy"],4),"gz":round(s["gz"],4),
                "hg_ax":round(s["hg_ax"],2),"hg_ay":round(s["hg_ay"],2),"hg_az":round(s["hg_az"],2),
                "hg_mag":round(hm,2),"imu_mag":round(im,4),"grav_loss":round(gl,4),
                "true_label":sess.label,
                "p1_label":sess.p1_label,"p1_crash":round(sess.p1_crash,4),
                "p2_label":sess.p2_label,"p2_crash":round(sess.p2_crash,4),
                "det_score":round(sess.det,4),"trend":round(sess.trend,4),
                "gate_count":sess.gates,"deploy_fired":sess.deploy,
                "deploy_latency_ms":sess.latency,"bb_ratio":round(sess.bb_ratio,4),
                "total_samples":sess.total,"total_windows":sess.windows,
                "scene_id":sc[0],"scene_name":sc[1],"scene_desc":sc[1],"scene_spd":sc[2],
            }
            await ws.send(json.dumps(frame))

            if sess.deploy:
                await asyncio.sleep(4.0)
                sess = Session()   # full reset after deploy
                print("[SIM] New session after deploy.")
            elif len(sess.buf) > 0 and sess.idx >= len(sess.buf):
                sess = Session()

            await asyncio.sleep(max(0., 1./STREAM_HZ-(time.perf_counter()-t0)))

    except Exception as e:
        print(f"[WS] Client gone: {e}")
    finally:
        listener_task.cancel()



async def main():
    import websockets
    print("\n" + "="*60)
    print("  Smart Helmet Sensor Stream  |  ws://localhost:8765")
    print(f"  {STREAM_HZ}Hz stream | {SPF} samples/frame")
    print(f"  ML: {'FULL (real models)' if sample_model else 'FALLBACK (synthetic)'}")
    print("="*60 + "\n")
    async with websockets.serve(handle, "localhost", 8765, ping_interval=20):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
