"""
layer_1/data_generator.py
--------------------------
Dynamic 9-channel IMU sensor data generator for the Smart Airbag Helmet.
Sourced and extended from plan_a/src/data_generator.py.

Hardware simulated:
    MPU6050  -> ax, ay, az (m/s²), gx, gy, gz (deg/s)   -- up to ±16 g
    ADXL377  -> hg_ax, hg_ay, hg_az (m/s²)              -- up to ±200 g (high-impact)

Classes (3-class):
    0 - Normal      : Stable riding, no abnormality
    1 - Near-Crash  : Pothole / Sudden braking / near-miss
    2 - Crash       : Multi-axis impact + sustained chaos
"""

import numpy as np
import pandas as pd

# -------------------------------------------------
#  TIMING CONSTANTS
# -------------------------------------------------
SAMPLE_RATE_HZ = 1000           # 1 kHz sampling
WINDOW_MS      = 50             # 50ms feature window
WINDOW_SIZE    = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)   # 50 samples
STRIDE_MS      = 1
STRIDE         = int(SAMPLE_RATE_HZ * STRIDE_MS / 1000)   # 1 sample

LABEL_MAP = {0: "Normal", 1: "Near-Crash", 2: "Crash"}

SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz", "hg_ax", "hg_ay", "hg_az"]


# =================================================
#  BASE SIGNAL GENERATORS  (label-level)
# =================================================

def _normal(n, rng):
    """Stable highway/city riding at 1000 Hz."""
    ax    = rng.normal(0.0,  0.3,  n)
    ay    = rng.normal(0.0,  0.3,  n)
    az    = rng.normal(9.81, 0.2,  n)
    gx    = rng.normal(0.0,  3.0,  n)
    gy    = rng.normal(0.0,  3.0,  n)
    gz    = rng.normal(0.0,  3.0,  n)
    hg_ax = ax + rng.normal(0.0, 0.05, n)
    hg_ay = ay + rng.normal(0.0, 0.05, n)
    hg_az = az + rng.normal(0.0, 0.05, n)
    return dict(ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz,
                hg_ax=hg_ax, hg_ay=hg_ay, hg_az=hg_az)


def _near_crash(n, rng):
    """Near-crash: pothole OR sudden brake. Moderate spikes."""
    event = rng.choice(["pothole", "brake"])
    if event == "pothole":
        ax = rng.normal(0.0, 0.5, n)
        ay = rng.normal(0.0, 0.5, n)
        az = rng.normal(9.81, 0.3, n)
        gx = rng.normal(0.0, 5.0, n)
        gy = rng.normal(0.0, 5.0, n)
        gz = rng.normal(0.0, 5.0, n)
        ss = rng.integers(5, max(6, n - 20))
        sl = rng.integers(10, min(25, max(11, n - ss - 1)))
        se = min(ss + sl, n)
        az[ss:se] += rng.uniform(3.0, 6.0, se - ss)
        ax[ss:se] += rng.uniform(-1.5, 1.5, se - ss)
        gx[ss:se] += rng.uniform(-20, 20, se - ss)
        gy[ss:se] += rng.uniform(-20, 20, se - ss)
    else:  # brake
        ax = rng.normal(-7.0, 1.5, n)
        ay = rng.normal(0.0,  0.5, n)
        az = rng.normal(9.81, 0.4, n)
        gx = rng.normal(0.0,  5.0, n)
        gy = rng.normal(35.0, 10.0, n)
        gz = rng.normal(0.0,  5.0, n)
        half = n // 2
        if half > 0:
            ramp = np.linspace(0.3, 1.0, half)
            ax[:half] *= ramp
            gy[:half] *= ramp
    hg_ax = ax * rng.uniform(1.0, 2.5, n)
    hg_ay = ay * rng.uniform(1.0, 2.5, n)
    hg_az = az * rng.uniform(0.9, 1.3, n)
    return dict(ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz,
                hg_ax=hg_ax, hg_ay=hg_ay, hg_az=hg_az)


def _crash(n, rng):
    """Full crash: MPU6050 saturates; ADXL377 captures 50–200 g peak."""
    ax = rng.normal(0.0,  0.3, n)
    ay = rng.normal(0.0,  0.3, n)
    az = rng.normal(9.81, 0.2, n)
    gx = rng.normal(0.0,  3.0, n)
    gy = rng.normal(0.0,  3.0, n)
    gz = rng.normal(0.0,  3.0, n)
    impact = rng.integers(10, max(11, n // 3))
    spike  = rng.integers(5, 15)
    end    = min(impact + spike, n)
    ax[impact:end] += rng.uniform(12, 20,  end - impact) * rng.choice([-1, 1], end - impact)
    ay[impact:end] += rng.uniform(8,  15,  end - impact) * rng.choice([-1, 1], end - impact)
    az[impact:end] += rng.uniform(-8, -4,  end - impact)
    gx[impact:end] += rng.uniform(150, 300, end - impact) * rng.choice([-1, 1], end - impact)
    gy[impact:end] += rng.uniform(150, 300, end - impact) * rng.choice([-1, 1], end - impact)
    gz[impact:end] += rng.uniform(100, 250, end - impact) * rng.choice([-1, 1], end - impact)
    if end < n:
        ax[end:] += rng.normal(5, 3, n - end)
        gz[end:] += rng.normal(50, 20, n - end)
    hg_ax = ax.copy()
    hg_ay = ay.copy()
    hg_az = az.copy()
    if end > impact:
        hg_peak = rng.uniform(490, 1765, end - impact)
        hg_ax[impact:end] += hg_peak * rng.choice([-1, 1], end - impact)
        hg_ay[impact:end] += hg_peak * 0.6 * rng.choice([-1, 1], end - impact)
        hg_az[impact:end] -= rng.uniform(100, 400, end - impact)
    return dict(ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz,
                hg_ax=hg_ax, hg_ay=hg_ay, hg_az=hg_az)


# =================================================
#  NAMED EVENT GENERATORS
#  These are called by the LLM generator by event name.
# =================================================

def _event_pothole(n, rng):
    """Isolated pothole strike — sharp vertical spike."""
    base = _normal(n, rng)
    ss = max(5, n // 4)
    sl = min(20, n - ss - 1)
    se = ss + sl
    base["az"][ss:se] += rng.uniform(4.0, 8.0, se - ss)
    base["ax"][ss:se] += rng.uniform(-2.0, 2.0, se - ss)
    base["gx"][ss:se] += rng.uniform(-30, 30, se - ss)
    base["hg_az"][ss:se] += rng.uniform(5.0, 12.0, se - ss)
    return base


def _event_swerve(n, rng):
    """Hard lateral swerve — high lateral accel and roll rate."""
    base = _normal(n, rng)
    ss = max(5, n // 5)
    se = min(ss + n // 3, n)
    base["ay"][ss:se] += rng.uniform(3.0, 7.0, se - ss) * rng.choice([-1, 1], se - ss)
    base["gz"][ss:se] += rng.uniform(40, 120, se - ss) * rng.choice([-1, 1], se - ss)
    base["hg_ay"][ss:se] += rng.uniform(4.0, 10.0, se - ss)
    return base


def _event_brake(n, rng):
    """Emergency braking — strong deceleration spike in ax."""
    base = _normal(n, rng)
    ss = max(5, n // 5)
    se = min(ss + n // 2, n)
    base["ax"][ss:se] += rng.uniform(-10.0, -5.0, se - ss)
    base["gy"][ss:se] += rng.uniform(20, 60, se - ss)
    return base


def _event_accel(n, rng):
    """Hard acceleration — forward pull in ax."""
    base = _normal(n, rng)
    ss = max(5, n // 5)
    se = min(ss + n // 2, n)
    base["ax"][ss:se] += rng.uniform(3.0, 8.0, se - ss)
    base["gy"][ss:se] += rng.uniform(-30, -10, se - ss)
    return base


def _event_highside(n, rng):
    """High-side crash — violent roll + high-G."""
    base = _crash(n, rng)
    base["ay"] += rng.uniform(8, 15, n) * rng.choice([-1, 1], n)
    base["gz"] += rng.uniform(100, 200, n) * rng.choice([-1, 1], n)
    return base


def _event_lowside(n, rng):
    """Low-side crash — sliding lateral impact."""
    base = _crash(n, rng)
    base["ay"] += rng.uniform(5, 12, n)
    base["gz"] += rng.uniform(80, 160, n)
    return base


def _event_front_collision(n, rng):
    """Head-on frontal impact — extreme forward deceleration."""
    base = _crash(n, rng)
    base["ax"] += rng.uniform(15, 25, n) * rng.choice([-1, 1], n)
    base["hg_ax"] += rng.uniform(200, 600, n) * rng.choice([-1, 1], n)
    return base


def _event_rear_collision(n, rng):
    """Rear-end strike — sudden forward lurch."""
    base = _crash(n, rng)
    base["ax"] += rng.uniform(10, 20, n)
    base["gy"] += rng.uniform(-80, -40, n)
    return base


def _event_gravel(n, rng):
    """Gravel slip — sustained oscillating lateral forces."""
    base = _near_crash(n, rng)
    base["ay"] += np.sin(np.linspace(0, 8 * np.pi, n)) * rng.uniform(1.5, 4.0)
    base["gz"] += np.sin(np.linspace(0, 6 * np.pi, n)) * rng.uniform(20, 50)
    return base


def _event_rain(n, rng):
    """Wet road riding — higher random vibration, reduced traction."""
    base = _normal(n, rng)
    base["ax"] += rng.normal(0, 1.5, n)
    base["ay"] += rng.normal(0, 1.5, n)
    base["gz"] += rng.normal(0, 15, n)
    return base


# Registry: map event name (string) -> generator function + label
EVENT_REGISTRY = {
    # Normal events (label 0)
    "normal":          (_normal,              0),
    "highway":         (_normal,              0),
    "city":            (_normal,              0),
    "accel":           (_event_accel,         0),
    "rain":            (_event_rain,          0),

    # Near-crash events (label 1)
    "near_crash":      (_near_crash,          1),
    "pothole":         (_event_pothole,       1),
    "swerve":          (_event_swerve,        1),
    "brake":           (_event_brake,         1),
    "gravel":          (_event_gravel,        1),

    # Crash events (label 2)
    "crash":           (_crash,               2),
    "highside":        (_event_highside,      2),
    "lowside":         (_event_lowside,       2),
    "front_collision": (_event_front_collision, 2),
    "rear_collision":  (_event_rear_collision,  2),
}

_BASE_GENERATORS = {0: _normal, 1: _near_crash, 2: _crash}


# =================================================
#  CORE TELEMETRY BUILDER
# =================================================

def generate_from_segments(segments, rng=None):
    """
    Generate a full 9-channel sensor DataFrame from a list of segments.

    Parameters
    ----------
    segments : list of dict
        Each dict must have:
            "event"    : str  — event name from EVENT_REGISTRY (e.g. "pothole")
            "duration_ms" : int — segment duration in milliseconds

    rng : numpy.random.Generator, optional

    Returns
    -------
    pd.DataFrame with columns:
        timestamp_ms, ax, ay, az, gx, gy, gz, hg_ax, hg_ay, hg_az, label, label_name
    """
    if rng is None:
        rng = np.random.default_rng()

    all_rows = []
    t_ms = 0.0
    dt_ms = 1000.0 / SAMPLE_RATE_HZ

    for seg in segments:
        event_name   = seg.get("event", "normal").lower().replace(" ", "_").replace("-", "_")
        duration_ms  = int(seg.get("duration_ms", 500))
        n_samples    = int(SAMPLE_RATE_HZ * duration_ms / 1000)

        if n_samples <= 0:
            continue

        if event_name in EVENT_REGISTRY:
            gen_fn, label = EVENT_REGISTRY[event_name]
        else:
            # Unknown event: fallback to normal
            print(f"[WARNING] Unknown event '{event_name}' — using Normal.")
            gen_fn, label = _normal, 0

        sig = gen_fn(n_samples, rng)

        for i in range(n_samples):
            all_rows.append({
                "timestamp_ms": round(t_ms, 3),
                "ax":           round(float(sig["ax"][i]),    4),
                "ay":           round(float(sig["ay"][i]),    4),
                "az":           round(float(sig["az"][i]),    4),
                "gx":           round(float(sig["gx"][i]),    4),
                "gy":           round(float(sig["gy"][i]),    4),
                "gz":           round(float(sig["gz"][i]),    4),
                "hg_ax":        round(float(sig["hg_ax"][i]), 4),
                "hg_ay":        round(float(sig["hg_ay"][i]), 4),
                "hg_az":        round(float(sig["hg_az"][i]), 4),
                "label":        label,
                "label_name":   LABEL_MAP[label],
            })
            t_ms += dt_ms

    return pd.DataFrame(all_rows)


def generate_procedural(total_duration_ms=5000, rng=None):
    """
    Procedurally build a randomised ride scenario and generate sensor data.
    Always starts and ends with Normal riding. Picks events in between randomly.

    Returns (segments_plan, DataFrame)
    """
    if rng is None:
        rng = np.random.default_rng()

    normal_events   = ["normal", "highway", "city", "accel", "rain"]
    nc_events       = ["pothole", "swerve", "brake", "gravel"]
    crash_events    = ["crash", "highside", "lowside", "front_collision", "rear_collision"]

    segments = []
    remaining_ms = total_duration_ms

    # Opening normal segment
    open_dur = int(rng.integers(400, 1000))
    segments.append({"event": rng.choice(normal_events), "duration_ms": open_dur})
    remaining_ms -= open_dur

    # Random middle events
    while remaining_ms > 400:
        roll = rng.random()
        if roll < 0.50:
            event = str(rng.choice(normal_events))
            dur   = int(rng.integers(300, 800))
        elif roll < 0.85:
            event = str(rng.choice(nc_events))
            dur   = int(rng.integers(200, 500))
        else:
            event = str(rng.choice(crash_events))
            dur   = int(rng.integers(300, 600))

        dur = min(dur, remaining_ms - 200)
        if dur <= 0:
            break
        segments.append({"event": event, "duration_ms": dur})
        remaining_ms -= dur

    # Closing normal segment
    if remaining_ms > 0:
        segments.append({"event": "normal", "duration_ms": remaining_ms})

    df = generate_from_segments(segments, rng)
    return segments, df
