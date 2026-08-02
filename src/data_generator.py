"""
data_generator.py
-----------------
Synthetic IMU time-series generator for Smart Airbag Helmet project.

Hardware simulated:
    MPU6050  -> ax, ay, az (m/s2), gx, gy, gz (deg/s)   -- up to +-16 g
    ADXL377  -> hg_ax, hg_ay, hg_az (m/s2)              -- up to +-200 g (high-impact)

Timing:
    SAMPLE_RATE_HZ = 1000           (1 kHz, per spec >=1000 Hz)
    WINDOW_MS      = 200            (200 ms rolling window per spec)
    WINDOW_SIZE    = 200 samples    (= SAMPLE_RATE_HZ * WINDOW_MS / 1000)

Classes (3-class as per project spec):
    0 - Normal     : Stable riding, no abnormality
    1 - Near-Crash : Pothole / Sudden braking / near-miss
    2 - Crash      : Multi-axis impact + sustained chaos
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# -------------------------------------------------
#  TIMING CONSTANTS (shared across all modules)
# -------------------------------------------------
SAMPLE_RATE_HZ = 1000          # Sensor sampling frequency (spec: >=1000 Hz)
WINDOW_MS      = 200           # Feature window duration in ms (spec: 200 ms)
WINDOW_SIZE    = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)   # = 200 samples
STRIDE_MS      = 100           # Stride between windows in ms
STRIDE         = int(SAMPLE_RATE_HZ * STRIDE_MS / 1000)   # = 100 samples

# -------------------------------------------------
#  LABEL MAP  (3-class per project spec)
# -------------------------------------------------
LABEL_MAP = {
    0: "Normal",
    1: "Near-Crash",
    2: "Crash"
}

# -------------------------------------------------
#  SIGNAL GENERATORS  (per class)
# -------------------------------------------------

def _normal(n, rng):
    """Stable riding at 1000 Hz."""
    ax = rng.normal(0.0,  0.3,  n)
    ay = rng.normal(0.0,  0.3,  n)
    az = rng.normal(9.81, 0.2,  n)
    gx = rng.normal(0.0,  3.0,  n)
    gy = rng.normal(0.0,  3.0,  n)
    gz = rng.normal(0.0,  3.0,  n)
    hg_ax = ax + rng.normal(0.0, 0.05, n)
    hg_ay = ay + rng.normal(0.0, 0.05, n)
    hg_az = az + rng.normal(0.0, 0.05, n)
    return {"ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
            "hg_ax": hg_ax, "hg_ay": hg_ay, "hg_az": hg_az}


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
        spike_start = rng.integers(5, max(6, n - 20))
        spike_len   = rng.integers(10, min(25, max(11, n - spike_start - 1)))
        spike_end   = min(spike_start + spike_len, n)
        az[spike_start:spike_end] += rng.uniform(3.0, 6.0, spike_end - spike_start)
        ax[spike_start:spike_end] += rng.uniform(-1.5, 1.5, spike_end - spike_start)
        gx[spike_start:spike_end] += rng.uniform(-20, 20, spike_end - spike_start)
        gy[spike_start:spike_end] += rng.uniform(-20, 20, spike_end - spike_start)
    else:
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
    return {"ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
            "hg_ax": hg_ax, "hg_ay": hg_ay, "hg_az": hg_az}


def _crash(n, rng):
    """Actual crash. MPU6050 saturates; ADXL377 captures 50-180 g peak."""
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
    return {"ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz,
            "hg_ax": hg_ax, "hg_ay": hg_ay, "hg_az": hg_az}


_GENERATORS = {0: _normal, 1: _near_crash, 2: _crash}

# -------------------------------------------------
#  SESSION GENERATOR
# -------------------------------------------------

def generate_session(session_id, session_duration_ms=2000, rng=None):
    """
    Generate one riding session at SAMPLE_RATE_HZ (1000 Hz).

    Returns DataFrame:
        session_id, timestamp_ms, ax, ay, az, gx, gy, gz,
        hg_ax, hg_ay, hg_az, label, label_name
    """
    if rng is None:
        rng = np.random.default_rng()
    session_len = int(SAMPLE_RATE_HZ * session_duration_ms / 1000)
    n_segments  = rng.integers(3, 7)
    remaining   = session_len
    segments    = []

    seg_len = rng.integers(200, 600)
    segments.append((0, min(seg_len, remaining)))
    remaining -= segments[-1][1]

    event_probs = [0.50, 0.30, 0.20]
    for _ in range(n_segments - 2):
        if remaining <= 0:
            break
        label   = rng.choice([0, 1, 2], p=event_probs)
        seg_len = rng.integers(100, 400)
        segments.append((label, min(seg_len, remaining)))
        remaining -= segments[-1][1]

    if remaining > 0:
        segments.append((0, remaining))

    all_rows = []
    t_ms = 0.0
    dt_ms = 1000.0 / SAMPLE_RATE_HZ

    for label, length in segments:
        if length <= 0:
            continue
        sig = _GENERATORS[label](length, rng)
        for i in range(length):
            all_rows.append({
                "session_id"  : session_id,
                "timestamp_ms": round(t_ms, 3),
                "ax"          : round(float(sig["ax"][i]),    4),
                "ay"          : round(float(sig["ay"][i]),    4),
                "az"          : round(float(sig["az"][i]),    4),
                "gx"          : round(float(sig["gx"][i]),    4),
                "gy"          : round(float(sig["gy"][i]),    4),
                "gz"          : round(float(sig["gz"][i]),    4),
                "hg_ax"       : round(float(sig["hg_ax"][i]), 4),
                "hg_ay"       : round(float(sig["hg_ay"][i]), 4),
                "hg_az"       : round(float(sig["hg_az"][i]), 4),
                "label"       : label,
                "label_name"  : LABEL_MAP[label]
            })
            t_ms += dt_ms

    return pd.DataFrame(all_rows)


# -------------------------------------------------
#  DATASET GENERATOR
# -------------------------------------------------

def generate_dataset(n_sessions=300, session_duration_ms=2000,
                     seed=42, save_path=None, verbose=True):
    """
    Generate full dataset of n_sessions riding sessions.
    Default: 300 sessions x 2000 ms = 600,000 samples at 1000 Hz.
    """
    rng = np.random.default_rng(seed)
    frames = []
    if verbose:
        print(f"Generating {n_sessions} sessions at {SAMPLE_RATE_HZ} Hz "
              f"({session_duration_ms} ms each)...")
    for i in range(n_sessions):
        df = generate_session(session_id=i,
                              session_duration_ms=session_duration_ms,
                              rng=rng)
        frames.append(df)
        if verbose and (i + 1) % 50 == 0:
            print(f"  Sessions generated: {i + 1}/{n_sessions}")
    dataset = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"\nDataset shape      : {dataset.shape}")
        print(f"Sample rate        : {SAMPLE_RATE_HZ} Hz")
        print(f"Window size        : {WINDOW_SIZE} samples ({WINDOW_MS} ms)")
        print(f"\nLabel distribution:")
        counts = dataset["label_name"].value_counts()
        total  = len(dataset)
        for name, cnt in counts.items():
            print(f"  {name:<15} {cnt:>8}  ({cnt/total*100:.1f}%)")
    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        dataset.to_csv(save_path, index=False)
        if verbose:
            print(f"\nSaved to: {save_path}")
    return dataset


if __name__ == "__main__":
    import os
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "synthetic", "helmet_imu_raw.csv"
    )
    generate_dataset(n_sessions=300, session_duration_ms=2000,
                     seed=42, save_path=out_path)
