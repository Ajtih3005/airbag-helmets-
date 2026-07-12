"""
data_generator.py
-----------------
Synthetic IMU time-series generator for Smart Airbag Helmet project.

Generates realistic riding sessions with the following classes:
    0 - Normal Riding
    1 - Pothole / Road Bump
    2 - Sudden Braking
    3 - Crash / Fall

Each session is a sequence of timesteps where the label can transition,
mimicking real-world riding dynamics (e.g., normal → brake → crash).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# ─────────────────────────────────────────────
#  LABEL MAP
# ─────────────────────────────────────────────
LABEL_MAP = {
    0: "Normal",
    1: "Pothole",
    2: "SuddenBrake",
    3: "Crash"
}

# ─────────────────────────────────────────────
#  BASE SIGNAL GENERATORS (per label)
# ─────────────────────────────────────────────

def _normal(n: int, rng: np.random.Generator) -> dict:
    """Stable riding. Low noise, gravity dominant on az."""
    return {
        "ax": rng.normal(0.0,  0.3,  n),
        "ay": rng.normal(0.0,  0.3,  n),
        "az": rng.normal(9.81, 0.2,  n),
        "gx": rng.normal(0.0,  3.0,  n),
        "gy": rng.normal(0.0,  3.0,  n),
        "gz": rng.normal(0.0,  3.0,  n),
    }


def _pothole(n: int, rng: np.random.Generator) -> dict:
    """
    Sudden vertical bump (az spike ±4–6 g) for ~3–5 samples,
    lateral shake (ax, ay), mild gyro spike.
    """
    ax = rng.normal(0.0, 0.5, n)
    ay = rng.normal(0.0, 0.5, n)
    az = rng.normal(9.81, 0.3, n)
    gx = rng.normal(0.0, 5.0, n)
    gy = rng.normal(0.0, 5.0, n)
    gz = rng.normal(0.0, 5.0, n)

    # Insert spike at random position
    spike_start = rng.integers(2, max(3, n - 5))
    spike_len   = rng.integers(3, 6)
    spike_end   = min(spike_start + spike_len, n)

    az[spike_start:spike_end] += rng.uniform(4.0, 7.0, spike_end - spike_start)
    ax[spike_start:spike_end] += rng.uniform(-1.5, 1.5, spike_end - spike_start)
    ay[spike_start:spike_end] += rng.uniform(-1.5, 1.5, spike_end - spike_start)
    gx[spike_start:spike_end] += rng.uniform(-20, 20, spike_end - spike_start)
    gy[spike_start:spike_end] += rng.uniform(-20, 20, spike_end - spike_start)

    return {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz}


def _sudden_brake(n: int, rng: np.random.Generator) -> dict:
    """
    Strong forward deceleration → ax goes strongly negative.
    Head pitches forward → gy increases.
    """
    ax = rng.normal(-8.0, 1.5, n)   # Strong forward deceleration
    ay = rng.normal(0.0,  0.5, n)
    az = rng.normal(9.81, 0.4, n)
    gx = rng.normal(0.0,  5.0, n)
    gy = rng.normal(40.0, 10.0, n)  # Pitch forward
    gz = rng.normal(0.0,  5.0, n)

    # Ramp-up: brake gets stronger in first half
    ramp = np.linspace(0.3, 1.0, n // 2)
    ax[:n // 2] *= ramp
    gy[:n // 2] *= ramp

    return {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz}


def _crash(n: int, rng: np.random.Generator) -> dict:
    """
    Multi-axis large spike followed by sustained deviation.
    All 6 DOF go haywire. Gravity vector lost (az drops).
    """
    ax = rng.normal(0.0,  0.3, n)
    ay = rng.normal(0.0,  0.3, n)
    az = rng.normal(9.81, 0.2, n)
    gx = rng.normal(0.0,  3.0, n)
    gy = rng.normal(0.0,  3.0, n)
    gz = rng.normal(0.0,  3.0, n)

    # Impact point — large multi-axis spike
    impact = rng.integers(5, max(6, n // 3))
    spike  = rng.integers(4, 8)
    end    = min(impact + spike, n)

    ax[impact:end] += rng.uniform(12, 20,  end - impact) * rng.choice([-1, 1], end - impact)
    ay[impact:end] += rng.uniform(8,  15,  end - impact) * rng.choice([-1, 1], end - impact)
    az[impact:end] += rng.uniform(-8, -4,  end - impact)   # Az drops (gravity confusion)
    gx[impact:end] += rng.uniform(150, 300, end - impact) * rng.choice([-1, 1], end - impact)
    gy[impact:end] += rng.uniform(150, 300, end - impact) * rng.choice([-1, 1], end - impact)
    gz[impact:end] += rng.uniform(100, 250, end - impact) * rng.choice([-1, 1], end - impact)

    # Post-crash: sustained chaos
    if end < n:
        ax[end:] += rng.normal(5, 3, n - end)
        gz[end:] += rng.normal(50, 20, n - end)

    return {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz}


_GENERATORS = {
    0: _normal,
    1: _pothole,
    2: _sudden_brake,
    3: _crash
}

# ─────────────────────────────────────────────
#  SESSION GENERATOR
# ─────────────────────────────────────────────

def generate_session(
    session_id: int,
    session_len: int = 100,
    rng: np.random.Generator = None
) -> pd.DataFrame:
    """
    Generate one riding session as a realistic sequence of segments.

    A session is built from 2–6 segments.  Each segment picks a class
    and generates sensor readings for that class for a random duration.

    Returns a DataFrame with columns:
        session_id, timestamp, ax, ay, az, gx, gy, gz, label
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Build segment plan ──────────────────────────────────────────
    # Always start with Normal, may transition through events, can end in Crash
    n_segments = rng.integers(3, 7)
    remaining  = session_len

    segments: List[Tuple[int, int]] = []   # (label, length)

    # First segment always Normal
    seg_len = rng.integers(10, 30)
    segments.append((0, min(seg_len, remaining)))
    remaining -= segments[-1][1]

    # Middle segments — weighted toward Normal, but allow events
    event_probs = [0.55, 0.15, 0.15, 0.15]   # Normal, Pothole, Brake, Crash
    for _ in range(n_segments - 2):
        if remaining <= 0:
            break
        label   = rng.choice([0, 1, 2, 3], p=event_probs)
        seg_len = rng.integers(8, 25)
        segments.append((label, min(seg_len, remaining)))
        remaining -= segments[-1][1]

    # Last segment: fill remaining
    if remaining > 0:
        segments.append((0, remaining))

    # ── Generate sensor data for each segment ──────────────────────
    all_rows = []
    t = 0
    for label, length in segments:
        if length <= 0:
            continue
        gen_fn = _GENERATORS[label]
        sig    = gen_fn(length, rng)

        for i in range(length):
            all_rows.append({
                "session_id": session_id,
                "timestamp":  t,
                "ax":  round(sig["ax"][i], 4),
                "ay":  round(sig["ay"][i], 4),
                "az":  round(sig["az"][i], 4),
                "gx":  round(sig["gx"][i], 4),
                "gy":  round(sig["gy"][i], 4),
                "gz":  round(sig["gz"][i], 4),
                "label": label,
                "label_name": LABEL_MAP[label]
            })
            t += 1

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────
#  DATASET GENERATOR
# ─────────────────────────────────────────────

def generate_dataset(
    n_sessions: int    = 500,
    session_len: int   = 100,
    seed: int          = 42,
    save_path: str     = None,
    verbose: bool      = True
) -> pd.DataFrame:
    """
    Generate full dataset of n_sessions riding sessions.

    Parameters
    ----------
    n_sessions : int
        Number of independent riding sessions.
    session_len : int
        Approximate number of timesteps per session.
    seed : int
        Random seed for reproducibility.
    save_path : str
        If provided, saves CSV to this path.
    verbose : bool
        Print progress info.

    Returns
    -------
    pd.DataFrame with all sessions concatenated.
    """
    rng = np.random.default_rng(seed)
    frames = []

    for i in range(n_sessions):
        df = generate_session(session_id=i, session_len=session_len, rng=rng)
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)

    if verbose:
        print(f"Dataset shape    : {dataset.shape}")
        print(f"Total sessions   : {n_sessions}")
        print(f"\nLabel distribution:")
        counts = dataset["label_name"].value_counts()
        total  = len(dataset)
        for name, cnt in counts.items():
            print(f"  {name:<15} {cnt:>6}  ({cnt/total*100:.1f}%)")

    if save_path:
        dataset.to_csv(save_path, index=False)
        print(f"\nSaved to: {save_path}")

    return dataset


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "helmet_imu_raw.csv")
    generate_dataset(n_sessions=500, session_len=100, seed=42, save_path=out_path)
