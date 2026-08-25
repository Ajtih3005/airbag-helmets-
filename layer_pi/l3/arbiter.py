"""
layer_3/arbiter.py
------------------
Dual-Layer Arbiter & Physics Threshold Trigger.
Bridges ML predictions with fast hardware physics logic.
"""

import math
from collections import deque


class FastPhysicsTrigger:
    """
    Evaluates instant peak acceleration, jerk, and rotation rate in real-time.
    """
    def __init__(self, peak_accel_thresh=15.0, jerk_thresh=50.0, rotation_thresh=300.0, extreme_thresh=25.0):
        self.peak_accel_thresh = peak_accel_thresh  # g
        self.jerk_thresh = jerk_thresh              # g/s
        self.rotation_thresh = rotation_thresh      # deg/s
        self.extreme_thresh = extreme_thresh        # g (instant hardware override)
        self.prev_accel_mag = 1.0
        self.prev_time = None

    def evaluate_sample(self, sample, current_time):
        ax, ay, az = sample.get("ax", 0.0), sample.get("ay", 0.0), sample.get("az", 0.0)
        gx, gy, gz = sample.get("gx", 0.0), sample.get("gy", 0.0), sample.get("gz", 0.0)

        accel_mag = math.sqrt(ax**2 + ay**2 + az**2)
        if accel_mag > 50.0:  # If values in m/s^2, convert to g
            accel_mag /= 9.81

        gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)

        jerk = 0.0
        if self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                jerk = abs(accel_mag - self.prev_accel_mag) / dt

        self.prev_accel_mag = accel_mag
        self.prev_time = current_time

        # Extreme override check (>= 25g)
        if accel_mag >= self.extreme_thresh:
            return True, f"EXTREME_PHYSICS_OVERRIDE (accel={accel_mag:.1f}g)", True

        # Tight physics check (at least 2 out of 3 conditions)
        c1 = accel_mag >= self.peak_accel_thresh
        c2 = gyro_mag >= self.rotation_thresh
        c3 = jerk >= self.jerk_thresh

        if (c1 and c2) or (c1 and c3) or (c2 and c3):
            return True, f"TIGHT_PHYSICS_TRIGGER (g={accel_mag:.1f}, gyro={gyro_mag:.1f}, jerk={jerk:.1f})", False

        return False, "NORMAL", False


class DualLayerArbiter:
    """
    Arbitrates between:
    1. ML Superiority rule (Path 2 window confirmation + Gate).
    2. Fast Physics Tight Condition Override when ML is buffering.
    3. False Positive Cancellation via rolling baseline variance.
    """
    def __init__(self, crash_prob_thresh=0.70, consecutive_req=3, max_blackbox_size=60000):
        self.crash_prob_thresh = crash_prob_thresh
        self.consecutive_req = consecutive_req
        self.consecutive_crash_count = 0
        self.deployed = False
        self.max_blackbox_size = max_blackbox_size
        self.blackbox_accel = deque(maxlen=max_blackbox_size)

    def update_blackbox(self, sample):
        ax = sample.get("ax", 0.0)
        ay = sample.get("ay", 0.0)
        az = sample.get("az", 0.0)
        mag = math.sqrt(ax**2 + ay**2 + az**2)
        if mag > 50.0:
            mag /= 9.81
        self.blackbox_accel.append(mag)

    def _is_false_positive(self, current_mag):
        if len(self.blackbox_accel) < 50:
            return False

        vals = list(self.blackbox_accel)
        mean_val = sum(vals) / len(vals)
        variance = sum((x - mean_val) ** 2 for x in vals) / len(vals)
        std_val = math.sqrt(variance)

        if std_val > 4.0:
            outlier_thresh = mean_val + 3.0 * std_val
            if current_mag < outlier_thresh:
                return True
        return False

    def evaluate(self, physics_result, ml_result=None, current_sample=None):
        if self.deployed:
            return False, "ALREADY_DEPLOYED"

        if current_sample:
            self.update_blackbox(current_sample)

        phys_crash, phys_reason, is_extreme = physics_result

        current_mag = 1.0
        if current_sample:
            ax = current_sample.get("ax", 0.0)
            ay = current_sample.get("ay", 0.0)
            az = current_sample.get("az", 0.0)
            current_mag = math.sqrt(ax**2 + ay**2 + az**2)
            if current_mag > 50.0:
                current_mag /= 9.81

        if is_extreme or phys_crash or (ml_result and ml_result.get("crash_prob", 0) >= self.crash_prob_thresh):
            if self._is_false_positive(current_mag):
                self.consecutive_crash_count = 0
                return False, "FALSE_POSITIVE_CANCELLED"

        # Condition 1: Extreme Physics Override
        if is_extreme:
            self.deployed = True
            return True, phys_reason

        # Condition 2: ML Analysis is Ready
        if ml_result is not None:
            crash_prob = ml_result.get("crash_prob", 0.0)
            pred_label = ml_result.get("label", 0)
            if crash_prob >= self.crash_prob_thresh:
                self.consecutive_crash_count += 1
                if self.consecutive_crash_count >= self.consecutive_req:
                    self.deployed = True
                    return True, f"ML_SUPERIOR_CRASH (prob={crash_prob:.1%}, count={self.consecutive_crash_count})"
                else:
                    return False, f"ML_CRASH_GATE ({self.consecutive_crash_count}/{self.consecutive_req})"
            else:
                self.consecutive_crash_count = 0
                return False, f"ML_NORMAL (p={crash_prob:.1%})"

        # Condition 3: Fast Physics Override while ML is buffering
        if phys_crash:
            self.deployed = True
            return True, f"TIGHT_CONDITION_PHYSICS_OVERRIDE ({phys_reason})"

        return False, "MONITORING"
