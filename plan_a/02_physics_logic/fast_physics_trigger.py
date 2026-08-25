"""
02_physics_logic/fast_physics_trigger.py
-----------------------------------------
Fast Physics Threshold Check (Low Latency Layer).
Evaluates peak acceleration, jerk, and rotation rate in real-time.
"""

import math

class FastPhysicsTrigger:
    def __init__(self, peak_accel_thresh=15.0, jerk_thresh=50.0, rotation_thresh=300.0, extreme_thresh=25.0):
        self.peak_accel_thresh = peak_accel_thresh  # g
        self.jerk_thresh = jerk_thresh              # g/s
        self.rotation_thresh = rotation_thresh      # deg/s
        self.extreme_thresh = extreme_thresh        # g (instant hardware override)
        self.prev_accel_mag = 1.0
        self.prev_time = None

    def evaluate_sample(self, sample, current_time):
        """
        Evaluates a single sample or window snapshot.
        Returns (is_crash: bool, reason: str, is_extreme: bool)
        """
        ax, ay, az = sample.get("ax", 0.0), sample.get("ay", 0.0), sample.get("az", 0.0)
        gx, gy, gz = sample.get("gx", 0.0), sample.get("gy", 0.0), sample.get("gz", 0.0)

        # Raw values in g or m/s^2 check
        accel_mag = math.sqrt(ax**2 + ay**2 + az**2)
        if accel_mag > 50.0:  # If values provided in m/s^2, convert to g
            accel_mag /= 9.81

        gyro_mag = math.sqrt(gx**2 + gy**2 + gz**2)

        # Calculate jerk (rate of change of acceleration)
        jerk = 0.0
        if self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                jerk = abs(accel_mag - self.prev_accel_mag) / dt

        self.prev_accel_mag = accel_mag
        self.prev_time = current_time

        # Extreme override check (>25g)
        if accel_mag >= self.extreme_thresh:
            return True, f"EXTREME_PHYSICS_OVERRIDE (accel={accel_mag:.1f}g)", True

        # Tight physics check (at least 2 out of 3 conditions)
        c1 = accel_mag >= self.peak_accel_thresh
        c2 = gyro_mag >= self.rotation_thresh
        c3 = jerk >= self.jerk_thresh

        if (c1 and c2) or (c1 and c3) or (c2 and c3):
            return True, f"TIGHT_PHYSICS_TRIGGER (g={accel_mag:.1f}, gyro={gyro_mag:.1f}, jerk={jerk:.1f})", False

        return False, "NORMAL", False
