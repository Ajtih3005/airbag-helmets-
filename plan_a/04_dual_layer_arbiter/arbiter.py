"""
04_dual_layer_arbiter/arbiter.py
----------------------------------
Dual-Layer Arbiter & Decision Engine.
Implements arbitration logic:
1. ML Superiority rule when window is processed.
2. Fast Physics Tight Condition Override when ML is still buffering or evaluating.
"""

import logging

log = logging.getLogger("arbiter")

import logging
import math
from collections import deque

log = logging.getLogger("arbiter")

class DualLayerArbiter:
    def __init__(self, crash_prob_thresh=0.70, consecutive_req=3, max_blackbox_size=60000):
        self.crash_prob_thresh = crash_prob_thresh
        self.consecutive_req = consecutive_req
        self.consecutive_crash_count = 0
        self.deployed = False
        self.max_blackbox_size = max_blackbox_size
        # 1-minute rolling buffer of raw acceleration magnitudes
        self.blackbox_accel = deque(maxlen=max_blackbox_size)

    def update_blackbox(self, sample):
        """Append raw sample acceleration magnitude to blackbox."""
        ax = sample.get("ax", 0.0)
        ay = sample.get("ay", 0.0)
        az = sample.get("az", 0.0)
        mag = math.sqrt(ax**2 + ay**2 + az**2)
        # Convert to g if in m/s^2
        if mag > 50.0:
            mag /= 9.81
        self.blackbox_accel.append(mag)

    def _is_false_positive(self, current_mag):
        """
        Compare current magnitude with the 1-minute baseline trend.
        If the baseline is extremely noisy (e.g. rough off-road terrain)
        and the current spike is not a statistical outlier, flag as false positive.
        """
        if len(self.blackbox_accel) < 50:
            # Not enough data for baseline comparison yet
            return False

        # Calculate baseline statistics from the 1-minute blackbox history
        vals = list(self.blackbox_accel)
        mean_val = sum(vals) / len(vals)
        variance = sum((x - mean_val) ** 2 for x in vals) / len(vals)
        std_val = math.sqrt(variance)

        # If baseline is highly volatile (std > 4.0g), check if the current spike is an outlier.
        # If it doesn't exceed mean + 3 * std, cancel deployment to avoid false positive.
        if std_val > 4.0:
            outlier_thresh = mean_val + 3.0 * std_val
            if current_mag < outlier_thresh:
                log.warning(
                    f"[ARBITER] FALSE POSITIVE CANCELLED: Peak {current_mag:.1f}g is not a statistical outlier "
                    f"under rough road baseline (mean={mean_val:.1f}g, std={std_val:.1f}g, thresh={outlier_thresh:.1f}g)."
                )
                return True
        return False

    def evaluate(self, physics_result, ml_result=None, current_sample=None):
        """
        physics_result: (is_physics_crash: bool, physics_reason: str, is_extreme: bool)
        ml_result: (pred_label: int, crash_prob: float, nc_prob: float) or None if window incomplete
        current_sample: raw sample dict for baseline comparison
        Returns: (should_deploy: bool, decision_reason: str)
        """
        if self.deployed:
            return False, "ALREADY_DEPLOYED"

        if current_sample:
            self.update_blackbox(current_sample)

        phys_crash, phys_reason, is_extreme = physics_result

        # Determine current acceleration magnitude
        current_mag = 1.0
        if current_sample:
            ax = current_sample.get("ax", 0.0)
            ay = current_sample.get("ay", 0.0)
            az = current_sample.get("az", 0.0)
            current_mag = math.sqrt(ax**2 + ay**2 + az**2)
            if current_mag > 50.0:
                current_mag /= 9.81

        # Check for False Positive Cancellation on candidate deployments
        if is_extreme or phys_crash or (ml_result and ml_result[1] >= self.crash_prob_thresh):
            if self._is_false_positive(current_mag):
                self.consecutive_crash_count = 0
                return False, "FALSE_POSITIVE_CANCELLED"

        # Condition 1: Extreme Physics Override (Bypasses ML completely)
        if is_extreme:
            self.deployed = True
            log.critical(f"[ARBITER] DEPLOY AIRBAG -> {phys_reason}")
            return True, phys_reason

        # Condition 2: ML Analysis is Ready -> ML is Superior
        if ml_result is not None:
            pred_label, crash_prob, nc_prob = ml_result
            if crash_prob >= self.crash_prob_thresh:
                self.consecutive_crash_count += 1
                if self.consecutive_crash_count >= self.consecutive_req:
                    self.deployed = True
                    reason = f"ML_SUPERIOR_CRASH (prob={crash_prob:.1%}, consecutive={self.consecutive_crash_count})"
                    log.critical(f"[ARBITER] DEPLOY AIRBAG -> {reason}")
                    return True, reason
                else:
                    reason = f"ML_CRASH_GATE ({self.consecutive_crash_count}/{self.consecutive_req})"
                    return False, reason
            else:
                self.consecutive_crash_count = 0
                return False, f"ML_SUPERIOR_NORMAL (label={pred_label}, p={crash_prob:.1%})"

        # Condition 3: ML is STILL buffering/analyzing -> Physics Logic check takes over in tight condition
        if phys_crash:
            self.deployed = True
            reason = f"TIGHT_CONDITION_PHYSICS_OVERRIDE ({phys_reason})"
            log.critical(f"[ARBITER] DEPLOY AIRBAG -> {reason}")
            return True, reason

        return False, "MONITORING"

