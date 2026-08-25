# Layer 3 — Machine Learning Pipeline & Dual-Layer Arbiter

## Overview
Layer 3 is responsible for:
1. **Feature Extraction:** Computing 81 sliding-window statistical features from the 1000Hz sensor telemetry.
2. **3-Path ML Classification:**
   - **Path 1 (Per-Sample Sentinel):** Instant 1ms inference + Deterioration Analyzer (`DetScore` & `Trend`).
   - **Path 2 (Window ML Classifier):** High-precision 50ms window Random Forest classification with a 3-window gate counter.
   - **Path 3 (Black Box RAM Validation):** 60-second rolling prediction history check to suppress isolated false-positive drops.
3. **Dual-Layer Arbiter:** Bridging ML predictions with the fast physical threshold trigger and controlling actuator/solenoid deployment signals.
