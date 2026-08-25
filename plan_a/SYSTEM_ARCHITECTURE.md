# ML-Driven Smart Helmet — System Architecture

---

## Overview

A 3-layer Machine Learning pipeline that reads raw IMU sensor data at 1000 Hz,
runs continuous ML inference, and deploys an airbag within 3ms of crash detection.

**Design Philosophy: "ALL ARE ML PREDICTION ONLY — PHYSICAL CANNOT THINK."**

---

## Hardware Layer (Input)

```
MPU6050 (Low-g Sensor)              ADXL377 (High-g Sensor)
  via I2C Bus                         via MCP3008 ADC on SPI CE0
  ├── ax  (accel X)  m/s²             ├── hg_ax  (±200g accel X)
  ├── ay  (accel Y)  m/s²             ├── hg_ay  (±200g accel Y)
  ├── az  (accel Z)  m/s²             └── hg_az  (±200g accel Z)
  ├── gx  (gyro  X)  deg/s
  ├── gy  (gyro  Y)  deg/s
  └── gz  (gyro  Z)  deg/s

Combined: 9-channel raw sensor vector @ 1000 Hz (every 1ms)
```

**Fallback Chain (auto-negotiated):**
1. Direct I2C (MPU6050 + ADXL377 SPI)
2. Serial UART (ESP32 streaming CSV @ 115200 baud)
3. Synthetic simulation (random noise, physics-based ride model)

---

## PATH 1 — Per-Sample ML (Sentinel)

```
File      : models/sample_model.pkl
Type      : sklearn Pipeline
  Step 1  : StandardScaler()           ← Normalizes 9 raw channels
  Step 2  : RandomForestClassifier(
               n_estimators = 100,     ← 100 decision trees
               max_depth    = 10,      ← Shallow trees = fast inference
               min_samples_leaf = 5,
               class_weight = 'balanced',
               random_state = 42
            )

Input     : 9 raw sensor values (ax, ay, az, gx, gy, gz, hg_ax, hg_ay, hg_az)
Output    : Label (Normal / Near-Crash / Crash) + Crash probability %
Latency   : < 0.1ms per sample
Accuracy  : 98.39%   Macro-F1: 95.13%
Runs      : Every 1ms (every single raw sample)
```

**Deterioration Sentinel (built on top of Path 1):**
```
Tracks last 20ms of Path 1 predictions.
Assigns weights: Normal=0.0, Near-Crash=0.5, Crash=1.0
Computes:
  - DetScore  : Rolling average weight (0.00 to 1.00)
  - Trend     : Slope of DetScore over 20ms window

Actions:
  - Trend >= +0.25  → PRE-ARM: Boost Gate counter to 2/3 instantly
                       (Path 2 only needs 1 more window to deploy)
  - Trend <= -0.20  → CANCEL: Reset Gate to 0
                       (Signal stabilizing, suppress false alarm)
```

---

## PATH 2 — Window ML (High-Precision Confirmation)

```
File      : models/best_model.pkl
Type      : sklearn Pipeline
  Step 1  : StandardScaler()           ← Normalizes 81 features
  Step 2  : RandomForestClassifier(
               n_estimators = 100,     ← 100 decision trees
               max_depth    = 15,      ← Deeper trees = more precision
               class_weight = 'balanced',
               random_state = 42
            )

Input     : 81 statistical features extracted from last 50 raw samples (50ms window)
            Features include:
              mean, std, min, max, range          (per channel)
              jerk (rate of change of accel)
              signal magnitude area (SMA)
              tilt angle estimation
              high-g ratio (samples > 3g)
              inter-axis correlations (ax-ay, ax-az, ay-az)

Output    : Label (Normal / Near-Crash / Crash) + Crash probability %
Latency   : ~1ms per window evaluation
Accuracy  : 99.17%   Macro-F1: 97.48%
Runs      : Every 1ms (sliding window, 50ms context, 1ms stride)
```

**Gate Counter (Path 2 Deployment Logic):**
```
Requires 3 consecutive Path 2 windows predicting Crash (p >= 70%)
to proceed to Path 3 validation.

Gate resets to 0 if Path 2 predicts Normal.
Gate jumps to 2/3 if Sentinel pre-arms it.
```

---

## PATH 3 — Black Box Validation (60-Second History Check)

```
Source    : RAM buffer (last 60,000 samples = 60 seconds at 1000 Hz)
Stores    : Every sample's P1 label, P2 label, timestamp, raw sensor values

When Gate reaches 3/3:
  1. Scan all predictions in last 60s from RAM buffer
  2. Count entries where P1=Crash/Near-Crash OR P2=Crash/Near-Crash
  3. Compute ratio = crash_entries / total_entries
  4. If ratio >= 5%  → CONFIRMED CRASH → DEPLOY
     If ratio <  5%  → FALSE POSITIVE (helmet drop) → CANCEL

On Deployment:
  - GPIO Pin 17 → HIGH (Raspberry Pi)
  - Solenoid valve opens → CO2 canister fires → Airbag inflates in ~15ms
  - SIM800L GSM module sends emergency SMS
  - Full 60s blackbox buffer dumped to CSV: logs/blackbox_crash_<timestamp>.csv
```

---

## Deployment Decision Flowchart

```
Every 1ms:
  RAW SAMPLE (9 channels)
         │
    ┌────┼────┐
    ▼    ▼    ▼
  PATH1  PATH2  BLACK BOX
  P.Sample  Window  RAM (60s)
  RF Model  RF Model
  9 inputs  81 feats
  <0.1ms    ~1ms
    │        │
    ▼        ▼
 SENTINEL  GATE (0/1/2/3)
 Det+Trend
    └────┬────┘
         ▼
    Gate reaches 3?
         │
         YES → PATH 3 VALIDATE
               ratio >= 5%?
               │
               YES → AIRBAG DEPLOY
                     GPIO17 → CO2 → Inflate
                     SMS Alert + CSV Dump
               NO  → False Positive. Reset.
```

---

## Timing Breakdown

| Stage                           | Time      |
|---------------------------------|-----------|
| Sensor read (hardware I2C)      | 0.5 ms    |
| Path 1 inference (RF, 9 inputs) | < 0.1 ms  |
| Path 2 inference (RF, 81 feats) | ~ 1.0 ms  |
| Path 3 blackbox scan (60s RAM)  | ~ 1.0 ms  |
| GPIO signal + solenoid open     | ~ 1.0 ms  |
| CO2 airbag full inflation       | ~15.0 ms  |
| **TOTAL (detection → inflate)** | **~18 ms**|

---

## File Map

```
a123/
├── models/
│   ├── sample_model.pkl          ← PATH 1: RF per-sample pipeline (6.6 MB)
│   ├── sample_model_meta.pkl     ← PATH 1: accuracy=98.39%, feature names
│   ├── best_model.pkl            ← PATH 2: RF window pipeline (1.6 MB)
│   └── model_meta.pkl            ← PATH 2: accuracy=99.17%, feature names
│
├── src/
│   ├── data_generator.py         ← Synthetic IMU ride data (seed=None)
│   ├── feature_engineering.py    ← 81-feature extractor for Path 2
│   ├── train_sample_model.py     ← Trains Path 1 model
│   ├── train_model.py            ← Trains Path 2 model
│   └── raspberry_pi_interface.py ← Live inference loop + DeteriorationAnalyzer
│
├── 01_hardware_sensors/
│   ├── sensor_reader.py          ← Unified sensor interface (I2C/Serial/Sim)
│   ├── mpu6050_i2c.py            ← MPU6050 I2C driver
│   └── adxl377_reader.py         ← ADXL377 via MCP3008 SPI ADC
│
├── combined_demo.py              ← Master demo: all 3 paths + Sentinel
├── 06_simulation_demo/
│   └── simulate_full_flow.py     ← Old 2-layer demo (reference only)
│
└── logs/
    └── blackbox_crash_*.csv      ← Auto-saved on every deployment
```

---

## Model Training Summary

| | Path 1 (Per-Sample) | Path 2 (Window) |
|---|---|---|
| **Model file** | `sample_model.pkl` | `best_model.pkl` |
| **Algorithm** | Random Forest | Random Forest |
| **Pipeline** | StandardScaler + RF | StandardScaler + RF |
| **Input features** | 9 raw sensor channels | 81 statistical features |
| **Tree depth** | max_depth=10 (fast) | max_depth=15 (precise) |
| **Training data** | 600,000 raw samples | ~580,000 windows |
| **Classes** | Normal / Near-Crash / Crash | Normal / Near-Crash / Crash |
| **Accuracy** | 98.39% | 99.17% |
| **Macro F1** | 95.13% | 97.48% |
| **Inference time** | < 0.1 ms | ~1 ms |
| **Role** | Instant sentinel + gate pre-arming | High-precision gate confirmation |

---

*Generated: 2026-08-05 | Smart Airbag Helmet Project*
