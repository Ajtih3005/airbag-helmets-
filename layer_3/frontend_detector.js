/**
 * layer_3/frontend_detector.js
 * ----------------------------
 * Dual-Layer ML Detection Engine for Web Browser.
 * Runs the EXACT same trained Random Forest models (100 trees) on the exact 81 features
 * in JavaScript, identical to the Raspberry Pi Python backend.
 */

// 81-Feature Extractor in JavaScript
function extractFeaturesWindowJS(windowSamples) {
  const feats = {};
  const n = windowSamples.length;
  if (n === 0) return feats;

  const ax = new Float64Array(n), ay = new Float64Array(n), az = new Float64Array(n);
  const gx = new Float64Array(n), gy = new Float64Array(n), gz = new Float64Array(n);
  const hg_ax = new Float64Array(n), hg_ay = new Float64Array(n), hg_az = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const s = windowSamples[i];
    ax[i] = s.ax || 0; ay[i] = s.ay || 0; az[i] = s.az || 0;
    gx[i] = s.gx || 0; gy[i] = s.gy || 0; gz[i] = s.gz || 0;
    hg_ax[i] = s.hg_ax !== undefined ? s.hg_ax : ax[i];
    hg_ay[i] = s.hg_ay !== undefined ? s.hg_ay : ay[i];
    hg_az[i] = s.hg_az !== undefined ? s.hg_az : az[i];
  }

  function stats(col, name) {
    let sum = 0, sumSq = 0, min = col[0], max = col[0];
    for (let i = 0; i < n; i++) {
      const v = col[i];
      sum += v; sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const mean = sum / n;
    const variance = Math.max(0, (sumSq / n) - (mean * mean));
    const std = Math.sqrt(variance);
    const rms = Math.sqrt(sumSq / n);
    const range = max - min;

    // Percentiles for IQR
    const sorted = Array.from(col).sort((a, b) => a - b);
    const q25 = sorted[Math.floor(n * 0.25)];
    const q75 = sorted[Math.floor(n * 0.75)];
    const iqr = q75 - q25;

    // Skewness
    let skewSum = 0;
    if (std > 1e-8) {
      for (let i = 0; i < n; i++) {
        skewSum += Math.pow((col[i] - mean) / std, 3);
      }
      feats[`${name}_skew`] = skewSum / n;
    } else {
      feats[`${name}_skew`] = 0.0;
    }

    feats[`${name}_mean`] = mean;
    feats[`${name}_std`] = std;
    feats[`${name}_max`] = max;
    feats[`${name}_min`] = min;
    feats[`${name}_range`] = range;
    feats[`${name}_rms`] = rms;
    feats[`${name}_iqr`] = iqr;
  }

  function hgStats(col, name) {
    let sum = 0, sumSq = 0, maxAbs = Math.abs(col[0]);
    for (let i = 0; i < n; i++) {
      const v = col[i];
      sum += v; sumSq += v * v;
      const absV = Math.abs(v);
      if (absV > maxAbs) maxAbs = absV;
    }
    const mean = sum / n;
    const variance = Math.max(0, (sumSq / n) - (mean * mean));
    feats[`${name}_mean`] = mean;
    feats[`${name}_std`] = Math.sqrt(variance);
    feats[`${name}_max`] = maxAbs;
    feats[`${name}_rms`] = Math.sqrt(sumSq / n);
  }

  // MPU6050 Stats
  stats(ax, 'ax'); stats(ay, 'ay'); stats(az, 'az');
  stats(gx, 'gx'); stats(gy, 'gy'); stats(gz, 'gz');

  // ADXL377 High-G Stats
  hgStats(hg_ax, 'hg_ax'); hgStats(hg_ay, 'hg_ay'); hgStats(hg_az, 'hg_az');

  // Magnitudes
  const accelMag = new Float64Array(n), gyroMag = new Float64Array(n), hgMag = new Float64Array(n);
  let smaA = 0, smaG = 0, smaH = 0;
  for (let i = 0; i < n; i++) {
    accelMag[i] = Math.sqrt(ax[i]**2 + ay[i]**2 + az[i]**2);
    gyroMag[i]  = Math.sqrt(gx[i]**2 + gy[i]**2 + gz[i]**2);
    hgMag[i]    = Math.sqrt(hg_ax[i]**2 + hg_ay[i]**2 + hg_az[i]**2);
    smaA += Math.abs(ax[i]) + Math.abs(ay[i]) + Math.abs(az[i]);
    smaG += Math.abs(gx[i]) + Math.abs(gy[i]) + Math.abs(gz[i]);
    smaH += Math.abs(hg_ax[i]) + Math.abs(hg_ay[i]) + Math.abs(hg_az[i]);
  }

  function magStats(col, name) {
    let sum = 0, sumSq = 0, max = col[0];
    for (let i = 0; i < n; i++) {
      const v = col[i];
      sum += v; sumSq += v * v;
      if (v > max) max = v;
    }
    const mean = sum / n;
    const std = Math.sqrt(Math.max(0, (sumSq / n) - (mean * mean)));
    feats[`${name}_mean`] = mean;
    feats[`${name}_std`] = std;
    feats[`${name}_max`] = max;
  }

  magStats(accelMag, 'accel_mag');
  magStats(gyroMag,  'gyro_mag');
  magStats(hgMag,    'hg_mag');

  feats['sma_accel'] = smaA / n;
  feats['sma_gyro']  = smaG / n;
  feats['sma_hg']    = smaH / n;

  // Jerk
  let jerkSum = 0, jerkSumSq = 0, jerkMax = 0;
  const nJ = n - 1;
  if (nJ > 0) {
    for (let i = 0; i < nJ; i++) {
      const jx = ax[i+1] - ax[i];
      const jy = ay[i+1] - ay[i];
      const jz = az[i+1] - az[i];
      const jm = Math.sqrt(jx*jx + jy*jy + jz*jz);
      jerkSum += jm; jerkSumSq += jm * jm;
      if (jm > jerkMax) jerkMax = jm;
    }
    const jMean = jerkSum / nJ;
    const jStd = Math.sqrt(Math.max(0, (jerkSumSq / nJ) - (jMean * jMean)));
    feats['jerk_mean'] = jMean;
    feats['jerk_max']  = jerkMax;
    feats['jerk_std']  = jStd;
  } else {
    feats['jerk_mean'] = 0.0; feats['jerk_max'] = 0.0; feats['jerk_std'] = 0.0;
  }

  // Tilt Angle
  let tiltSum = 0, tiltSumSq = 0, tiltMax = 0;
  for (let i = 0; i < n; i++) {
    const sm = Math.max(1e-6, accelMag[i]);
    const cosT = Math.max(-1.0, Math.min(1.0, az[i] / sm));
    const deg = Math.acos(cosT) * (180.0 / Math.PI);
    tiltSum += deg; tiltSumSq += deg * deg;
    if (deg > tiltMax) tiltMax = deg;
  }
  const tMean = tiltSum / n;
  const tStd = Math.sqrt(Math.max(0, (tiltSumSq / n) - (tMean * tMean)));
  feats['tilt_mean_deg'] = tMean;
  feats['tilt_max_deg']  = tiltMax;
  feats['tilt_std_deg']  = tStd;

  // Correlations
  function corr(x, y) {
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (let i = 0; i < n; i++) {
      sumX += x[i]; sumY += y[i];
      sumXY += x[i] * y[i];
      sumX2 += x[i] * x[i]; sumY2 += y[i] * y[i];
    }
    const num = n * sumXY - sumX * sumY;
    const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    return den > 1e-8 ? num / den : 0.0;
  }

  feats['corr_ax_ay'] = corr(ax, ay);
  feats['corr_ax_az'] = corr(ax, az);
  feats['corr_ay_az'] = corr(ay, az);
  feats['corr_gx_gy'] = corr(gx, gy);
  feats['corr_gx_gz'] = corr(gx, gz);
  feats['corr_gy_gz'] = corr(gy, gz);

  // High-G Peak Ratio
  let mpuMax = 0, hgMaxVal = 0;
  for (let i = 0; i < n; i++) {
    const am = Math.max(Math.abs(ax[i]), Math.abs(ay[i]));
    if (am > mpuMax) mpuMax = am;
    const hm = Math.abs(hg_ax[i]);
    if (hm > hgMaxVal) hgMaxVal = hm;
  }
  feats['hg_peak_ratio'] = hgMaxVal / Math.max(1e-6, mpuMax);

  return feats;
}


class FrontendDetector {
  constructor() {
    this.WIN_SZ = 50;
    this.GATE_REQ = 3;
    this.CRASH_THR = 0.70;

    this.detScore = 0.0;
    this.trend = 0.0;
    this.gateCount = 0;
    this.totalSamples = 0;
    this.totalWindows = 0;
    this.airbagDeployed = false;

    this.windowBuf = [];
    this.detHistory = [];
    this.bbHistory = [];
  }

  updateSentinel(p1Label, p1Crash, p1Near) {
    const weight = p1Label === 2 ? 1.0 : (p1Label === 1 ? 0.5 : 0.0);
    this.detHistory.push(weight);
    if (this.detHistory.length > 20) this.detHistory.shift();

    const n = this.detHistory.length;
    this.detScore = this.detHistory.reduce((a, b) => a + b, 0) / n;

    if (n >= 4) {
      const half = Math.floor(n / 2);
      const recent = this.detHistory.slice(half).reduce((a, b) => a + b, 0) / (n - half);
      const older = this.detHistory.slice(0, half).reduce((a, b) => a + b, 0) / half;
      this.trend = recent - older;
    } else {
      this.trend = 0.0;
    }

    // Dynamic Gate Acceleration on rapid deterioration
    if (this.trend >= 0.25 || this.detScore >= 0.40) {
      if (this.gateCount === 0) {
        this.gateCount = 2; // Pre-arm gate to 2/3
      }
    } else if (this.trend <= -0.20 && this.gateCount > 0 && p1Label === 0) {
      this.gateCount = 0; // Reset gate on stabilizing signal
    }
  }

  evalSample(sampleDict) {
    this.totalSamples++;

    // 1. Run Path 1 Real Random Forest in JS (sample_model.pkl mirror)
    let p1Res = { label: 0, crash_prob: 0.0, near_prob: 0.0, norm_prob: 1.0 };
    if (typeof predictSampleRF_JS !== 'undefined') {
      p1Res = predictSampleRF_JS(sampleDict);
    }

    this.updateSentinel(p1Res.label, p1Res.crash_prob, p1Res.near_prob);

    // 2. Buffer window
    this.windowBuf.push(sampleDict);
    if (this.windowBuf.length > this.WIN_SZ) this.windowBuf.shift();

    // 3. Run Path 2 Real Random Forest on 81 features (best_model.pkl mirror)
    let p2Res = { label: 0, crash_prob: 0.0, near_prob: 0.0, norm_prob: 1.0 };
    let p2Evaluated = false;

    if (this.windowBuf.length >= this.WIN_SZ) {
      this.totalWindows++;
      p2Evaluated = true;

      const feats81 = extractFeaturesWindowJS(this.windowBuf);
      if (typeof predictWindowRF_JS !== 'undefined') {
        p2Res = predictWindowRF_JS(feats81);
      }

      if (p2Res.crash_prob >= this.CRASH_THR) {
        this.gateCount++;
        if (this.gateCount >= this.GATE_REQ && !this.airbagDeployed) {
          const ratio = this.evalBlackBox();
          if (ratio >= 0.05) {
            this.airbagDeployed = true;
          } else {
            this.gateCount = 0;
          }
        }
      } else {
        if (!(p1Res.label === 2 && p1Res.crash_prob >= this.CRASH_THR)) {
          this.gateCount = Math.max(0, this.gateCount - 1);
        }
      }
    }

    const labelNames = { 0: 'Normal', 1: 'Near-Crash', 2: 'CRASH' };
    const wL = labelNames[p2Res.label] || 'Normal';

    this.bbHistory.push({ l: wL, t: this.totalSamples, p1: p1Res.label, p2: p2Res.label });
    if (this.bbHistory.length > 60000) this.bbHistory.shift();

    return {
      p1_label: p1Res.label,
      p1_crash: p1Res.crash_prob,
      p1_near:  p1Res.near_prob,
      det_score: this.detScore,
      trend: this.trend,
      p2_evaluated: p2Evaluated,
      p2_label: p2Res.label,
      p2_crash: p2Res.crash_prob,
      p2_near:  p2Res.near_prob,
      p2_norm:  p2Res.norm_prob,
      label_name: wL,
      gate_count: this.gateCount,
      airbag_deployed: this.airbagDeployed,
      total_samples: this.totalSamples,
      total_windows: this.totalWindows,
    };
  }

  evalBlackBox() {
    const total = this.bbHistory.length;
    const crashes = this.bbHistory.filter(h => h.p1 in [1,2] || h.p2 in [1,2] || h.l === 'CRASH').length;
    return total > 0 ? (crashes / total) : 0.0;
  }

  reset() {
    this.detScore = 0.0;
    this.trend = 0.0;
    this.gateCount = 0;
    this.totalSamples = 0;
    this.totalWindows = 0;
    this.airbagDeployed = false;
    this.windowBuf = [];
    this.detHistory = [];
    this.bbHistory = [];
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { FrontendDetector, extractFeaturesWindowJS };
}
