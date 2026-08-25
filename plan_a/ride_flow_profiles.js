// ================================================================
// RIDE FLOW PROFILES — Generates coordinates/states over time
// This file is NOT known to the ML engine.
// The ML engine only analyzes raw sensor telemetry derived from these movements.
// ================================================================

const RIDE_FLOWS = {
  // --- NORMAL SCENARIOS ---
  highway: (t) => {
    return {
      speed: 110,
      x: 0,
      y: Math.sin(t * 12) * 0.015, // highway micro-bounce
      z: 0,
      lean: Math.sin(t * 0.5) * 0.02, // slow cruising sway
      pitch: 0,
      vibration: 0.04,
      stage: 0
    };
  },
  city: (t) => {
    // Stop-and-go speed profile
    const speedCycle = 25 + Math.sin(t * 0.4) * 15;
    return {
      speed: speedCycle,
      x: Math.sin(t * 0.2) * 0.4, // gentle city steering
      y: Math.sin(t * 8) * 0.02,
      z: 0,
      lean: Math.sin(t * 0.8) * 0.04,
      pitch: Math.cos(t * 0.4) * 0.02, // acceleration/deceleration pitching
      vibration: 0.08,
      stage: 0
    };
  },
  corner_l: (t) => {
    // Lean left turn
    const targetLean = -0.42;
    const progress = Math.min(1, t / 1.5); // 1.5s to roll in
    const activeLean = targetLean * progress;
    return {
      speed: 65,
      x: -0.8 * progress,
      y: 0,
      z: 0,
      lean: activeLean,
      pitch: 0,
      vibration: 0.06,
      stage: 0
    };
  },
  corner_r: (t) => {
    // Lean right turn
    const targetLean = 0.42;
    const progress = Math.min(1, t / 1.5);
    const activeLean = targetLean * progress;
    return {
      speed: 65,
      x: 0.8 * progress,
      y: 0,
      z: 0,
      lean: activeLean,
      pitch: 0,
      vibration: 0.06,
      stage: 0
    };
  },
  accel: (t) => {
    // Hard acceleration
    const speed = Math.min(125, 30 + t * 45); // rapid speed ramp
    const pitch = -0.12 * Math.max(0, 1 - t / 3.0); // pitch back, then settle
    return {
      speed: speed,
      x: 0,
      y: 0.02,
      z: 0,
      lean: 0,
      pitch: pitch,
      vibration: 0.10,
      stage: 0
    };
  },
  brk_soft: (t) => {
    // Gentle braking
    const speed = Math.max(20, 70 - t * 15);
    const pitch = 0.07 * Math.max(0, 1 - t / 2.5); // nose down
    return {
      speed: speed,
      x: 0,
      y: -0.01,
      z: 0,
      lean: 0,
      pitch: pitch,
      vibration: 0.05,
      stage: 0
    };
  },
  rain: (t) => {
    // Wet road cruising
    return {
      speed: 55,
      x: 0,
      y: Math.sin(t * 15) * 0.02,
      z: 0,
      lean: Math.sin(t * 0.3) * 0.015,
      pitch: 0,
      vibration: 0.16, // higher vibration on wet asphalt
      stage: 0
    };
  },
  night: (t) => {
    // Night cruise
    return {
      speed: 100,
      x: 0,
      y: Math.sin(t * 10) * 0.012,
      z: 0,
      lean: Math.sin(t * 0.4) * 0.02,
      pitch: 0,
      vibration: 0.03,
      stage: 0
    };
  },

  // --- NEAR-CRASH HAZARDS ---
  pothole: (t) => {
    let dy = 0;
    let pitch = 0;
    // Pothole strike occurs at t = 1.0s
    if (t >= 1.0 && t < 1.3) {
      const et = t - 1.0;
      if (et < 0.08) {
        dy = -0.22 * (et / 0.08); // sharp dip down
        pitch = 0.14 * (et / 0.08);
      } else {
        const rebound = (et - 0.08) / 0.22;
        dy = -0.22 + 0.26 * rebound; // rapid upward spring back
        pitch = 0.14 - 0.20 * rebound;
      }
    }
    return {
      speed: 68,
      x: 0,
      y: dy,
      z: 0,
      lean: 0,
      pitch: pitch,
      vibration: t >= 1.0 && t < 1.6 ? 0.38 : 0.06,
      stage: 1
    };
  },
  bump: (t) => {
    let dy = 0;
    let pitch = 0;
    // Speed bump strike at t = 1.0s
    if (t >= 1.0 && t < 1.4) {
      const et = t - 1.0;
      if (et < 0.15) {
        dy = 0.18 * (et / 0.15); // bounce up
        pitch = -0.09 * (et / 0.15);
      } else {
        const drop = (et - 0.15) / 0.25;
        dy = 0.18 - 0.20 * drop; // settle down
        pitch = -0.09 + 0.12 * drop;
      }
    }
    return {
      speed: 52,
      x: 0,
      y: dy,
      z: 0,
      lean: 0,
      pitch: pitch,
      vibration: t >= 1.0 && t < 1.8 ? 0.28 : 0.07,
      stage: 1
    };
  },
  emr_brk: (t) => {
    // ABS Emergency stop
    const speed = Math.max(0, 80 - (t * 50)); // rapid stop
    const pitch = speed > 0 ? 0.18 + Math.sin(t * 20) * 0.03 : 0; // intense nose dive
    return {
      speed: speed,
      x: 0,
      y: -0.03,
      z: 0,
      lean: 0,
      pitch: pitch,
      vibration: speed > 0 ? 0.32 : 0, // ABS pump vibration
      stage: 1
    };
  },
  swerve: (t) => {
    // S-curve dodge swerve
    let x = 0;
    let lean = 0;
    if (t >= 0.5 && t < 2.0) {
      const et = t - 0.5;
      x = Math.sin(et * Math.PI * 1.5) * 1.2; // violent lane shift
      lean = -Math.cos(et * Math.PI * 1.5) * 0.46; // corresponding deep banking
    }
    return {
      speed: 75,
      x: x,
      y: 0,
      z: 0,
      lean: lean,
      pitch: 0,
      vibration: 0.18,
      stage: 1
    };
  },
  gravel: (t) => {
    // Slipping on loose gravel
    let lean = Math.sin(t * 0.5) * 0.02;
    if (t >= 1.0 && t < 2.8) {
      lean = Math.sin(t * 18) * 0.14; // rapid slip/recovery oscillations
    }
    return {
      speed: 58,
      x: Math.sin(t * 0.5) * 0.1,
      y: 0,
      z: 0,
      lean: lean,
      pitch: 0,
      vibration: t >= 1.0 && t < 2.8 ? 0.44 : 0.06,
      stage: 1
    };
  },
  tankslap: (t) => {
    // High-speed steering wobbles (90km/h)
    let lean = 0;
    let x = 0;
    if (t >= 0.8 && t < 2.5) {
      const et = t - 0.8;
      // Handlebars oscillate violently, shaking the bike left/right
      lean = Math.sin(et * 25) * 0.35 * Math.min(1, et * 2);
      x = Math.sin(et * 25) * 0.25;
    }
    return {
      speed: 90,
      x: x,
      y: 0,
      z: 0,
      lean: lean,
      pitch: 0,
      vibration: t >= 0.8 ? 0.42 : 0.08,
      stage: 1
    };
  },
  nearlow: (t) => {
    // Rear wheel slides, almost low-sides, then recovers grip
    let lean = 0;
    let x = 0;
    if (t >= 0.8 && t < 2.2) {
      const et = t - 0.8;
      if (et < 0.6) {
        lean = -0.58 * (et / 0.6); // sudden sliding lean angle
        x = -0.8 * (et / 0.6);
      } else {
        const rec = (et - 0.6) / 0.8;
        lean = -0.58 + 0.58 * rec; // violent snap back to vertical
        x = -0.8 + 0.8 * rec;
      }
    }
    return {
      speed: 66,
      x: x,
      y: 0,
      z: 0,
      lean: lean,
      pitch: 0,
      vibration: t >= 0.8 && t < 1.4 ? 0.38 : 0.08,
      stage: 1
    };
  },

  // --- CATASTROPHIC CRASH SCENARIOS (Stage 2 triggered) ---
  front: (t) => {
    // Head-on crash: instant wall deceleration at t = 1.0s
    let speed = 80;
    let x = 0, y = 0, z = 0;
    let pitch = 0, lean = 0;
    let stage = 0;

    if (t >= 1.0) {
      stage = 2; // crash onset
      const et = t - 1.0;
      speed = Math.max(0, 80 - et * 4000); // stops in 20ms
      pitch = Math.min(1.4, et * 4.5); // nose flips down/up
      y = Math.min(2.5, et * 5.0); // bike collapses/flies up
      z = -et * 2.0;
    }
    return { speed, x, y, z, lean, pitch, vibration: stage === 2 ? 1.5 : 0.07, stage };
  },
  rear: (t) => {
    // Hit from behind at t = 1.0s
    let speed = 55;
    let x = 0, y = 0, z = 0;
    let pitch = 0, lean = 0;
    let stage = 0;

    if (t >= 1.0) {
      stage = 2;
      const et = t - 1.0;
      speed = speed + Math.max(0, 35 - et * 120); // sudden push forward
      pitch = Math.max(-0.6, -et * 2.8); // wheelie lurch backwards
      y = Math.min(1.2, et * 2.5);
    }
    return { speed, x, y, z, lean, pitch, vibration: stage === 2 ? 1.3 : 0.08, stage };
  },
  highside: (t) => {
    // High-side crash at t = 1.0s (bike snaps, throws rider)
    let speed = 72;
    let x = 0, y = 0, z = 0;
    let pitch = 0, lean = 0;
    let stage = 0;

    if (t >= 1.0) {
      stage = 2;
      const et = t - 1.0;
      speed = Math.max(0, 72 - et * 60);
      lean = Math.min(2.2, et * 5.8); // bike pitches over violently
      y = Math.min(3.2, et * 4.0); // flying bike trajectory
      pitch = et * 0.8;
    }
    return { speed, x, y, z, lean, pitch, vibration: stage === 2 ? 1.6 : 0.08, stage };
  },
  lowside: (t) => {
    // Slides out flat onto the asphalt at t = 1.0s
    let speed = 65;
    let x = 0, y = 0, z = 0;
    let pitch = 0, lean = 0;
    let stage = 0;

    if (t >= 1.0) {
      stage = 2;
      const et = t - 1.0;
      speed = Math.max(0, 65 - et * 35);
      lean = Math.max(-1.5, -et * 5.2); // slides flat on side
      y = -0.04;
      x = -et * 3.5; // sliding off to the left
    }
    return { speed, x, y, z, lean, pitch, vibration: stage === 2 ? 0.9 : 0.08, stage };
  },
  swipe: (t) => {
    // Side swipe from the right at t = 1.0s
    let speed = 60;
    let x = 0, y = 0, z = 0;
    let pitch = 0, lean = 0;
    let stage = 0;

    if (t >= 1.0) {
      stage = 2;
      const et = t - 1.0;
      speed = Math.max(0, 60 - et * 40);
      x = -et * 5.0; // swiped laterally to the left
      lean = Math.max(-1.3, -et * 3.8); // bike rolls left
    }
    return { speed, x, y, z, lean, pitch, vibration: stage === 2 ? 1.1 : 0.07, stage };
  },
  hw_crash: (t) => {
    // High-speed motorway impact (110 km/h) at t = 1.0s
    let speed = 110;
    let x = 0, y = 0, z = 0;
    let pitch = 0, lean = 0;
    let stage = 0;

    if (t >= 1.0) {
      stage = 2;
      const et = t - 1.0;
      speed = Math.max(0, 110 - et * 5000); // violent stop
      y = Math.min(4.5, et * 8.5); // vertical launching
      pitch = et * 5.5; // wild tumbling
      lean = et * 4.2;
    }
    return { speed, x, y, z, lean, pitch, vibration: stage === 2 ? 2.5 : 0.06, stage };
  },
  offroad: (t) => {
    // launched off-road at t = 1.0s
    let speed = 50;
    let x = 0, y = 0, z = 0;
    let pitch = 0, lean = 0;
    let stage = 0;

    if (t >= 1.0) {
      stage = 2;
      const et = t - 1.0;
      speed = Math.max(0, 50 - et * 45);
      y = Math.min(3.8, et * 5.2);
      pitch = et * 4.0; // forward tumble
    }
    return { speed, x, y, z, lean, pitch, vibration: stage === 2 ? 1.4 : 0.09, stage };
  }
};
