import sys, os, time
sys.path.insert(0, '.')
from importlib import import_module

print("=" * 50)
print("  SMART HELMET — FULL SYSTEM HEALTH CHECK")
print("=" * 50)

# Layer 1
try:
    sr = import_module("01_hardware_sensors.sensor_reader")
    sensors = sr.SensorReader(simulate=True)
    s = sensors.read_sample()
    print("[OK] Layer 1 — sensor_reader.py")
    print(f"     Mode: {sensors.active_mode} | ax={s['ax']:.3f}, az={s['az']:.3f}")
    print(f"     Sim state: {sensors.sim_state} | Step counter: {sensors.step_counter}")
except Exception as e:
    print(f"[FAIL] Layer 1: {e}")

# Layer 2
try:
    ph = import_module("02_physics_logic.fast_physics_trigger")
    physics = ph.FastPhysicsTrigger()
    res_normal = physics.evaluate_sample({"ax":0.1,"ay":0.0,"az":9.81,"gx":1.0,"gy":0.5,"gz":0.2}, time.time())
    res_crash = physics.evaluate_sample({"ax":20.0,"ay":10.0,"az":5.0,"gx":320.0,"gy":100.0,"gz":200.0}, time.time())
    print("[OK] Layer 2 — fast_physics_trigger.py")
    print(f"     Normal sample:   {res_normal[1]}")
    print(f"     Crash sample:    {res_crash[1]} | is_extreme={res_crash[2]}")
except Exception as e:
    print(f"[FAIL] Layer 2: {e}")

# Layer 3
try:
    ml = import_module("03_ml_pipeline.ml_inference_engine")
    engine = ml.MLInferenceEngine(model_dir="models")
    print("[OK] Layer 3 — ml_inference_engine.py")
    print(f"     Model loaded: {engine.model is not None}")
    if not engine.model:
        print("     [INFO] model not found - would need to run: python src/train_model.py")
except Exception as e:
    print(f"[FAIL] Layer 3: {e}")

# Layer 4
try:
    arb = import_module("04_dual_layer_arbiter.arbiter")
    arbiter = arb.DualLayerArbiter()
    # Test 1: Normal state
    d, r = arbiter.evaluate((False, "NORMAL", False), (0, 0.02, 0.05), current_sample=s)
    # Test 2: Extreme crash override
    d2, r2 = arbiter.evaluate((True, "EXTREME_PHYSICS_OVERRIDE (accel=28g)", True), None, current_sample=s)
    print("[OK] Layer 4 — arbiter.py")
    print(f"     Normal test:          deploy={d} | {r}")
    print(f"     Extreme crash test:   deploy={d2} | {r2}")
    print(f"     Blackbox buffer size: {len(arbiter.blackbox_accel)} / {arbiter.max_blackbox_size}")
except Exception as e:
    print(f"[FAIL] Layer 4: {e}")

# Layer 5
try:
    act = import_module("05_actuation_alerts.actuator_alert_system")
    actuation = act.ActuatorAlertSystem(hardware=False)
    print("[OK] Layer 5 — actuator_alert_system.py")
    print(f"     Hardware GPIO: {actuation.gpio_ready} | Mode: simulation")
    print(f"     Emergency number: {actuation.emergency_number}")
except Exception as e:
    print(f"[FAIL] Layer 5: {e}")

# Layer 6
try:
    sim = import_module("06_simulation_demo.simulate_full_flow")
    print("[OK] Layer 6 — simulate_full_flow.py")
    print("     run_demo() function: ready")
except Exception as e:
    print(f"[FAIL] Layer 6: {e}")

# Logs
logs = os.path.join(".", "logs")
if os.path.exists(logs):
    files = os.listdir(logs)
    print(f"\n[OK] Logs Directory — {len(files)} file(s):")
    for f in files:
        size = os.path.getsize(os.path.join(logs, f))
        print(f"     {f} ({size} bytes)")
else:
    print("\n[INFO] Logs directory not yet created")

print("=" * 50)
print("  SYSTEM CHECK COMPLETE")
print("=" * 50)
