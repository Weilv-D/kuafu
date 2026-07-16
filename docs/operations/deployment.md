# Deployment Operations

## Export

Export only a schema-`v1.1.0`-compatible 140-dimensional Actor checkpoint:

```bash
rl/.venv/bin/python rl/export/export_policy.py \
  --ckpt rl/checkpoints/ppo-v1-s42/teacher/model_5000.pt \
  --out artifacts/kuafu-policy.onnx
```

The exporter rejects incompatible checkpoint dimensions, checks bounded finite ONNX output, verifies Torch/ONNX maximum absolute error below `1e-5`, and writes `kuafu-policy.onnx.manifest.json`. The manifest contains schema version, physical-model hash, calibration-table hash, input/output dimensions, action names, and transform. Checkpoints load CPU-first, so export works on CPU-only hosts.

## Pi5 Runtime

Install `pi5_runtime/requirements.txt`, deploy the repository source tree (at least `kuafu_physics.py`, `rl/`, and `pi5_runtime/`), and copy the ONNX file, its manifest, and the generated `fivebar_ik_table.json` together. Run from that tree or set `PYTHONPATH` explicitly.

The Pi5↔STM32 link runs over the SoC PL011 on the board's 3-pin JST debug connector, which enumerates as `/dev/ttyAMA10`. The runtime user must be in the `dialout` group to open the device (`sudo usermod -aG dialout $USER`, then re-login or use `sg dialout -c '...'`). The UART loop is:

```bash
cd /opt/kuafu && PYTHONPATH=/opt/kuafu python -m pi5_runtime.serial_node \
  --model /opt/kuafu/kuafu-policy.onnx \
  --port /dev/ttyAMA10
```

`PolicyRuntime` validates the ONNX digest and calibration-table digest before startup, sends a model-hash HELLO session frame, maintains the four-frame causal history (reset to `[0, 0, 0, current]`), runs ONNX, clamps the action, and emits heartbeat/action frames with monotonically increasing sequences. Forward velocity and yaw-rate inputs are wheel-odometry estimates; `prev_applied_action` is the delayed action actually sent to actuators.

`pi5_runtime.serial_node` is the provided low-level UART adapter. It decodes STM32 IMU/joint frames, computes the hardware-equivalent Actor frame, enforces 100 ms telemetry freshness, sends the model-hash HELLO, and sends paired versioned frames every 20 ms. Its `set_command` method is an integration boundary, not a navigation arbiter; production navigation must place the project command arbiter above it and must not write actuator frames directly.

## Teleop (Gamepad)

Bluetooth/USB gamepad teleop runs as two processes connected by a Unix domain socket (`/tmp/kuafu-cmd.sock`, JSON payload `{v,omega,d0,mode}`). The command arbiter (`rl/teleop/arbiter.py`) runs inside `serial_node`, so it owns the safety layer — ramp, limit, estop, and timeout. The separate `teleop_node` process only reads the gamepad and forwards raw commands; if it crashes or the Bluetooth link drops, the arbiter's IPC source goes stale (0.5 s) and the robot parks at the safe default `[0, 0, D0_MIN]` instead of holding the last command.

Start the serial loop with teleop enabled, then the teleop publisher:

```bash
cd /opt/kuafu && PYTHONPATH=/opt/kuafu python -m pi5_runtime.serial_node \
  --model /opt/kuafu/kuafu-policy.onnx \
  --port /dev/ttyAMA10 \
  --enable-teleop
```

```bash
cd /opt/kuafu && PYTHONPATH=/opt/kuafu python -m pi5_runtime.teleop_node --device gamepad
```

`--device keyboard` falls back to WASD/QE/space when no gamepad is present; `teleop_node` also auto-falls back if no joystick is detected. On a headless Pi5, export `SDL_VIDEODRIVER=dummy` so pygame can run its event pump without a display. Gamepad layout: left stick Y → forward speed, right stick X → yaw rate, LT/RT → crouch/stand (D0), A or B → emergency stop (latches firmware FAULT until reset). The future Nav2 planner will reuse this path by feeding the `AutonomousSource` instead of the IPC source; the arbiter and policy are unchanged.

## Firmware

Before flashing, run `rl/verify/generate_artifacts.py`; `stm32_firmware/Core/Inc/kuafu_generated.h` must match `kuafu_physics.py`. The generated header carries the roll gains (`ROLL_KP = 190 mm/rad`, `ROLL_KD = 5.0`), the D0 high-speed gate thresholds, and the LQR/LQI gains. The firmware consumes version-1 frames, rejects replay and CRC errors, keeps partial DMA frames, and treats heartbeat freshness independently from action freshness.

HIL is mandatory before motor enable. Verify torque signs with wheels unloaded, verify servo dwell zero and mirror signs with torque limited, then proceed through the sequence in the acceptance matrix.
