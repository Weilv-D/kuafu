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

### No-policy mode (baseline LQR only)

Running without ``policy.onnx`` is fully supported for teleop, hardware bring-up, and
baseline characterization. The ``--no-policy`` flag skips ``PolicyRuntime`` (no ONNX
loading, no manifest validation, no calibration-table digest check) and sends
zero-action residual frames. The STM32 firmware gates the RL residual off when the
action frame carries zeros — identical to a missing or stale action frame — so the
robot runs on the built-in 250 Hz LQR/LQI baseline alone.

Start the serial loop without an ONNX model:

```bash
cd /opt/kuafu && PYTHONPATH=/opt/kuafu python -m pi5_runtime.serial_node \
  --port /dev/ttyAMA10 \
  --no-policy \
  --enable-teleop
```

The ``--model`` argument is optional when ``--no-policy`` is set. All teleop behaviour
(arm/disarm/estop, stick shaping, hot-plug, haptics) is identical to the full-policy
path. The only difference is that the RL residual is not computed, so the robot has
no learned disturbance compensation — the LQR baseline holds balance, tracks velocity
and yaw commands, and levels roll via the fixed-gain D0 offset.

## Teleop (Gamepad)

Bluetooth/USB gamepad teleop runs as two processes connected by a Unix domain socket (`/tmp/kuafu-cmd.sock`, JSON payload `{v,omega,d0,mode}`). The command arbiter (`rl/teleop/arbiter.py`) runs inside `serial_node`, so it owns the safety layer — ramp, limit, estop, and timeout. The separate `teleop_node` process only reads the gamepad and forwards raw commands; if it crashes or the Bluetooth link drops, the arbiter's IPC source goes stale (0.5 s) and the robot parks at the safe default `[0, 0, D0_MIN]` instead of holding the last command.

Start the serial loop with teleop enabled, then the teleop publisher.
With the ONNX policy deployed:

```bash
cd /opt/kuafu && PYTHONPATH=/opt/kuafu python -m pi5_runtime.serial_node \
  --model /opt/kuafu/kuafu-policy.onnx \
  --port /dev/ttyAMA10 \
  --enable-teleop
```

Without the ONNX policy (baseline LQR only):

```bash
cd /opt/kuafu && PYTHONPATH=/opt/kuafu python -m pi5_runtime.serial_node \
  --port /dev/ttyAMA10 \
  --no-policy \
  --enable-teleop
```

```bash
cd /opt/kuafu && PYTHONPATH=/opt/kuafu python -m pi5_runtime.teleop_node --device gamepad
```

`--device keyboard` falls back to WASD/QE/Enter/Backspace/space when no gamepad is present; `teleop_node` also falls back if pygame itself is unavailable. On a headless Pi5, export `SDL_VIDEODRIVER=dummy` so pygame can run its event pump without a display.

### Gamepad layout

The source starts **DISARMED** (safe): the wheels hold balance via LQR but do not track commands, and the RL residual is gated off. The operator must explicitly arm before motion is possible.

| Input | Action |
|---|---|
| `START` (arm) | ARMED → firmware ACTIVE: wheels track commands, RL residual on. Also clears an ESTOP latch. |
| `Select` / `Back` (disarm) | DISARMED → firmware STAND: LQR still holds balance, but commands are ignored and the residual is off. |
| `A` (estop) | ESTOP latch → firmware FAULT: actuators disabled until reset. |
| Left stick Y | forward speed `v` (±0.5 m/s) |
| Right stick X | yaw rate `ω` (±1.0 rad/s) |
| LT / RT | crouch / stand D0 (rate-based, 40 mm/s, with a trigger deadzone) |

Stick response is shaped by a deadzone (`0.08`) then a square curve (`sign(x)·|x|²`), so small deflections produce small commands for precise low-speed control while full deflection still reaches the limit. Override the shaping via `ArbiterConfig` (`stick_deadzone`, `stick_gamma`, `trigger_deadzone`, `d0_rate_mm_s`).

Gamepad environment variables (defaults are Xbox-layout; Flydigi/PS/Switch usually differ):

| Variable | Default | Meaning |
|---|---|---|
| `KUAFU_AXIS_V` | 1 | left stick Y axis index |
| `KUAFU_AXIS_W` | 2 | right stick X axis index |
| `KUAFU_AXIS_LT` | 4 | LT trigger axis index |
| `KUAFU_AXIS_RT` | 5 | RT trigger axis index |
| `KUAFU_AXIS_V_INVERT` | 1 | invert the v axis (pygame Y is positive-down) |
| `KUAFU_AXIS_W_INVERT` | 0 | invert the ω axis |
| `KUAFU_BTN_ARM` | 7 | arm button (START) |
| `KUAFU_BTN_DISARM` | 6 | disarm button (`Select`/`Back`) |
| `KUAFU_BTN_ESTOP` | 0 | estop button (A) |
| `KUAFU_RUMBLE` | 1 | haptic feedback on arm/disarm/estop/reconnect; set 0 to disable |

Hot-plug is supported: if the controller disconnects, `poll()` returns ESTOP and the arbiter parks the robot; on reconnect a short rumble confirms, and the operator must arm again. Calibrate a new controller's axis and button mapping with `python -m rl.teleop.calibrate_native`; the interactive tool reads `/dev/input/js0` directly (bypassing pygame/SDL, which can drop events on Bluetooth LE devices), guides you through each stick and trigger, auto-detects the v-axis invert, and prints ready-to-export `KUAFU_AXIS_*` / `KUAFU_BTN_*` lines. If the gamepad goes silent after being connected, disconnect and reconnect it to wake it from its Bluetooth idle state.

Keyboard layout: `W/S` → v (±0.25 m/s), `A/D` → ω (±0.8 rad/s), `Q/E` → D0, `Enter` → arm, `Backspace` → disarm, `Space` → ESTOP, `R` → clear ESTOP latch. Keyboard commands are discrete gears and do not apply the stick curve.

The future Nav2 planner will reuse this path by feeding the `AutonomousSource` instead of the IPC source; the arbiter and policy are unchanged.

## Firmware

Before flashing, run `rl/verify/generate_artifacts.py`; `stm32_firmware/Core/Inc/kuafu_generated.h` must match `kuafu_physics.py`. The generated header carries the roll gains (`ROLL_KP = 190 mm/rad`, `ROLL_KD = 5.0`), the D0 high-speed gate thresholds, and the LQR/LQI gains. The firmware consumes version-1 frames, rejects replay and CRC errors, keeps partial DMA frames, and treats heartbeat freshness independently from action freshness.

HIL is mandatory before motor enable. Verify torque signs with wheels unloaded, verify servo dwell zero and mirror signs with torque limited, then proceed through the sequence in the acceptance matrix.
