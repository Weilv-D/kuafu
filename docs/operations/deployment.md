# Deployment Operations

## Export

Export only a schema-compatible 140-dimensional Actor checkpoint:

```bash
rl/.venv/bin/python rl/export/export_policy.py \
  --ckpt rl/checkpoints/stage-s7/teacher/model_5000.pt \
  --out artifacts/kuafu-policy.onnx
```

The exporter rejects incompatible checkpoint dimensions, checks bounded finite ONNX output, verifies Torch/ONNX maximum absolute error below `1e-5`, and writes `kuafu-policy.onnx.manifest.json`. The manifest contains schema version, physical-model hash, calibration-table hash, input/output dimensions, action names, and transform.

## Pi5 Runtime

Install `pi5_runtime/requirements.txt`, deploy the repository source tree (at least `kuafu_physics.py`, `rl/`, and `pi5_runtime/`), and copy the ONNX file, its manifest, and the generated `fivebar_ik_table.json` together. Run from that tree or set `PYTHONPATH` explicitly. The actual UART loop is:

```bash
cd /opt/kuafu && PYTHONPATH=/opt/kuafu python -m pi5_runtime.serial_node \
  --model /opt/kuafu/kuafu-policy.onnx \
  --port /dev/ttyAMA0
```

`PolicyRuntime` validates the ONNX digest and calibration-table digest before startup, sends a model-hash HELLO session frame, maintains the four-frame causal history, runs ONNX, clamps the action, and emits heartbeat/action frames with monotonically increasing sequences. The runtime-only command remains useful for manifest validation, but does not open a serial port.

`pi5_runtime.serial_node` is the provided low-level UART adapter. It decodes STM32 IMU/joint frames, computes the hardware-equivalent Actor frame, enforces 100 ms telemetry freshness, sends the model-hash HELLO, and sends paired versioned frames every 20 ms. Its `set_command` method is an integration boundary, not a navigation arbiter; production navigation must place the project command arbiter above it and must not write actuator frames directly.

## Firmware

Before flashing, run `rl/verify/generate_artifacts.py`; `stm32_firmware/Core/Inc/kuafu_generated.h` must match `kuafu_physics.py`. The firmware consumes version-1 frames, rejects replay and CRC errors, keeps partial DMA frames, and treats heartbeat freshness independently from action freshness.

HIL is mandatory before motor enable. Verify torque signs with wheels unloaded, verify servo dwell zero and mirror signs with torque limited, then proceed through the sequence in the acceptance matrix.
