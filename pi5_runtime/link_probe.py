"""Pi5 <-> STM32 串口通信探针 (无需 RL 模型即可验证链路)。

用途
----
在没有 policy.onnx 的情况下验证 Pi5↔STM32 的物理串口链路、协议编解码、
STM32 状态机响应。绕过 PolicyRuntime (强依赖 ONNX)，只复用 protocol.py 的
已验证编解码。

行为
----
1. 打开 /dev/ttyAMA10 @ 921600
2. 发送 HELLO 帧 (用 kuafu_physics.model_hash()) —— STM32 校验后置 link_compatible
3. 以 50 Hz 周期发送成对的 HEARTBEAT + 零 ACTION 帧
   - mode=1 (STAND): 只做平衡 hold + D0 跟踪, RL 残差不生效 (安全)
   - 零 ACTION 在 STAND 模式下被固件忽略 (非 ACTIVE)
4. 实时解码并打印 STM32 回传的 IMU / JOINTS / HEALTH / DIAG
5. Ctrl-C 退出时发 ESTOP (mode=4) 确保安全

用法
----
    sg dialout -c "PYTHONPATH=/path/to/kuafu_repo \
        .venv/bin/python -m pi5_runtime.link_probe"
    # 或指定参数:
    ... python -m pi5_runtime.link_probe --port /dev/ttyAMA10 --mode 1

判定通信成功的标志
------------------
- 收到 IMU / JOINTS 帧, 数值合理
- HEALTH 的 mode 从 0(INIT) 推进到 1(STAND)
- fault_mask == 0
- 各设备 age 在合理范围 (IMU~4ms, 轮~8ms, 舵机~20ms)
"""

from __future__ import annotations

import argparse
import struct
import time

import numpy as np

import kuafu_physics as P
from pi5_runtime.protocol import (
    Frame,
    StreamDecoder,
    TEL_DIAG,
    TEL_HEALTH,
    TEL_IMU,
    TEL_JOINTS,
    command_frames,
    decode_health_payload,
    hello_frame,
)

# 固件 RobotMode 整数码 (见 docs/architecture/system.md operating-modes 表)
MODE_INIT = 0
MODE_STAND = 1
MODE_ACTIVE = 2
MODE_CLIMB = 3
MODE_FAULT = 4
MODE_NAMES = {0: "INIT", 1: "STAND", 2: "ACTIVE", 3: "CLIMB", 4: "FAULT"}

# 固件 fault_mask 位定义 (见 safety_state.c, docs/architecture/system.md 十类故障)
FAULT_BITS = {
    0: "OVER_TILT",
    1: "OVER_PITCH_RATE",
    2: "OVER_TEMP",
    3: "IMU_LOST",
    4: "WHEEL_L_LOST",
    5: "WHEEL_R_LOST",
    6: "SERVO_LOST",
    7: "ESTOP",
    8: "INIT_FAILED",
    9: "INTERNAL",
}


def fault_str(mask: int) -> str:
    bits = [name for bit, name in FAULT_BITS.items() if mask & (1 << bit)]
    return f"0x{mask:08x}" + (f" ({','.join(bits)})" if bits else " (none)")


def run_loopback(port: str, baudrate: int, duration: float = 3.0) -> None:
    """回环测试: 短接 JST 调试口的 TX/RX, Pi5 发的数据应能被自己收到。

    用途: 定位 Pi5 端 UART 收发硬件是否正常 (与 STM32 端问题隔离)。
    操作: 断开 STM32, 用一根导线短接 JST 3-pin 的 TX 和 RX 引脚, 然后运行。
    """
    import serial

    print("=" * 64)
    print("Pi5 串口回环测试 (请确保 JST 调试口 TX/RX 已短接)")
    print("=" * 64)
    print(f"串口: {port} @ {baudrate}")
    print(f"持续: {duration}s")
    print("判定: 收到带 'LOOPBACK' 标记的数据 → Pi5 端 UART 硬件正常")
    print("-" * 64)

    ser = serial.Serial(port, baudrate=baudrate, timeout=0)
    marker = b"LOOPBACK-PING-MARKER-0123456789\r\n"
    sent = 0
    received = 0
    start = time.monotonic()
    deadline = start

    try:
        while time.monotonic() - start < duration:
            ser.write(marker)
            sent += 1
            chunk = ser.read(256)
            if chunk:
                received += chunk.count(b"LOOPBACK")
            deadline += 0.05
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        ser.close()

    print("-" * 64)
    print(f"发送 {sent} 次标记, 收到 {received} 次 'LOOPBACK'")
    if received > 0:
        print("✅ 回环成功 → Pi5 端 UART TX/RX 硬件正常")
        print("   问题在 STM32 侧: 接线方向(TX/RX交叉)/STM32未上电/固件未跑")
    else:
        print("❌ 回环失败 → Pi5 端 UART 收发不通")
        print("   检查: TX/RX 是否真短接、串口设备是否正确、波特率")


def make_heartbeat_action_frames(sequence: int, timestamp_ms: int, mode: int):
    """构造成对的 HEARTBEAT + 零 ACTION 帧, 复用 protocol.command_frames。

    command_frames 要求 action 长度 == ACTION_DIM 且每个值在 [-1,1]; 零向量满足。
    """
    action = np.zeros(6, dtype=np.float32)
    return command_frames(sequence, timestamp_ms, mode, 0.0, 0.0, float(P.D0_MIN), action)


def main() -> None:
    parser = argparse.ArgumentParser(description="KUAFU Pi5<->STM32 串口通信探针 (无需模型)")
    parser.add_argument("--port", default="/dev/ttyAMA10",
                        help="串口设备 (Pi5 JST 调试口 = /dev/ttyAMA10)")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--mode", type=int, default=MODE_STAND,
                        help="请求的固件模式: 0=INIT 1=STAND 2=ACTIVE 3=CLIMB 4=FAULT(ESTOP)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行秒数, 0=无限 (Ctrl-C 退出)")
    parser.add_argument("--quiet", action="store_true",
                        help="只打印摘要, 不逐帧打印 IMU/Joints")
    parser.add_argument("--loopback", action="store_true",
                        help="回环测试模式: 短接 JST TX/RX 验证 Pi5 UART 硬件")
    parser.add_argument("--hash", default=None,
                        help="覆盖 HELLO 帧的 model_hash (16位hex); "
                             "默认用 kuafu_physics.model_hash()。"
                             "用于与已烧录但未更新哈希的固件握手")
    parser.add_argument("--listen-only", action="store_true",
                        help="只监听 STM32 主动上报的遥测, 不发送任何命令/心跳/ESTOP。"
                             "不会影响机器人当前模式。注意: 如果已有程序控制串口, "
                             "此探针会抢占端口, 导致原控制器失效。")
    args = parser.parse_args()

    if args.loopback:
        run_loopback(args.port, args.baudrate, duration=args.duration or 3.0)
        return

    import serial

    model_hash = args.hash if args.hash else P.model_hash()

    print("=" * 64)
    print("KUAFU Pi5 <-> STM32 通信探针" + (" [只监听, 不发命令]" if args.listen_only else ""))
    print("=" * 64)
    print(f"串口: {args.port} @ {args.baudrate}")
    if args.listen_only:
        print("模式: 只监听遥测 (不发送 HELLO/心跳/ESTOP)")
    else:
        print(f"请求模式: {MODE_NAMES.get(args.mode, args.mode)}")
        print(f"model_hash (HELLO): {model_hash}", "(覆盖)" if args.hash else "")
    print(f"D0_MIN: {P.D0_MIN} mm")
    if args.listen_only:
        print("按 Ctrl-C 退出 (不会发送 ESTOP)")
    else:
        print("按 Ctrl-C 停止 (会发 ESTOP)")
    print("-" * 64)

    ser = serial.Serial(args.port, baudrate=args.baudrate, timeout=0)
    decoder = StreamDecoder()
    sequence = 0
    start = time.monotonic()

    # 统计
    counts = {TEL_IMU: 0, TEL_JOINTS: 0, TEL_HEALTH: 0, TEL_DIAG: 0}
    last_health = None
    last_print = 0.0
    last_mode_seen = None

    if not args.listen_only:
        # 1. 发送 HELLO
        ts = int(time.monotonic() * 1000)
        hello = hello_frame(sequence, ts, model_hash).encode()
        ser.write(hello)
        sequence += 1
        print(f"[{0:6.2f}s] TX HELLO ({len(hello)}B) model_hash={model_hash}")

    # 2. 50Hz 循环: 发 heartbeat+action (非监听模式), 收 telemetry
    period = 0.02
    deadline = time.monotonic()
    try:
        while True:
            if args.duration > 0 and (time.monotonic() - start) > args.duration:
                break

            if not args.listen_only:
                # 发送成对帧
                ts = int(time.monotonic() * 1000) & 0xFFFFFFFF
                hb, act = make_heartbeat_action_frames(sequence, ts, args.mode)
                ser.write(hb.encode() + act.encode())
                sequence += 2

            # 读取回传
            chunk = ser.read(512)
            for frame in decoder.feed(chunk):
                if frame.type in counts:
                    counts[frame.type] += 1
                _handle_frame(frame, args.quiet, time.monotonic() - start)

                # 记录并检测模式变化
                if frame.type == TEL_HEALTH:
                    h = decode_health_payload(frame.payload)
                    last_health = h
                    if h.mode != last_mode_seen:
                        print(f"[{time.monotonic()-start:6.2f}s] *** STM32 模式变化: "
                              f"{MODE_NAMES.get(last_mode_seen,'?')} -> "
                              f"{MODE_NAMES.get(h.mode, h.mode)} ***")
                        last_mode_seen = h.mode

            # 摘要打印 (每 2 秒一次, quiet 模式下更频繁的诊断)
            now = time.monotonic()
            if not args.quiet:
                if now - last_print >= 2.0:
                    _print_summary(counts, last_health, now - start)
                    last_print = now
            else:
                if now - last_print >= 5.0:
                    _print_summary(counts, last_health, now - start)
                    last_print = now

            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))

    except KeyboardInterrupt:
        print("\n[Ctrl-C] 停止中...")
    finally:
        if not args.listen_only:
            # 发送 ESTOP
            ts = int(time.monotonic() * 1000) & 0xFFFFFFFF
            hb, act = make_heartbeat_action_frames(sequence, ts, MODE_FAULT)
            try:
                ser.write(hb.encode() + act.encode())
                print(f"[ESTOP] 已发送 mode=FAULT(4) 帧")
            except Exception:
                pass
        ser.close()
        print("\n" + "=" * 64)
        _print_summary(counts, last_health, time.monotonic() - start)
        print("=" * 64)
        _print_verdict(counts, last_health, last_mode_seen)


def _handle_frame(frame: Frame, quiet: bool, elapsed: float) -> None:
    """处理单帧并按需打印。"""
    if quiet:
        return
    if frame.type == TEL_IMU and len(frame.payload) == 12:
        vals = struct.unpack(">" + "h" * 6, frame.payload)
        roll, pitch, yaw, gx, gy, gz = (v / 1000.0 for v in vals)
        print(f"[{elapsed:6.2f}s] RX IMU  roll={roll:+.3f} pitch={pitch:+.3f} "
              f"yaw={yaw:+.3f} gyro=({gx:+.3f},{gy:+.3f},{gz:+.3f})")
    elif frame.type == TEL_JOINTS and len(frame.payload) == 36:
        vals = struct.unpack(">" + "h" * 18, frame.payload)
        v = [x / 1000.0 for x in vals]
        v[2] = vals[2] / 10000.0
        v[5] = vals[5] / 10000.0
        # 顺序 [wheel_L, wheel_R, A_l, A_r, B_l, B_r] 每组 (pos,vel,tau)
        print(f"[{elapsed:6.2f}s] RX JNT  wheelL=({v[0]:+.2f},{v[1]:+.3f}) "
              f"wheelR=({v[3]:+.2f},{v[4]:+.3f})")
    elif frame.type == TEL_HEALTH and len(frame.payload) == 46:
        h = decode_health_payload(frame.payload)
        ages = f"imu={h.imu_age_ms}ms wheel=({h.wheel_age_ms[0]},{h.wheel_age_ms[1]})ms"
        serr = f"servo=({','.join(str(x) for x in h.servo_age_ms)})ms"
        print(f"[{elapsed:6.2f}s] RX HLTH mode={MODE_NAMES.get(h.mode,h.mode)} "
              f"fault={fault_str(h.fault_mask)}")
        print(f"           age: {ages} {serr}")
        print(f"           errs: imu={h.imu_errors} wheel=({h.wheel_errors[0]},{h.wheel_errors[1]}) "
              f"servo=({','.join(str(x) for x in h.servo_errors)})")
        # DDSM 错误类型细分 (timeout=不回复 / checksum=脏帧 / protocol=硬件错误)
        lt, lc, lp = h.wheel_error_breakdown[0]
        rt, rc, rp = h.wheel_error_breakdown[1]
        print(f"           wheel errs breakdown: L=(t/o={lt},chk={lc},proto={lp}) "
              f"R=(t/o={rt},chk={rc},proto={rp})")
    elif frame.type == TEL_DIAG and len(frame.payload) == 4:
        bat_mv, temp_c, legacy = struct.unpack(">HBB", frame.payload)
        print(f"[{elapsed:6.2f}s] RX DIAG battery={bat_mv}mV temp={temp_c}C "
              f"(bat=0 表示未接线)")


def _print_summary(counts: dict, health, elapsed: float) -> None:
    print(f"\n--- 摘要 @{elapsed:.1f}s ---")
    print(f"  收到帧数: IMU={counts[TEL_IMU]} Joints={counts[TEL_JOINTS]} "
          f"Health={counts[TEL_HEALTH]} Diag={counts[TEL_DIAG]}")
    if health is not None:
        print(f"  当前模式: {MODE_NAMES.get(health.mode, health.mode)}")
        print(f"  故障掩码: {fault_str(health.fault_mask)}")
    print()


def _print_verdict(counts: dict, health, last_mode: int | None) -> None:
    """通信验证结论。"""
    print("通信验证结论:")
    total_rx = sum(counts.values())
    if total_rx == 0:
        print("  ❌ 未收到任何 STM32 数据 —— 检查接线(TX/RX交叉)、GND、波特率、")
        print("     Pi5 UART 是否启用、STM32 是否上电并烧录了固件")
        return
    print(f"  ✅ 收到 {total_rx} 帧 (IMU={counts[TEL_IMU]} Joints={counts[TEL_JOINTS]} "
          f"Health={counts[TEL_HEALTH]} Diag={counts[TEL_DIAG]})")
    if health is not None:
        if health.fault_mask == 0:
            print(f"  ✅ 无故障锁存 (fault_mask=0)")
        else:
            print(f"  ⚠️  有故障: {fault_str(health.fault_mask)}")
            print("     对照 docs/architecture/system.md 的十类故障码定位")
        if last_mode == MODE_INIT:
            print(f"  ⚠️  STM32 卡在 INIT —— 某设备未 fresh, 看 Health 的 age/errors")
        elif last_mode == MODE_STAND:
            print(f"  ✅ STM32 处于 STAND (设备 fresh, 平衡保持)")
            print(f"     注: STAND 不需要握手。若请求了 ACTIVE 却停在 STAND,")
            print(f"     说明 link_compatible=0 (model_hash 不匹配固件)")
        elif last_mode == MODE_ACTIVE:
            print(f"  ✅ STM32 已进入 ACTIVE (握手成功, RL 残差策略已启用)")
        elif last_mode == MODE_FAULT:
            print(f"  ⚠️  STM32 在 FAULT (见上方故障掩码)")
        else:
            print(f"  ℹ️  STM32 模式: {MODE_NAMES.get(last_mode, last_mode)}")


if __name__ == "__main__":
    main()
