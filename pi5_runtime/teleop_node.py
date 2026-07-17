"""Standalone teleop process — reads a gamepad/keyboard and sends raw commands.

This process owns **no safety layer**. It polls a :class:`~rl.teleop.command.CommandSource`
(gamepad, falling back to keyboard) at ~50 Hz and forwards each ``Command`` over a
Unix socket to the serial_node process, which runs the
:class:`~rl.teleop.arbiter.CommandArbiter`. Keeping the arbiter in serial_node means
that if this process dies or the Bluetooth link drops, serial_node's IPC source
goes stale and the arbiter emits its safe default — the robot stops rather than
running the last command.

Run it alongside serial_node (which must be started with ``--enable-teleop``):

    PYTHONPATH=/opt/kuafu python -m pi5_runtime.teleop_node --device gamepad

On a headless Pi5, export ``SDL_VIDEODRIVER=dummy`` (or run under a display) so
pygame can create the event window its input pump needs; ``pygame_base`` already
falls back to a dummy display if none is available.
"""

from __future__ import annotations

import argparse
import signal
import time

from pi5_runtime.command_socket import COMMAND_SOCKET_PATH, CommandSocketClient

# Mode integer codes on the wire mirror the firmware RobotMode.
# MANUAL=ACTIVE(2) 出力; IDLE=STAND(1) 保平衡不跟走; ESTOP=FAULT(4) 锁存急停。
def _mode_to_wire(mode) -> int:
    from rl.teleop.command import Mode
    if mode == Mode.ESTOP:
        return 4
    if mode == Mode.IDLE:
        return 1
    return 2  # MANUAL / AUTONOMOUS / ASSISTED -> ACTIVE


def _make_source(device: str):
    """Build the requested CommandSource, falling back to keyboard on failure."""
    if device == "keyboard":
        from rl.teleop.keyboard_source import KeyboardSource
        return KeyboardSource()
    # device == "gamepad": GamepadSource no longer raises on a missing joystick
    # (it waits for hot-plug and returns ESTOP); fall back to keyboard only when
    # explicitly requested or when pygame itself is unavailable.
    try:
        from rl.teleop.gamepad_source import GamepadSource
        src = GamepadSource()
    except Exception as exc:  # pygame missing / display init failure
        print(f"[teleop] gamepad source unavailable ({exc}); falling back to keyboard")
        from rl.teleop.keyboard_source import KeyboardSource
        return KeyboardSource()
    joy = src._joy
    if joy is not None:
        print(f"[teleop] {joy.get_name()}: "
              f"v=ax{src._axis_v}{'(inv)' if src._invert_v else ''} "
              f"w=ax{src._axis_w}{'(inv)' if src._invert_w else ''} "
              f"lt=ax{src._axis_lt} rt=ax{src._axis_rt} "
              f"arm=btn{src._btn_arm} disarm=btn{src._btn_disarm} "
              f"estop=btn{src._btn_estop}")
    else:
        print("[teleop] gamepad source started without a controller; "
              "waiting for hot-plug (poll() returns ESTOP until connect)")
    print("[teleop] (override with KUAFU_AXIS_V/W/LT/RT, KUAFU_BTN_ARM/DISARM/ESTOP, "
          "KUAFU_AXIS_V/W_INVERT, KUAFU_RUMBLE)")
    return src


def main() -> None:
    parser = argparse.ArgumentParser(description="KUAFU teleop command publisher")
    parser.add_argument("--device", choices=("gamepad", "keyboard"), default="gamepad",
                        help="command input device")
    parser.add_argument("--cmd-socket", default=COMMAND_SOCKET_PATH,
                        help="path to the command Unix socket served by serial_node")
    parser.add_argument("--rate", type=float, default=50.0, help="poll rate in Hz")
    args = parser.parse_args()

    source = _make_source(args.device)
    client = CommandSocketClient(args.cmd_socket)

    # Reconnect helper: serial_node may not be up yet, or the link may drop.
    def ensure_connected() -> None:
        if not client.connected():
            print(f"[teleop] connecting to {args.cmd_socket} ...")
            client.connect()
            print("[teleop] connected")

    stopping = {"flag": False}

    def _shutdown(_signum, _frame) -> None:
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    ensure_connected()
    period = 1.0 / max(args.rate, 1.0)
    deadline = time.monotonic()

    try:
        while not stopping["flag"]:
            cmd = source.poll()
            if cmd is not None:
                wire_mode = _mode_to_wire(cmd.mode)
                try:
                    client.send_command(cmd.v, cmd.omega, cmd.d0, wire_mode)
                except (BrokenPipeError, ConnectionResetError, ConnectionError):
                    # Link lost; reconnect on the next iteration. While
                    # disconnected, serial_node's IPC source goes stale and the
                    # arbiter parks the robot safely.
                    print("[teleop] socket link lost; reconnecting ...")
                    try:
                        ensure_connected()
                    except KeyboardInterrupt:
                        stopping["flag"] = True
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        # On exit, send one ESTOP so the robot does not hold the last command.
        # d0/v/omega are inert under ESTOP but must still satisfy the wire schema;
        # use D0_MIN rather than a bare literal so it tracks kuafu_physics.
        from rl.teleop.command import D0_CMD_RANGE
        if client.connected():
            try:
                client.send_command(0.0, 0.0, D0_CMD_RANGE[0], 4)
            except (BrokenPipeError, ConnectionResetError, ConnectionError):
                pass
        client.close()
        print("[teleop] stopped")


if __name__ == "__main__":
    main()
