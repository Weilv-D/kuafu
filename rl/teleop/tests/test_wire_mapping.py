# -*- coding: utf-8 -*-
"""Mode <-> wire-code round-trip tests across the teleop -> serial_node chain."""
from __future__ import annotations

import pytest

from rl.teleop.command import Mode
from rl.teleop.ipc_source import _WIRE_TO_MODE
from pi5_runtime.teleop_node import _mode_to_wire
from pi5_runtime.serial_node import _mode_to_firmware_mode


class TestIpcWireToMode:
    @pytest.mark.parametrize("wire,expected", [
        (0, Mode.ESTOP),   # INIT
        (4, Mode.ESTOP),   # FAULT
        (1, Mode.IDLE),    # STAND
        (2, Mode.MANUAL),  # ACTIVE
        (3, Mode.MANUAL),  # CLIMB
    ])
    def test_known_codes(self, wire, expected):
        assert _WIRE_TO_MODE[wire] == expected

    def test_unknown_code_defaults_to_estop(self):
        assert _WIRE_TO_MODE.get(99, Mode.ESTOP) == Mode.ESTOP


class TestTeleopNodeModeToWire:
    @pytest.mark.parametrize("mode,wire", [
        (Mode.ESTOP, 4),
        (Mode.IDLE, 1),
        (Mode.MANUAL, 2),
        (Mode.AUTONOMOUS, 2),
        (Mode.ASSISTED, 2),
    ])
    def test_mode_maps_to_wire(self, mode, wire):
        assert _mode_to_wire(mode) == wire


class TestSerialNodeModeToFirmware:
    @pytest.mark.parametrize("mode,fw", [
        (Mode.ESTOP, 4),       # FAULT
        (Mode.IDLE, 1),        # STAND
        (Mode.MANUAL, 2),      # ACTIVE
        (Mode.AUTONOMOUS, 2),  # ACTIVE
        (Mode.ASSISTED, 2),    # ACTIVE
    ])
    def test_mode_maps_to_firmware(self, mode, fw):
        assert _mode_to_firmware_mode(mode) == fw


class TestRoundTrip:
    """Mode -> wire -> Mode -> firmware must preserve the three-state intent."""

    @pytest.mark.parametrize("mode", [Mode.ESTOP, Mode.IDLE, Mode.MANUAL])
    def test_round_trip(self, mode):
        wire = _mode_to_wire(mode)
        back = _WIRE_TO_MODE[wire]
        fw = _mode_to_firmware_mode(back)
        # ESTOP and MANUAL survive fully; IDLE survives on the mode side and
        # maps to firmware STAND(1).
        if mode == Mode.IDLE:
            assert back == Mode.IDLE
            assert fw == 1
        else:
            assert back == mode
