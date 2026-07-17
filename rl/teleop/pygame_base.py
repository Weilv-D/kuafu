# -*- coding: utf-8 -*-
"""pygame bootstrap shared by the gamepad and keyboard sources.

On a headless Pi5 the display falls back to a dummy driver so the event pump
still runs. ``pump_events`` returns the event queue so callers (e.g. the
gamepad hot-plug handler) can react to ``JOYDEVICEADDED`` / ``JOYDEVICEREMOVED``.
"""
from __future__ import annotations

import os

import pygame

_initialized = False


def init_pygame(window_title: str = "KUAFU teleop") -> None:
    """Idempotent pygame init with a 1x1 window; safe to call from both sources."""
    global _initialized
    if _initialized:
        return
    if os.environ.get("SDL_VIDEODRIVER") is None and not os.environ.get("DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.joystick.init()
    try:
        pygame.display.set_mode((1, 1), pygame.SCALED)
        pygame.display.set_caption(window_title)
    except pygame.error:
        # No display at all (truly headless without dummy): events still pump.
        pass
    _initialized = True


def pump_events() -> list:
    """Pump the pygame event queue and return the drained events.

    ``pygame.event.pump()`` only updates internal state without returning
    events; for hot-plug handling we need the actual ``JOYDEVICE*`` events, so
    we drain the queue here. The pump still runs as part of ``event.get``.
    """
    if not _initialized:
        init_pygame()
    return pygame.event.get()
