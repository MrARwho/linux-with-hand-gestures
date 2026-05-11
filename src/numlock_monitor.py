"""NumLock LED state monitor for gesture mode gating."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

NUMLOCK_PATHS = [
    "/sys/class/leds/numlock",
    "/sys/class/leds/sysrq\:numlock",
]


def _find_numlock_led() -> str | None:
    for path in NUMLOCK_PATHS:
        if os.path.exists(path):
            return path
    return None


def _read_numlock_state(path: str) -> bool:
    brightness_path = os.path.join(path, "brightness")
    try:
        with open(brightness_path, "r") as f:
            val = int(f.read().strip())
        return val != 0
    except (OSError, ValueError):
        return False


def _read_numlock_state_dmesg() -> bool:
    """Read NumLock state from kernel LED trigger."""
    for led_dir in NUMLOCK_PATHS:
        trigger_path = os.path.join(led_dir, "trigger")
        if os.path.exists(trigger_path):
            try:
                with open(trigger_path, "r") as f:
                    trigger = f.read().strip().strip("[]")
                if "numlock" in trigger:
                    return _read_numlock_state(led_dir)
            except OSError:
                continue
    return False


class NumLockMonitor:
    """Monitors NumLock LED state and fires callbacks on toggle."""

    def __init__(self, poll_interval: float = 0.2) -> None:
        self.poll_interval = poll_interval
        self.led_path = _find_numlock_led() or _read_numlock_state_dmesg()
        self._led_path = self.led_path
        self._is_on = False
        self._running = False
        self._callbacks: list[Callable[[bool], None]] = []
        self._last_state: bool | None = None

    @property
    def is_gesture_mode(self) -> bool:
        return self._is_on

    @property
    def led_path(self) -> str | None:
        return self._led_path

    @led_path.setter
    def led_path(self, path: str | None) -> None:
        self._led_path = path

    def add_callback(self, callback: Callable[[bool], None]) -> None:
        self._callbacks.append(callback)

    def _fire(self, state: bool) -> None:
        for cb in self._callbacks:
            cb(state)

    def start(self) -> None:
        self._running = True
        self._last_state = None
        if not self.led_path:
            return

        while self._running:
            try:
                current = _read_numlock_state(self.led_path)
                if self._last_state is None:
                    self._is_on = current
                    self._last_state = current
                elif current != self._last_state:
                    self._last_state = current
                    self._is_on = current
                    self._fire(current)
            except OSError:
                pass
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False
