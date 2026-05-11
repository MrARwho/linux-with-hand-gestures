"""D-Bus Action Dispatcher and Key Simulation."""

from __future__ import annotations

import time
from collections.abc import Callable

from src.config import (
    SUPER_KEYS,
    SUPER_KEYS_ALT,
    SUPER_KEYS_SCROLL,
    SUPER_KEYS_SHIFT,
)


class ActionDispatcher:
    """Dispatches gestures to actions via D-Bus and key simulation."""

    def __init__(self) -> None:
        self._dbus = None  # lazy init
        self._key_sim = None  # lazy init
        self._pending_super: list[str] = []
        self._pending_super_timer: float = 0.0
        self._pending_super_timeout = 1.5

    @staticmethod
    def _get_dbus() -> object:
        if _get_dbus._dbus is None:  # type: ignore[attr-defined]
            import dbus
            _get_dbus._dbus = dbus.SessionBus()  # type: ignore[attr-defined]
        return _get_dbus._dbus  # type: ignore[attr-defined]

    @property
    def dbus(self) -> object:
        return self._get_dbus()

    def _simulate_key(self, gesture: str) -> None:
        if self._key_sim is None:
            import pyautogui
            self._key_sim = pyautogui
        if gesture in _key_map:
            self._key_sim.press(_key_map[gesture])

    def _release_super(self) -> None:
        if self._key_sim is None:
            import pyautogui
            self._key_sim = pyautogui
        for k in reversed(self._pending_super):
            self._key_sim.release(k)
        self._pending_super.clear()

    def _clear_pending(self) -> None:
        if time.time() - self._pending_super_timer > self._pending_super_timeout:
            self._release_super()

    def dispatch(self, gesture: str) -> None:
        """Execute action for gesture."""
        self._clear_pending()

        if gesture not in _action_map:
            return

        action = _action_map[gesture]
        action(self)

    def press_super(self, key: str) -> None:
        if self._key_sim is None:
            import pyautogui
            self._key_sim = pyautogui
        self._key_sim.press(key)
        self._pending_super.append(key)
        self._pending_super_timer = time.time()

    def release_super(self) -> None:
        self._release_super()


_key_map: dict[str, str] = {
    "volume_up": "volumeup",
    "volume_down": "volumedown",
    "volume_mute": "mute",
    "brightness_up": "brightnessup",
    "brightness_down": "brightnessdown",
    "media_play_pause": "play",
    "media_next_track": "next",
    "media_previous_track": "previous",
    "screenshot": "print",
    "lock_screen": "l",
    "calculator": "c",
    "file_manager": "e",
}


def _super_key(dispatcher: ActionDispatcher, gesture: str, keys: list[str]) -> None:
    dispatcher.press_super(keys[0])


def _super_key_release(dispatcher: ActionDispatcher, gesture: str) -> None:
    dispatcher.release_super()


def _super_key_combo(dispatcher: ActionDispatcher, gesture: str, keys: list[str]) -> None:
    for key in keys:
        dispatcher.press_super(key)


def _key_press(dispatcher: ActionDispatcher, gesture: str) -> None:
    dispatcher._simulate_key(gesture)


def _swipe(dispatcher: ActionDispatcher, gesture: str) -> None:
    swipe_map = {
        "swipe_left": ["super", "left"],
        "swipe_right": ["super", "right"],
        "swipe_up": ["alt", "tab"],
        "swipe_down": ["alt", "shift", "tab"],
    }
    keys = swipe_map.get(gesture, [])
    for key in keys:
        dispatcher.press_super(key)
    for key in reversed(keys):
        dispatcher._key_sim.release(key)


_action_map: dict[str, Callable[[ActionDispatcher, str], None]] = {}

for g in SUPER_KEYS:
    _action_map[g] = lambda d, g=g: _super_key(d, g, SUPER_KEYS[g])

for g in SUPER_KEYS_ALT:
    _action_map[g] = lambda d, g=g: _super_key_combo(d, g, ["alt"] + SUPER_KEYS_ALT[g])

for g in SUPER_KEYS_SCROLL:
    _action_map[g] = lambda d, g=g: _super_key_combo(d, g, ["shift"] + SUPER_KEYS_SCROLL[g])

for g in SUPER_KEYS_SHIFT:
    _action_map[g] = lambda d, g=g: _super_key_combo(d, g, ["shift"] + SUPER_KEYS_SHIFT[g])

for g in _key_map:
    _action_map[g] = _key_press

for g in ["swipe_left", "swipe_right", "swipe_up", "swipe_down"]:
    _action_map[g] = _swipe
