"""
Bheema Keyboard Teleop (Arrow Keys)
=====================================
Non-blocking keyboard listener for real-time velocity commands.
Uses arrow keys to avoid conflicts with MuJoCo viewer shortcuts.

Controls:
    ↑ / ↓       Forward / Backward
    ← / →       Strafe Left / Right
    Z / X       Turn Left / Right
    SPACE       Emergency stop
    ESC         Quit

Usage:
    from teleop import Teleop
    teleop = Teleop()
    teleop.start()
    vx, vy, z_pos, yaw_rate = teleop.get_cmd()
"""

import numpy as np
import threading
import time
import sys

MAX_VX = 1.0
MAX_VY = 0.4
MAX_YAW_RATE = 0.6
NOMINAL_Z = 0.66
Z_STEP = 0.02
RAMP_RATE = 2.0
YAW_RAMP_RATE = 1.5


class Teleop:
    def __init__(self, max_vx=MAX_VX, max_vy=MAX_VY, max_yaw_rate=MAX_YAW_RATE,
                 nominal_z=NOMINAL_Z, ramp_rate=RAMP_RATE):
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_yaw_rate = max_yaw_rate
        self.ramp_rate = ramp_rate
        self.yaw_ramp_rate = YAW_RAMP_RATE
        self.nominal_z = nominal_z

        self.vx_target = 0.0
        self.vy_target = 0.0
        self.yaw_target = 0.0
        self.z_pos = nominal_z

        self.vx_cmd = 0.0
        self.vy_cmd = 0.0
        self.yaw_cmd = 0.0

        self._keys_pressed = set()
        self._lock = threading.Lock()
        self._running = False
        self._listener = None
        self._ramp_thread = None
        self._last_ramp_time = None
        self._e_stop = False

    def start(self):
        try:
            from pynput import keyboard
            self._keyboard = keyboard
        except ImportError:
            print("[Teleop] pynput not found. pip install pynput")
            return False

        self._running = True
        self._last_ramp_time = time.perf_counter()

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

        self._ramp_thread = threading.Thread(target=self._ramp_loop, daemon=True)
        self._ramp_thread.start()

        self._print_controls()
        return True

    def stop(self):
        self._running = False
        if self._listener is not None:
            self._listener.stop()

    def get_cmd(self):
        with self._lock:
            return self.vx_cmd, self.vy_cmd, self.z_pos, self.yaw_cmd

    def is_running(self):
        return self._running

    def _on_press(self, key):
        k = None

        # Character keys
        if hasattr(key, 'char') and key.char:
            try:
                k = key.char.lower()
            except AttributeError:
                pass

        # Special keys
        if key == self._keyboard.Key.space:
            self._emergency_stop()
            return
        if key == self._keyboard.Key.esc:
            self._running = False
            return
        if key == self._keyboard.Key.up:
            k = 'UP'
        elif key == self._keyboard.Key.down:
            k = 'DOWN'
        elif key == self._keyboard.Key.left:
            k = 'LEFT'
        elif key == self._keyboard.Key.right:
            k = 'RIGHT'

        if k is None:
            return

        with self._lock:
            self._keys_pressed.add(k)
            self._e_stop = False
            self._update_targets()

    def _on_release(self, key):
        k = None

        if hasattr(key, 'char') and key.char:
            try:
                k = key.char.lower()
            except AttributeError:
                pass

        if key == self._keyboard.Key.up:
            k = 'UP'
        elif key == self._keyboard.Key.down:
            k = 'DOWN'
        elif key == self._keyboard.Key.left:
            k = 'LEFT'
        elif key == self._keyboard.Key.right:
            k = 'RIGHT'

        if k is None:
            return

        with self._lock:
            self._keys_pressed.discard(k)
            self._update_targets()

    def _update_targets(self):
        keys = self._keys_pressed

        # Forward / backward (arrow keys)
        if 'UP' in keys and 'DOWN' not in keys:
            self.vx_target = self.max_vx
        elif 'DOWN' in keys and 'UP' not in keys:
            self.vx_target = -self.max_vx
        else:
            self.vx_target = 0.0

        # Strafe left / right (arrow keys)
        if 'LEFT' in keys and 'RIGHT' not in keys:
            self.vy_target = self.max_vy
        elif 'RIGHT' in keys and 'LEFT' not in keys:
            self.vy_target = -self.max_vy
        else:
            self.vy_target = 0.0

        # Turn left / right (Z/X)
        if 'z' in keys and 'x' not in keys:
            self.yaw_target = self.max_yaw_rate
        elif 'x' in keys and 'z' not in keys:
            self.yaw_target = -self.max_yaw_rate
        else:
            self.yaw_target = 0.0

        # Height adjust (C/V, single-shot)
        if 'c' in keys:
            self.z_pos = min(self.z_pos + Z_STEP, self.nominal_z + 0.10)
            self._keys_pressed.discard('c')
        if 'v' in keys:
            self.z_pos = max(self.z_pos - Z_STEP, self.nominal_z - 0.15)
            self._keys_pressed.discard('v')

    def _emergency_stop(self):
        with self._lock:
            self._e_stop = True
            self.vx_target = 0.0
            self.vy_target = 0.0
            self.yaw_target = 0.0
            self.vx_cmd = 0.0
            self.vy_cmd = 0.0
            self.yaw_cmd = 0.0
            self._keys_pressed.clear()
        print("\n[Teleop] EMERGENCY STOP")

    def _ramp_loop(self):
        while self._running:
            now = time.perf_counter()
            dt = now - self._last_ramp_time
            self._last_ramp_time = now

            with self._lock:
                if not self._e_stop:
                    self.vx_cmd = self._ramp_toward(self.vx_cmd, self.vx_target, self.ramp_rate * dt)
                    self.vy_cmd = self._ramp_toward(self.vy_cmd, self.vy_target, self.ramp_rate * dt)
                    self.yaw_cmd = self._ramp_toward(self.yaw_cmd, self.yaw_target, self.yaw_ramp_rate * dt)

            time.sleep(0.005)

    @staticmethod
    def _ramp_toward(current, target, max_step):
        diff = target - current
        if abs(diff) <= max_step:
            return target
        return current + np.sign(diff) * max_step

    @staticmethod
    def _print_controls():
        print("\n" + "=" * 50)
        print("  BHEEMA KEYBOARD TELEOP")
        print("=" * 50)
        print("  ↑ / ↓       Forward / Backward")
        print("  ← / →       Strafe Left / Right")
        print("  Z / X       Turn Left / Right")
        print("  C / V       Raise / Lower CoM")
        print("  SPACE       Emergency Stop")
        print("  ESC         Quit")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    print("Testing Teleop — press keys to see velocity commands")
    print("Press ESC to exit\n")

    teleop = Teleop()
    if not teleop.start():
        sys.exit(1)

    try:
        while teleop.is_running():
            vx, vy, z, yaw = teleop.get_cmd()
            print(f"\r  vx={vx:+.2f}  vy={vy:+.2f}  z={z:.3f}  yaw={yaw:+.2f}  ",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.stop()
        print("\nDone.")