"""
Bheema Keyboard Teleop
Controls:
    W / S       Forward / Backward
    A / D       Strafe Left / Right
    Q / E       Turn Left / Turn Right
    Z / X       Raise / Lower CoM height
    SPACE       Emergency stop (zero all velocities)
    ESC         Quit

Usage in main.py:
    from bheema.teleop import Teleop

    teleop = Teleop()
    teleop.start()

    # In the control loop, replace get_body_cmd():
    x_vel, y_vel, z_pos, yaw_rate = teleop.get_cmd()

    # After simulation:
    teleop.stop()
"""

import numpy as np
import threading
import time
import sys


# ==============================================================================
# Configuration
# ==============================================================================

MAX_VX = 1.0        # m/s forward/backward
MAX_VY = 0.4        # m/s lateral
MAX_YAW_RATE = 0.6  # rad/s turning
NOMINAL_Z = 0.66    # m default CoM height
Z_STEP = 0.02       # m per keypress

# How quickly velocity ramps up/down (prevents jerky commands)
RAMP_RATE = 2.0     # m/s per second (for linear velocities)
YAW_RAMP_RATE = 1.5 # rad/s per second (for yaw rate)


class Teleop:
    def __init__(self,
                 max_vx=MAX_VX,
                 max_vy=MAX_VY,
                 max_yaw_rate=MAX_YAW_RATE,
                 nominal_z=NOMINAL_Z,
                 ramp_rate=RAMP_RATE):

        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_yaw_rate = max_yaw_rate
        self.ramp_rate = ramp_rate
        self.yaw_ramp_rate = YAW_RAMP_RATE

        # Target velocities (what the keys request)
        self.vx_target = 0.0
        self.vy_target = 0.0
        self.yaw_target = 0.0
        self.z_pos = nominal_z
        self.nominal_z = nominal_z

        # Smoothed velocities (what we actually send to the robot)
        self.vx_cmd = 0.0
        self.vy_cmd = 0.0
        self.yaw_cmd = 0.0

        # Key state tracking
        self._keys_pressed = set()
        self._lock = threading.Lock()
        self._running = False
        self._listener = None
        self._ramp_thread = None
        self._last_ramp_time = None
        self._e_stop = False

    def start(self):
        """Start the keyboard listener and velocity ramping thread."""
        try:
            from pynput import keyboard
            self._keyboard = keyboard
        except ImportError:
            print("[Teleop] pynput not found. Install with: pip install pynput")
            print("[Teleop] Falling back to static commands.")
            return False

        self._running = True
        self._last_ramp_time = time.perf_counter()

        # Keyboard listener (runs in its own thread)
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

        # Velocity ramping thread (smooth acceleration)
        self._ramp_thread = threading.Thread(target=self._ramp_loop, daemon=True)
        self._ramp_thread.start()

        self._print_controls()
        return True

    def stop(self):
        """Stop the keyboard listener."""
        self._running = False
        if self._listener is not None:
            self._listener.stop()

    def get_cmd(self):
        """
        Returns (x_vel_body, y_vel_body, z_pos, yaw_rate_body).
        Drop-in replacement for get_body_cmd() in main.py.
        """
        with self._lock:
            return self.vx_cmd, self.vy_cmd, self.z_pos, self.yaw_cmd

    def is_running(self):
        """Returns False if ESC was pressed."""
        return self._running

    # --------------------------------------------------------------------------
    # Internal
    # --------------------------------------------------------------------------

    def _on_press(self, key):
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            k = None

        # Handle special keys
        if key == self._keyboard.Key.space:
            self._emergency_stop()
            return
        if key == self._keyboard.Key.esc:
            self._running = False
            return

        if k is None:
            return

        with self._lock:
            self._keys_pressed.add(k)
            self._e_stop = False
            self._update_targets()

    def _on_release(self, key):
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            k = None

        if k is None:
            return

        with self._lock:
            self._keys_pressed.discard(k)
            self._update_targets()

    def _update_targets(self):
        """Map currently pressed keys to target velocities."""
        keys = self._keys_pressed

        # Forward / backward
        if 'w' in keys and 's' not in keys:
            self.vx_target = self.max_vx
        elif 's' in keys and 'w' not in keys:
            self.vx_target = -self.max_vx
        else:
            self.vx_target = 0.0

        # Strafe left / right
        if 'a' in keys and 'd' not in keys:
            self.vy_target = self.max_vy
        elif 'd' in keys and 'a' not in keys:
            self.vy_target = -self.max_vy
        else:
            self.vy_target = 0.0

        # Turn left / right
        if 'q' in keys and 'e' not in keys:
            self.yaw_target = self.max_yaw_rate
        elif 'e' in keys and 'q' not in keys:
            self.yaw_target = -self.max_yaw_rate
        else:
            self.yaw_target = 0.0

        # Height adjust (instant, not ramped)
        if 'z' in keys:
            self.z_pos = min(self.z_pos + Z_STEP, self.nominal_z + 0.10)
            self._keys_pressed.discard('z')  # Single-shot
        if 'x' in keys:
            self.z_pos = max(self.z_pos - Z_STEP, self.nominal_z - 0.15)
            self._keys_pressed.discard('x')  # Single-shot

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
        """Smoothly ramp commanded velocities toward targets."""
        while self._running:
            now = time.perf_counter()
            dt = now - self._last_ramp_time
            self._last_ramp_time = now

            with self._lock:
                if not self._e_stop:
                    self.vx_cmd = self._ramp_toward(
                        self.vx_cmd, self.vx_target, self.ramp_rate * dt)
                    self.vy_cmd = self._ramp_toward(
                        self.vy_cmd, self.vy_target, self.ramp_rate * dt)
                    self.yaw_cmd = self._ramp_toward(
                        self.yaw_cmd, self.yaw_target, self.yaw_ramp_rate * dt)

            time.sleep(0.005)  # 200 Hz update

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
        print("  W/S     Forward / Backward")
        print("  A/D     Strafe Left / Right")
        print("  Q/E     Turn Left / Right")
        print("  Z/X     Raise / Lower CoM")
        print("  SPACE   Emergency Stop")
        print("  ESC     Quit")
        print("=" * 50 + "\n")


# ==============================================================================
# Gamepad support (optional, for PS4/Xbox controllers)
# ==============================================================================

class GamepadTeleop:
    """
    Gamepad teleop using pygame.
    Left stick  = forward/lateral velocity
    Right stick = yaw rate (horizontal axis)
    L1/R1       = lower/raise CoM
    
    Usage:
        teleop = GamepadTeleop()
        if teleop.start():
            x_vel, y_vel, z_pos, yaw_rate = teleop.get_cmd()
    """

    def __init__(self,
                 max_vx=MAX_VX,
                 max_vy=MAX_VY,
                 max_yaw_rate=MAX_YAW_RATE,
                 nominal_z=NOMINAL_Z,
                 deadzone=0.1):

        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_yaw_rate = max_yaw_rate
        self.nominal_z = nominal_z
        self.z_pos = nominal_z
        self.deadzone = deadzone

        self.vx_cmd = 0.0
        self.vy_cmd = 0.0
        self.yaw_cmd = 0.0

        self._running = False
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        try:
            import pygame
            self._pygame = pygame
        except ImportError:
            print("[GamepadTeleop] pygame not found. Install with: pip install pygame")
            return False

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[GamepadTeleop] No gamepad detected.")
            return False

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        print(f"[GamepadTeleop] Connected: {self._joystick.get_name()}")

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if hasattr(self, '_pygame'):
            self._pygame.quit()

    def get_cmd(self):
        with self._lock:
            return self.vx_cmd, self.vy_cmd, self.z_pos, self.yaw_cmd

    def is_running(self):
        return self._running

    def _apply_deadzone(self, val):
        if abs(val) < self.deadzone:
            return 0.0
        # Rescale so output is 0 at deadzone edge, 1 at full tilt
        sign = np.sign(val)
        return sign * (abs(val) - self.deadzone) / (1.0 - self.deadzone)

    def _poll_loop(self):
        while self._running:
            self._pygame.event.pump()

            # Left stick: axis 0 = lateral, axis 1 = forward (inverted)
            raw_vx = -self._joystick.get_axis(1)  # Forward is negative on most gamepads
            raw_vy = -self._joystick.get_axis(0)   # Left is negative

            # Right stick: axis 2 or 3 = yaw
            raw_yaw = -self._joystick.get_axis(2)  # Left turn positive

            vx = self._apply_deadzone(raw_vx) * self.max_vx
            vy = self._apply_deadzone(raw_vy) * self.max_vy
            yaw = self._apply_deadzone(raw_yaw) * self.max_yaw_rate

            # Shoulder buttons for height (button indices may vary by controller)
            try:
                if self._joystick.get_button(4):  # L1
                    self.z_pos = max(self.z_pos - 0.001, self.nominal_z - 0.15)
                if self._joystick.get_button(5):  # R1
                    self.z_pos = min(self.z_pos + 0.001, self.nominal_z + 0.10)
            except Exception:
                pass

            with self._lock:
                self.vx_cmd = vx
                self.vy_cmd = vy
                self.yaw_cmd = yaw

            time.sleep(0.01)  # 100 Hz polling


# ==============================================================================
# Standalone test
# ==============================================================================

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