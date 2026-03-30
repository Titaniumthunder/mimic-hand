"""
Watch the trained PPO hand policy in the MuJoCo viewer.

Run with any Python interpreter — the script bootstraps itself into the
MuJoCo native binary directly (bypassing mjpython's shebang, which breaks
when the project path contains spaces).

    python3 watch_hand.py
    python3 watch_hand.py checkpoints/hand_ppo_2000000_steps.zip
"""
import os
import sys
import time

# inside the while loop, after env.step():

def _bootstrap_mjpython():
    """
    Replicate what the mjpython script does, but call os.execve directly
    into the native MuJoCo binary so we never touch a shebang line.
    This makes spaces in any path harmless.
    """
    import ctypes
    import importlib.util
    import re
    import subprocess

    spec = importlib.util.find_spec('mujoco')
    if spec is None:
        return  # mujoco not installed — will fail later with a clear error
    module_dir = os.path.dirname(spec.origin)

    mjpython_bin = os.path.join(
        module_dir, 'MuJoCo_(mjpython).app', 'Contents', 'MacOS', 'mjpython'
    )
    if not os.path.isfile(mjpython_bin):
        return  # not macOS or non-standard install — try anyway

    # --- replicated from mjpython: set required env vars ---
    os.environ['MJPYTHON_BIN'] = mjpython_bin

    _NSGetExecutablePath = ctypes.CDLL(None)._NSGetExecutablePath
    c_size = ctypes.c_int32(0)
    _NSGetExecutablePath(None, ctypes.byref(c_size))
    c_path = (ctypes.c_char * c_size.value)()
    _NSGetExecutablePath(ctypes.byref(c_path), ctypes.byref(c_size))
    libpython_path = c_path.value.decode()
    os.environ['MJPYTHON_LIBPYTHON'] = libpython_path

    # Build DYLD_FALLBACK_LIBRARY_PATH (needed for @executable_path dylibs)
    libpython_dir = os.path.dirname(libpython_path)
    dyld_paths = []
    pattern = re.compile(r'@executable_path/(.+) \(offset \d+\)\Z')
    try:
        otool_out = subprocess.run(
            ['otool', '-l', libpython_path],
            capture_output=True, check=True
        ).stdout.decode()
        for line in otool_out.split('\n'):
            m = pattern.search(line)
            if m:
                resolved = os.path.dirname(os.path.join(libpython_dir, m.group(1)))
                if resolved not in dyld_paths:
                    dyld_paths.insert(0, resolved)
    except Exception:
        pass  # otool unavailable — skip, dyld will fall back on its own

    if dyld_paths:
        existing = os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')
        if existing:
            dyld_paths.extend(existing.split(':'))
        else:
            dyld_paths.extend(['/usr/local/lib', '/usr/lib'])
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join(dyld_paths)

    # argv[0] must be sys.executable (mjpython convention)
    argv = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
    os.execve(mjpython_bin, argv, os.environ)  # replaces this process entirely


# Only bootstrap once; MJPYTHON_BIN is set by the native binary on re-entry.
if 'MJPYTHON_BIN' not in os.environ:
    _bootstrap_mjpython()

# ── from here we are running inside the MuJoCo native binary ──────────────────
import time
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from grasp_env import HandGraspEnv

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "hand_sac.zip"
print(f"Loading policy: {checkpoint}")

env = HandGraspEnv()
model = SAC.load(checkpoint, env=env, device="cpu")

obs, _ = env.reset()
total_reward = 0.0
episode = 1

print(f"Starting episode {episode} — close the viewer window to exit.\n")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.distance = 0.6
    viewer.cam.elevation = -30
    viewer.cam.azimuth  = 180

    while viewer.is_running():
        step_start = time.time()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode {episode:3d} done | total reward: {total_reward:.2f}")
            episode += 1
            obs, _ = env.reset()
            total_reward = 0.0

        viewer.sync()

        elapsed = time.time() - step_start
        time.sleep(max(0, 1 / 30 - elapsed))
