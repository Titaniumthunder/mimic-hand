import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

# Geometry design:
#   Palm at (0, 0, 0.5) rotated 180° around X → local +Z maps to world -Z (down).
#   Finger knuckles at local (x, 0.05, 0.01) → world (x, -0.05, 0.49); segments grow in local +Z → world -Z.
#   Thumb knuckle at local (0.04, 0.04, 0.04) → world (0.04, -0.04, 0.46); also extends in local +Z → world -Z.
#   Ball spawned inside the finger space so tips surround it on 3 sides at reset.
#
# Joint axis reasoning (palm frame: local X=world X, local Y=world -Y, local Z=world -Z):
#   4 main fingers: axis="1 0 0" (world X) → positive angle rotates local +Z toward world +Y
#                   → tips curl from y=-0.05 toward ball at y=-0.044 (correct closing direction).
#   Thumb:          axis="0 -1 0" (world +Y) → positive angle rotates local +Z toward world -X
#                   → thumb tip curls from x=+0.04 toward ball at x=0 (correct opposition).
#
# Fingertip world positions at zero joint angles:
#   index  ( 0.020, -0.050, 0.360)
#   middle ( 0.000, -0.050, 0.344)
#   ring   (-0.020, -0.050, 0.360)
#   pinky  (-0.035, -0.048, 0.386)
#   thumb  ( 0.040, -0.040, 0.370)
#   ball   ( 0.000, -0.044, 0.360)  ← inside finger space; fingertips surround on 3 sides

XML = """
<mujoco>
  <option gravity="0 0 -2" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>

    <!-- Ball spawned inside finger space: tips surround it on -X, +X, and -Y sides at rest -->
    <body name="object" pos="0 -0.044 0.36">
      <freejoint name="object_joint"/>
      <geom name="object_geom" type="sphere" size="0.025" rgba="1 0 0 1" mass="0.05"/>
    </body>

    <!-- Palm rotated 180 deg around X so local +Z points world -Z (fingers hang down) -->
    <body name="palm" pos="0 0 0.5" euler="180 0 0">
      <geom type="box" size=".04 .05 .015" rgba=".8 .6 .4 1"/>

      <!-- Index finger: axis="1 0 0" → tips close in world +Y (toward ball) -->
      <body name="index_knuckle" pos="0.02 0.05 0.01">
        <joint name="j0" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
        <geom type="box" size=".012 .012 .035" rgba="1 0.5 0 1"/>
        <body name="index_middle" pos="0 0 0.07">
          <joint name="j1" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
          <geom type="box" size=".011 .011 .03" rgba="1 0.65 0 1"/>
          <body name="index_tip" pos="0 0 0.06">
            <joint name="j2" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
            <geom type="box" size=".01 .01 .025" rgba="1 0.8 0 1"/>
          </body>
        </body>
      </body>

      <!-- Middle finger -->
      <body name="middle_knuckle" pos="0 0.05 0.01">
        <joint name="j3" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
        <geom type="box" size=".012 .012 .04" rgba="0.5 0.8 1 1"/>
        <body name="middle_middle" pos="0 0 0.08">
          <joint name="j4" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
          <geom type="box" size=".011 .011 .033" rgba="0.5 0.9 1 1"/>
          <body name="middle_tip" pos="0 0 0.066">
            <joint name="j5" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
            <geom type="box" size=".01 .01 .027" rgba="0.6 1 1 1"/>
          </body>
        </body>
      </body>

      <!-- Ring finger -->
      <body name="ring_knuckle" pos="-0.02 0.05 0.01">
        <joint name="j6" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
        <geom type="box" size=".012 .012 .035" rgba="0.5 1 0.5 1"/>
        <body name="ring_middle" pos="0 0 0.07">
          <joint name="j7" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
          <geom type="box" size=".011 .011 .03" rgba="0.6 1 0.6 1"/>
          <body name="ring_tip" pos="0 0 0.06">
            <joint name="j8" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
            <geom type="box" size=".01 .01 .025" rgba="0.7 1 0.7 1"/>
          </body>
        </body>
      </body>

      <!-- Pinky finger -->
      <body name="pinky_knuckle" pos="-0.035 0.048 0.01">
        <joint name="j9" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
        <geom type="box" size=".01 .01 .028" rgba="0.8 0.5 1 1"/>
        <body name="pinky_middle" pos="0 0 0.056">
          <joint name="j10" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
          <geom type="box" size=".009 .009 .024" rgba="0.85 0.6 1 1"/>
          <body name="pinky_tip" pos="0 0 0.048">
            <joint name="j11" type="hinge" axis="1 0 0" range="0 90" damping="0.5" armature="0.01"/>
            <geom type="box" size=".008 .008 .02" rgba="0.9 0.7 1 1"/>
          </body>
        </body>
      </body>

      <!-- Thumb: axis="0 -1 0" (world +Y) → positive angle curls tip toward world -X (toward ball) -->
      <body name="thumb_knuckle" pos="0.04 0.04 0.04">
        <joint name="j12" type="hinge" axis="0 -1 0" range="0 70" damping="0.5" armature="0.01"/>
        <geom type="box" size=".015 .012 .025" rgba="1 0.4 0.4 1"/>
        <body name="thumb_middle" pos="0 0 0.05">
          <joint name="j13" type="hinge" axis="0 -1 0" range="0 60" damping="0.5" armature="0.01"/>
          <geom type="box" size=".013 .011 .022" rgba="1 0.5 0.5 1"/>
          <body name="thumb_tip" pos="0 0 0.04">
            <joint name="j14" type="hinge" axis="0 -1 0" range="0 50" damping="0.5" armature="0.01"/>
            <geom type="box" size=".011 .01 .018" rgba="1 0.6 0.6 1"/>
          </body>
        </body>
      </body>

    </body>
  </worldbody>
  <actuator>
    <position name="a0"  joint="j0"  kp="5"/>
    <position name="a1"  joint="j1"  kp="5"/>
    <position name="a2"  joint="j2"  kp="5"/>
    <position name="a3"  joint="j3"  kp="5"/>
    <position name="a4"  joint="j4"  kp="5"/>
    <position name="a5"  joint="j5"  kp="5"/>
    <position name="a6"  joint="j6"  kp="5"/>
    <position name="a7"  joint="j7"  kp="5"/>
    <position name="a8"  joint="j8"  kp="5"/>
    <position name="a9"  joint="j9"  kp="5"/>
    <position name="a10" joint="j10" kp="5"/>
    <position name="a11" joint="j11" kp="5"/>
    <position name="a12" joint="j12" kp="5"/>
    <position name="a13" joint="j13" kp="5"/>
    <position name="a14" joint="j14" kp="5"/>
  </actuator>
</mujoco>
"""

class HandGraspEnv(gym.Env):
    def __init__(self, verbose=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data  = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(33,), dtype=np.float32)
        self.action_space      = spaces.Box(-1, 1, shape=(15,), dtype=np.float32)

        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.ball_geom_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.tip_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["index_tip", "middle_tip", "ring_tip", "pinky_tip", "thumb_tip"]
        ]
        self.tip_names = ["index_tip", "middle_tip", "ring_tip", "pinky_tip", "thumb_tip"]

        self.max_steps  = 500
        self.step_count = 0
        self.verbose    = verbose

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        # Spawn ball inside the finger space: x jitter keeps training varied,
        # y stays near fingertip level (-0.05) so tips surround on 3 sides,
        # z at fingertip height so fingers aren't reaching above the ball.
        # Freejoint qpos layout: [x, y, z, qw, qx, qy, qz] at indices 15-21.
        self.data.qpos[15] = np.random.uniform(-0.015, 0.015)   # x
        self.data.qpos[16] = np.random.uniform(-0.048, -0.040)  # y  (at fingertip y=-0.05)
        self.data.qpos[17] = 0.36                               # z  (at fingertip height)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        if self.verbose:
            self._verify_geometry()
        return self._get_obs(), {}

    def _verify_geometry(self):
        """Print fingertip positions relative to ball and confirm 3-side coverage."""
        ball = self.data.xpos[self.object_body_id]
        print(f"  ball   x={ball[0]:+.3f}  y={ball[1]:+.3f}  z={ball[2]:+.3f}")
        left = right = below = above = 0
        for tip_id, name in zip(self.tip_ids, self.tip_names):
            tip  = self.data.xpos[tip_id]
            dist = np.linalg.norm(tip - ball)
            print(f"  {name:12s}  x={tip[0]:+.3f}  y={tip[1]:+.3f}  z={tip[2]:+.3f}"
                  f"  dist={dist*100:.1f}cm")
            if tip[0] < ball[0] - 0.005: left  += 1
            if tip[0] > ball[0] + 0.005: right += 1
            if tip[1] < ball[1] - 0.005: below += 1
            if tip[1] > ball[1] + 0.005: above += 1
        sides = sum([left > 0, right > 0, below > 0, above > 0])
        status = "OK" if sides >= 3 else "FAIL"
        print(f"  coverage: {sides}/4 sides (L={left} R={right} -Y={below} +Y={above}) [{status}]")

    def step(self, action):
        target_angles = (action + 1) / 2 * math.radians(90)
        self.data.ctrl[:] = target_angles
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        obs        = self._get_obs()
        reward     = self._get_reward()
        terminated = self._is_done()
        truncated  = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        joint_angles  = self.data.qpos[:15].astype(np.float32)
        object_pos    = self.data.xpos[self.object_body_id].astype(np.float32)
        tip_positions = np.array(
            [self.data.xpos[tid] for tid in self.tip_ids], dtype=np.float32
        ).flatten()  # 5 tips × 3 = 15 values
        return np.concatenate([joint_angles, object_pos, tip_positions])

    def _count_ball_contacts(self):
        """Count contacts involving the ball geom (physical touch, not proximity)."""
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.ball_geom_id or c.geom2 == self.ball_geom_id:
                n += 1
        return n

    def _get_reward(self):
        object_pos = self.data.xpos[self.object_body_id]
        n_contacts = self._count_ball_contacts()
        reward     = 0.0

        lift_height = object_pos[2] - 0.36

        # v3: reward only when ball is physically elevated with 3+ contacts.
        # No shaping, no curl reward — pure lift signal.
        if n_contacts >= 3 and lift_height > 0.0:
            reward += lift_height * 50.0

        return reward

    def _is_done(self):
        object_z = self.data.xpos[self.object_body_id][2]
        # Ball flung too high or dropped to floor
        return object_z > 0.56 or object_z < 0.10

if __name__ == "__main__":
    env = HandGraspEnv(verbose=True)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
    print(f"\nReward after random actions: {reward:.4f}")
    print("Environment working!")

    try:
        import mujoco.viewer
        print("\nLaunching MuJoCo viewer — close the window to exit.")
        env2 = HandGraspEnv(verbose=True)
        obs, _ = env2.reset()
        with mujoco.viewer.launch_passive(env2.model, env2.data) as viewer:
            viewer.cam.distance = 0.8
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 45
            while viewer.is_running():
                mujoco.mj_step(env2.model, env2.data)
                viewer.sync()
    except Exception as e:
        print(f"Viewer not available in this environment: {e}")
