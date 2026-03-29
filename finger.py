import mujoco

xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>

    <body name="knuckle" pos="0 0 0.3">
      <joint name="j0" type="hinge" axis="0 1 0" range="0 90"/>
      <geom type="box" size=".015 .015 .04" rgba="1 0.5 0 1"/>

      <body name="middle" pos="0 0 0.08">
        <joint name="j1" type="hinge" axis="0 1 0" range="0 90"/>
        <geom type="box" size=".013 .013 .035" rgba="1 0.7 0 1"/>

        <body name="tip" pos="0 0 0.07">
          <joint name="j2" type="hinge" axis="0 1 0" range="0 90"/>
          <geom type="box" size=".011 .011 .03" rgba="1 0.9 0 1"/>
        </body>

      </body>
    </body>

  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)

# Curl the finger by setting joint angles (in radians)
import math
data.qpos[0] = math.radians(45)  # knuckle — halfway curled
data.qpos[1] = math.radians(60)  # middle
data.qpos[2] = math.radians(80)  # tip — almost fully curled

mujoco.mj_forward(model, data)  # apply the positions

# Print where the tip ended up
tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
print(f"Tip position: {data.xpos[tip_id]}")
print("Finger working!")