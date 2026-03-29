import mujoco
import math

xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>

    <body name="palm" pos="0 0 0.1">
      <geom type="box" size=".04 .05 .015" rgba=".8 .6 .4 1"/>

      <!-- Index finger -->
      <body name="index_knuckle" pos="0.02 0.05 0.01">
        <joint name="j0" type="hinge" axis="0 1 0" range="0 90"/>
        <geom type="box" size=".012 .012 .035" rgba="1 0.5 0 1"/>
        <body name="index_middle" pos="0 0 0.07">
          <joint name="j1" type="hinge" axis="0 1 0" range="0 90"/>
          <geom type="box" size=".011 .011 .03" rgba="1 0.65 0 1"/>
          <body name="index_tip" pos="0 0 0.06">
            <joint name="j2" type="hinge" axis="0 1 0" range="0 90"/>
            <geom type="box" size=".01 .01 .025" rgba="1 0.8 0 1"/>
          </body>
        </body>
      </body>

      <!-- Middle finger -->
      <body name="middle_knuckle" pos="0 0.05 0.01">
        <joint name="j3" type="hinge" axis="0 1 0" range="0 90"/>
        <geom type="box" size=".012 .012 .04" rgba="0.5 0.8 1 1"/>
        <body name="middle_middle" pos="0 0 0.08">
          <joint name="j4" type="hinge" axis="0 1 0" range="0 90"/>
          <geom type="box" size=".011 .011 .033" rgba="0.5 0.9 1 1"/>
          <body name="middle_tip" pos="0 0 0.066">
            <joint name="j5" type="hinge" axis="0 1 0" range="0 90"/>
            <geom type="box" size=".01 .01 .027" rgba="0.6 1 1 1"/>
          </body>
        </body>
      </body>

      <!-- Ring finger -->
      <body name="ring_knuckle" pos="-0.02 0.05 0.01">
        <joint name="j6" type="hinge" axis="0 1 0" range="0 90"/>
        <geom type="box" size=".012 .012 .035" rgba="0.5 1 0.5 1"/>
        <body name="ring_middle" pos="0 0 0.07">
          <joint name="j7" type="hinge" axis="0 1 0" range="0 90"/>
          <geom type="box" size=".011 .011 .03" rgba="0.6 1 0.6 1"/>
          <body name="ring_tip" pos="0 0 0.06">
            <joint name="j8" type="hinge" axis="0 1 0" range="0 90"/>
            <geom type="box" size=".01 .01 .025" rgba="0.7 1 0.7 1"/>
          </body>
        </body>
      </body>

      <!-- Pinky -->
      <body name="pinky_knuckle" pos="-0.035 0.048 0.01">
        <joint name="j9" type="hinge" axis="0 1 0" range="0 90"/>
        <geom type="box" size=".01 .01 .028" rgba="0.8 0.5 1 1"/>
        <body name="pinky_middle" pos="0 0 0.056">
          <joint name="j10" type="hinge" axis="0 1 0" range="0 90"/>
          <geom type="box" size=".009 .009 .024" rgba="0.85 0.6 1 1"/>
          <body name="pinky_tip" pos="0 0 0.048">
            <joint name="j11" type="hinge" axis="0 1 0" range="0 90"/>
            <geom type="box" size=".008 .008 .02" rgba="0.9 0.7 1 1"/>
          </body>
        </body>
      </body>

      <!-- Thumb -->
      <body name="thumb_knuckle" pos="0.042 0.01 0.01">
        <joint name="j12" type="hinge" axis="1 0 0" range="0 70"/>
        <geom type="box" size=".015 .03 .015" rgba="1 0.4 0.4 1"/>
        <body name="thumb_middle" pos="0 0.06 0">
          <joint name="j13" type="hinge" axis="1 0 0" range="0 60"/>
          <geom type="box" size=".013 .025 .013" rgba="1 0.5 0.5 1"/>
          <body name="thumb_tip" pos="0 0.05 0">
            <joint name="j14" type="hinge" axis="1 0 0" range="0 50"/>
            <geom type="box" size=".011 .02 .011" rgba="1 0.6 0.6 1"/>
          </body>
        </body>
      </body>

    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)

# Curl all fingers into a fist
for i in range(15):
    data.qpos[i] = math.radians(60)

mujoco.mj_forward(model, data)

print(f"Hand has {model.njnt} joints")
print(f"Observation size: {model.nq} numbers")
print("Hand working!")