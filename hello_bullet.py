import mujoco

xml = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)

for _ in range(200):
    mujoco.mj_step(model, data)

print(f"Box Z position: {data.qpos[2]:.4f}")
print("MuJoCo is working!")