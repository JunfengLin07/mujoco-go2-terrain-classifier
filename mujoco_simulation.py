import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    '/Users/ethanlin/Desktop/mujoco_simulation/mujoco_menagerie/unitree_go2/scene.xml'
)
data = mujoco.MjData(model)

freq = 0.1

#FL=0 FR=pi RL=pi RR=0
phases = np.array([0,np.pi,np.pi,0])

def get_trot_targets(t):
    targets = np.zeros(12)
    
    for i in range(4):  # 4 legs
        phase = phases[i]
        swing = np.sin(2 * np.pi * freq * t + phase)
        
        hip   = i * 3      # index for this leg's hip
        thigh = i * 3 + 1  # index for this leg's thigh
        calf  = i * 3 + 2  # index for this leg's calf
        
        targets[hip]   = 0.0
        targets[thigh] = 0.9 + 0.3 * swing   # oscillates between 0.6 and 1.2
        targets[calf]  = -1.8 - 0.3 * swing  # opposite to thigh
    
    return targets

# Standing pose joint angles (radians)
# For each leg: hip=0, thigh=0.9, calf=-1.8
standing_pos = np.array([
    0.0,  0.9, -1.8,   # FL: hip, thigh, calf
    0.0,  0.9, -1.8,   # FR: hip, thigh, calf
    0.0,  0.9, -1.8,   # RL: hip, thigh, calf
    0.0,  0.9, -1.8,   # RR: hip, thigh, calf
])

# Set initial joint positions directly
data.qpos[7:19] = standing_pos

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        
        targets = standing_pos
        #get_trot_targets(data.time)

        # PD controller — push joints toward standing position
        kp = 40.0  # position gain
        kd = 2.0   # velocity gain
        
        current_pos = data.qpos[7:19]
        current_vel = data.qvel[6:18]
        
        data.ctrl[:] = kp * (targets - current_pos) - kd * current_vel
        
        mujoco.mj_step(model, data)
        viewer.sync()
