import mujoco
import math
import mujoco.viewer
import numpy as np
import csv

def set_terrain(model, terrain_type):
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    
    # give floor higher priority so its params dominate the contact
    model.geom_priority[floor_id] = 1   # feet default to 0
    
    if terrain_type == 'hard':
        model.geom_friction[floor_id] = [1.0, 0.005, 0.0001]
        model.geom_solref[floor_id] = [0.02, 1.0]
    elif terrain_type == 'sand':
        model.geom_friction[floor_id] = [0.15, 0.005, 0.0001]  # very slippery
        model.geom_solref[floor_id]   = [0.005, 1.0]            # stiff

    elif terrain_type == 'mud':
        model.geom_friction[floor_id] = [0.8, 0.005, 0.0001]  # sticky
        model.geom_solref[floor_id]   = [0.1, 1.0]             # moderately soft

    elif terrain_type == 'soft':
        model.geom_friction[floor_id] = [0.5, 0.005, 0.0001]  # medium
        model.geom_solref[floor_id]   = [0.4, 1.0]             # very soft


def leg_ik(foot_x,foot_z):

    up_leg = 0.213
    low_leg = 0.213
    dis_sq = foot_x**2 + foot_z**2
    alpha = math.acos((dis_sq-(up_leg**2)-(low_leg**2))/(-2*up_leg*low_leg))
    ca_an = -(math.pi - alpha)
    #th_an = (math.pi - ca_an)/2
    beta = math.asin(low_leg*math.sin(alpha)/math.sqrt(dis_sq))
    th_an = math.atan2(foot_x,-foot_z) + beta

    return th_an,ca_an

def get_trot_targets(t):
    A = 0.15
    freq = 0.5
    H = 0.28
    B = [0.07,0.07,0.1,0.1]
    target = np.zeros(12)
    phases = np.array([np.pi, 0, 0, np.pi])

    for i in range(4):
        angle = 2 * np.pi * freq * t + phases[i]
        if np.sin(angle) > 0:
            z = -H + B[i] * np.cos(angle)
        else:
            z = -H
        #mjpython IK-based_trot.py
        #z = -H + B[i] * np.cos(angle)
        if i == 0 or i == 1:
            x = -A * np.sin(angle)
        else:
            x = -A * np.sin(angle)

        thigh, calf = leg_ik(x, z)
        target[i*3]   = 0.0
        target[i*3+1] = thigh
        target[i*3+2] = calf

    return target

model = mujoco.MjModel.from_xml_path(
    '/Users/ethanlin/Desktop/mujoco-go2-terrain-classifier/mujoco_menagerie/unitree_go2/scene.xml'
)

set_terrain(model,'soft')
data = mujoco.MjData(model)

terrains = ['hard','sand','mud','soft']

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        targets = get_trot_targets(data.time)
        kp = 40.0
        kd = 5.0
        data.ctrl[:] = kp * (targets-data.qpos[7:19]) - kd*( data.qvel[6:18])

        mujoco.mj_step(model,data)
        viewer.sync()

