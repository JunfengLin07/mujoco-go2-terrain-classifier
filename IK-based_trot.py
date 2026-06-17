import mujoco
import math
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    '/Users/ethanlin/Desktop/mujoco-go2-terrain-classifier/mujoco_menagerie/unitree_go2/scene.xml'
)
data = mujoco.MjData(model)

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

def get_trot_targets(t,yaw_correction=0):
    A = 0.15
    freq = 1.2
    H = 0.28
    B = 0.04
    target = np.zeros(12)
    
    phases = np.array([0,np.pi,np.pi,0])
    

    yaw = data.qpos[5]
    #correction = -0.05 * yaw
    #hip_offsets = [correction,correction,correction,correction]

    for i in range (4):
        phase = phases[i]
        angle = 2*np.pi*freq*t
        x = - A * np.sin(angle+phase)
        z = - H + B*np.cos(angle+phase)

        thigh,calf = leg_ik(x,z)

        target[i*3] = 0
        target[i*3+1] = thigh
        target[i*3+2] = calf
    
    return target


with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():

        if int(data.time*500) % 100 == 0:
            print(f"\nTime: {data.time:.2f}s")
            print(f"Joint torques: {data.qfrc_actuator[6:18].round(2)}")
            print(f"Body accel XYZ: {data.qacc[:3].round(3)}")
            print(f"Body height: {data.qpos[2]:.3f}")
            
            quat = data.qpos[3:7]
            roll  = 2 * math.atan2(quat[1], quat[0])
            pitch = 2 * math.atan2(quat[2], quat[0])
            yaw   = 2 * math.atan2(quat[3], quat[0])
            print(f"Roll:{roll:.3f} Pitch:{pitch:.3f} Yaw:{yaw:.3f}")

        targets = get_trot_targets(data.time)
        kp=40.0
        kd = 5.0
        data.ctrl[:] = kp * (targets-data.qpos[7:19]) - kd*( data.qvel[6:18])
        mujoco.mj_step(model,data)
        viewer.sync()