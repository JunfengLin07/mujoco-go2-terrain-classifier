import classifier
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import numpy as np
import math
from IKbased_trot import get_trot_targets,set_terrain

net = classifier.classifier()
net.load_state_dict(torch.load('terrain_classifier.pth'))
net.eval()

model = mujoco.MjModel.from_xml_path(
    'mujoco_menagerie/unitree_go2/scene.xml'
)
data = mujoco.MjData(model)

sensor_buffer = []
step = 0
terrain_names = {0: 'hard', 1: 'soft', 2: 'mud', 3: 'sand'}
current_terrain = 'hard'
set_terrain(model,'sand')

gait_params = {
    'hard': {'A': 0.15, 'freq': 1.2, 'B': [0.07, 0.07, 0.10, 0.10]},
    'sand': {'A': 0.08, 'freq': 0.6, 'B': [0.05, 0.05, 0.07, 0.07]},
    'mud':  {'A': 0.08, 'freq': 0.6, 'B': [0.05, 0.05, 0.07, 0.07]},
    'soft': {'A': 0.08, 'freq': 0.6, 'B': [0.10, 0.10, 0.14, 0.14]},
}
checking = True
step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # collect sensor every 10 steps
        #if int(data.time*500) >= 0:
            #set_terrain(model,'soft')
        
        if checking:
            if (int(data.time * 500) % 10 == 0):
                quat = data.qpos[3:7]
                roll  = 2 * math.atan2(quat[1], quat[0])
                pitch = 2 * math.atan2(quat[2], quat[0])
                yaw   = 2 * math.atan2(quat[3], quat[0])

                contact_forces = np.zeros(4)
                for i in range(min(data.ncon, 4)):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, force)
                    contact_forces[i] = force[0]

                reading = []
                reading += data.qfrc_actuator[6:18].tolist()
                reading += data.qacc[:3].tolist()
                reading += [data.qpos[2]]
                reading += contact_forces.tolist()
                reading += [roll, pitch, yaw]
                
                sensor_buffer.append(reading)
                
                # classify when buffer is full
                if len(sensor_buffer) == 50:
                    window = np.array(sensor_buffer)
                    # normalize
                    """
                    mean = window.mean(axis=0, keepdims=True)
                    std  = window.std(axis=0, keepdims=True) + 1e-8
                    window = (window - mean) / std
                    """
                    window = window.T
                    window = torch.FloatTensor(window).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = net(window)
                        predicted = output.argmax(1).item()
                    
                    current_terrain = terrain_names[predicted]
                    #print(f"Time: {data.time:.1f}s | Terrain: {current_terrain}")
                    sensor_buffer.pop(0)
                    step += 1
                    if(step == 100):
                        checking = False
                        step = 0

            targets = get_trot_targets(data.time,A = 0.15,freq = 0.5)
            kp = 40.0
            kd = 5.0
            data.ctrl[:] = kp * (targets-data.qpos[7:19]) - kd*( data.qvel[6:18])

        else:
            targets = get_trot_targets(data.time, A = gait_params[f'{current_terrain}']['A'],freq = gait_params[f'{current_terrain}']['freq'], B = gait_params[f'{current_terrain}']['B'])
            kp = 40.0
            kd = 5.0
            data.ctrl[:] = kp * (targets-data.qpos[7:19]) - kd*( data.qvel[6:18])
            if (int(data.time * 500) % 10 == 0):
                step += 1
            if(step == 1000):
                sensor_buffer = []
                checking = True

        mujoco.mj_step(model, data)
        viewer.sync()
