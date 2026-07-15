import mujoco
import math
import numpy as np
from IKbased_trot import get_trot_targets, set_terrain

def evaluate_gait(terrain, A, freq, B, steps=1000):
    # fresh model for each evaluation
    model = mujoco.MjModel.from_xml_path(
        'mujoco_menagerie/unitree_go2/scene.xml'
    )
    data = mujoco.MjData(model)
    set_terrain(model, terrain)
    
    # let robot settle for 200 steps first
    for _ in range(200):
        targets = get_trot_targets(data.time, A=A, freq=freq, B=B)
        kp, kd = 40.0, 5.0
        data.ctrl[:] = kp * (targets - data.qpos[7:19]) - kd * data.qvel[6:18]
        mujoco.mj_step(model, data)
    
    # now measure performance
    rolls, pitches, heights, torques = [], [], [], []
    yaws = []
    contact_vars = []
    x_start = data.qpos[0]
    
    for _ in range(steps):
        targets = get_trot_targets(data.time, A=A, freq=freq, B=B)
        kp, kd = 40.0, 5.0
        data.ctrl[:] = kp * (targets - data.qpos[7:19]) - kd * data.qvel[6:18]
        mujoco.mj_step(model, data)
        
        quat = data.qpos[3:7]
        roll  = 2 * math.atan2(quat[1], quat[0])
        pitch = 2 * math.atan2(quat[2], quat[0])
        
        rolls.append(roll**2)
        pitches.append(pitch**2)
        heights.append(data.qpos[2])
        torques.append(np.abs(data.qfrc_actuator[6:18]).sum())
         
        yaw = 2 * math.atan2(quat[3], quat[0])
        yaws.append(yaw**2)
        
        contact_forces = np.zeros(4)
        for i in range(min(data.ncon, 4)):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            contact_forces[i] = force[0]
        contact_vars.append(np.std(contact_forces))
        
    yaw_drift = np.mean(yaws)
    contact_variance = np.mean(contact_vars)
    height_drop = 0.28 - np.mean(heights)  # how much below normal height
    x_end = data.qpos[0]

    # compute metrics
    stability   = math.exp(-np.mean(rolls) - np.mean(pitches))
    distance    = abs(x_end - x_start)
    height_std  = np.std(heights)
    avg_torque  = np.mean(torques)

    # terrain specific scoring
    if terrain == 'hard':
        score = 0.4*stability + 0.4*min(distance,1.0) + 0.2*math.exp(-height_std*10)
    
    elif terrain == 'sand':
        # penalize yaw drift heavily — slipping causes turning
        score = 0.3*stability + 0.3*min(distance,1.0) + 0.4*math.exp(-yaw_drift*10)
    
    elif terrain == 'mud':
        # penalize contact variance — sticky ground causes irregular forces
        score = 0.3*stability + 0.3*min(distance,1.0) + 0.4*math.exp(-contact_variance*0.1)
    
    elif terrain == 'soft':
        # penalize height drop — soft ground makes robot sink
        score = 0.3*stability + 0.3*min(distance,1.0) + 0.4*math.exp(-height_drop*10)
    
    return score, stability, distance, height_std, avg_torque, yaw_drift, contact_variance

# parameter options to test
A_options    = [0.08, 0.12, 0.15]
freq_options = [0.6, 0.9, 1.2]
B_options    = [[0.05, 0.05, 0.07, 0.07],
                [0.07, 0.07, 0.10, 0.10],
                [0.10, 0.10, 0.14, 0.14]]

terrains = ['hard', 'sand', 'mud', 'soft']

best_params = {}

for terrain in terrains:
    best_score = -999
    best = None
    
    for A in A_options:
        for freq in freq_options:
            for B in B_options:
                score, stability, distance, height_std, avg_torque, yaw_drift, contact_variance = evaluate_gait(
                    terrain, A, freq, B
                )
                print(f"{terrain} | A={A} freq={freq} B={B[0]} | "
                      f"score={score:.3f} stab={stability:.3f} "
                      f"dist={distance:.3f} hstd={height_std:.4f} "
                      f"torque={avg_torque:.1f}")
                
                if score > best_score:
                    best_score = score
                    best = {'A': A, 'freq': freq, 'B': B}
    
    best_params[terrain] = best
    print(f"\nBest for {terrain}: {best} score={best_score:.3f}\n")

print("\nFinal best params per terrain:")
for terrain, params in best_params.items():
    print(f"{terrain}: {params}")
