# Dataset

Proprioceptive sensor data collected from Unitree Go2 quadruped 
in MuJoCo simulation across 4 terrain types.

## Terrain Types
- Hard floor: friction=1.0, solref=0.005 (stiff, high grip)
- Sand: friction=0.15, solref=0.005 (stiff, slippery)
- Mud: friction=0.8, solref=0.2 (soft, sticky)
- Soft ground: friction=0.5, solref=0.4 (very soft)

## Sensor Channels (23 total)
- Joint torques: 12 (one per leg joint)
- Body acceleration: 3 (x, y, z)
- Body height: 1
- Foot contact forces: 4
- Orientation: 3 (roll, pitch, yaw)

## Collection Method
Robot walks with IK-based trot gait for 2000 timesteps per terrain.
Logged every 10 simulation steps. Sliding windows of 50 timesteps
with stride 12 used for training.
