# MuJoCo Go2 Terrain Classifier

Proprioceptive terrain sensing and classification system for the 
Unitree Go2 quadruped robot in MuJoCo simulation.

## Overview
The robot walks on 4 different terrain types and classifies them 
purely from proprioceptive signals — no camera or external sensors.
Inspired by RoboLAND lab's work on terrain-aware locomotion.

## Results

| Terrain | Validation Accuracy |
|---------|-------------------|
| Hard    | 100%              |
| Sand    | 100%              |
| Mud     | 100%              |
| Soft    | 100%              |

Overall: 99.7% validation accuracy

## Architecture
1D-CNN classifier:
- Input: [batch, 23 sensors, 50 timesteps]
- Conv1d: 23→64→128→256 channels
- AdaptiveAvgPool1d → Linear(256, 4)
- Output: 4 terrain classes

## Files
- IK-based_trot.py: Go2 trot gait with inverse kinematics
- data_collection.py: automated terrain data collection
- classifier.py: TerrainDataset + 1D-CNN model
- training.py: training loop with time-based train/val split
- visualize.py: terrain signal comparison

## Stack
MuJoCo, PyTorch, NumPy, Pandas, Python
