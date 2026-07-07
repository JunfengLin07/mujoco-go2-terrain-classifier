## Terrain Classifier Results

| Terrain | Correctly Classified |
|---------|---------------------|
| Hard    | ✓ 100%              |
| Sand    | ✓ 100%              |
| Mud     | ✓ 100%              |
| Soft    | ✓ 100%              |

Overall validation accuracy: 99.7%
Architecture: 1D-CNN (Conv1d 23→64→128→256, AdaptiveAvgPool, Linear)
Input: 50-timestep windows of 23 proprioceptive sensor channels
