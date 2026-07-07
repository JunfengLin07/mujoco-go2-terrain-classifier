import pandas as pd
import matplotlib.pyplot as plt

# load all four terrains
hard = pd.read_csv('sensor_log_hard.csv')
sand = pd.read_csv('sensor_log_sand.csv')
mud  = pd.read_csv('sensor_log_mud.csv')
soft = pd.read_csv('sensor_log_soft.csv')

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# plot 1 — body height
axes[0].plot(hard['time'], hard['height'], label='hard', alpha=0.7)
axes[0].plot(sand['time'], sand['height'], label='sand', alpha=0.7)
axes[0].plot(mud['time'],  mud['height'],  label='mud',  alpha=0.7)
axes[0].plot(soft['time'], soft['height'], label='soft', alpha=0.7)
axes[0].set_title('Body Height')
axes[0].legend()

# plot 2 — FL thigh torque
axes[1].plot(hard['time'], hard['torque_1'], label='hard', alpha=0.7)
axes[1].plot(sand['time'], sand['torque_1'], label='sand', alpha=0.7)
axes[1].plot(mud['time'],  mud['torque_1'],  label='mud',  alpha=0.7)
axes[1].plot(soft['time'], soft['torque_1'], label='soft', alpha=0.7)
axes[1].set_title('FL Thigh Torque')
axes[1].legend()

# plot 3 — contact force 0
axes[2].plot(hard['time'], hard['contact_0'], label='hard', alpha=0.7)
axes[2].plot(sand['time'], sand['contact_0'], label='sand', alpha=0.7)
axes[2].plot(mud['time'],  mud['contact_0'],  label='mud',  alpha=0.7)
axes[2].plot(soft['time'], soft['contact_0'], label='soft', alpha=0.7)
axes[2].set_title('Contact Force 0')
axes[2].legend()

# plot 4 — accel z
axes[3].plot(hard['time'], hard['accel_z'], label='hard', alpha=0.7)
axes[3].plot(sand['time'], sand['accel_z'], label='sand', alpha=0.7)
axes[3].plot(mud['time'],  mud['accel_z'],  label='mud',  alpha=0.7)
axes[3].plot(soft['time'], soft['accel_z'], label='soft', alpha=0.7)
axes[3].set_title('Vertical Acceleration')
axes[3].legend()

plt.tight_layout()
plt.savefig('terrain_comparison.png')
plt.show()
