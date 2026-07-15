import pandas as pd
import numpy
import matplotlib.pyplot as plt

df = [None] * 4
terrains = ['hard','sand','mud','soft']
for i in range(4):
    df[i] = pd.read_csv(f'sensor_log_{terrains[i]}.csv')

fig,axes = plt.subplots(4,1,figsize=(12,10))
signals = ['height','contact_0','accel_z','torque_1']
titles = ['Body Height','Contact Force','Vertical Acceleration','FL Thigh Torque']

for i in range(4):
    for j in range(4):
        axes[i].plot(df[j]['time'],df[j][signals[i]],label=terrains[j])
    axes[i].set_title(titles[i])
    axes[i].legend()

plt.tight_layout()
plt.savefig('terrain_signals.png')
plt.show()

