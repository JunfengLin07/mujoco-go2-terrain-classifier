import classifier
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import numpy as np
import math
from IKbased_trot import get_trot_targets,set_terrain