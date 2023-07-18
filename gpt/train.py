import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
import json

with open("../config/poem_gpt_config.json", "r") as f:
    config = json.load(f)

print(config)