import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(img, targets):
    img = torch.flip(img, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return img, targets
