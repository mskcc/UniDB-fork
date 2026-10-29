import torch.nn.functional as F


######## super-resolution ###########

def upscale(tensor, scale=4, mode='bicubic'):
    tensor = F.interpolate(tensor, scale_factor=scale, mode=mode)
    return tensor




