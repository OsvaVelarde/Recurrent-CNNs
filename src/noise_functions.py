import torchvision.transforms as tt
import torch

def pixel_noise(img,level):
    return img + level * torch.randn(*img.shape)

def blurr_noise(img,level):
    return tt.GaussianBlur(kernel_size=(5, 5), sigma=level)(img)

def contr_noise(img,level):
    return tt.functional.adjust_contrast(img, contrast_factor=level)

def occlu_noise(img,level):
    return tt.RandomErasing(p=1, scale=(0.02, level), ratio=(1, 1))(img)

def brightness(img,level):
    return tt.adjust_brightness(img,brightness_factor=level)

def sharp_noise(img,level):
    return tt.functional.adjust_sharpness(img,sharpness_factor=level)