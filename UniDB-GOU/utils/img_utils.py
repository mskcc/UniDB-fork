import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils as util
from utils.helpers import gaus2d

accimage = None


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array

    currently, min_max and type conversion not applied
    """

    img_np = tensor.detach().cpu().numpy()
    if len(img_np.shape) == 3:
        img_np = np.transpose(img_np, (1, 2, 0)) 

    return img_np


def save_img(input, gt, output, img_path, mode="RGB"):
    # cv2.imwrite(img_path, img)
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4)) # figsize adjusts the overall size of the figure
    axs[0].imshow(np.rot90(np.sum(input, axis=2), -1), cmap='gray')
    axs[0].set_title('Input')
    plt.axis('off')

    axs[1].imshow(np.rot90(np.sum(gt, axis=2), -1), cmap='gray')
    axs[1].set_title('GT')
    plt.axis('off')


    axs[2].imshow(np.rot90(np.sum(output, axis=2), -1), cmap='gray')
    axs[2].set_title('Out')

    plt.axis('off')
    plt.savefig(img_path, dpi=300)
    plt.close()


def get_results(test_data, model, sde, slabmodel, avg_L1, scale, inferences):
    #### input dataset_LQ
    LQ, GT = test_data["LQ"], test_data["GT"]
    LQ = util.upscale(LQ, scale)
    LQ = LQ.to(sde.device) 

    SR_images = torch.zeros((inferences, LQ.shape[1], LQ.shape[2], LQ.shape[2]))
    # comment
    for j in range(inferences):
        model.feed_data(LQ, LQ, GT)  # current_state, LQ, GT
        sde.set_condition(LQ) # NEW
        sde.set_mu(LQ) # NEW
        model.test(sde, save_states=False)
        visuals = model.get_current_visuals()
        SR_images[j] = visuals["Output"]

    SR_img_gauss = torch.mean(SR_images, dim=0)

    input_gauss = visuals["Input"]
    ref_gauss = visuals["GT"]

    if LQ.shape[1] != 1:
        slabmodel.feed_data(LQ, GT)  # current_state, LQ, GT
        avg_L1 += slabmodel.check_parameters()
        visuals = slabmodel.get_current_visuals()
        slab_SR_img_gauss = visuals["Output"]
    else:
        slab_SR_img_gauss = torch.zeros((LQ.shape[1], LQ.shape[2], LQ.shape[2]))

    x = np.linspace(-LQ.shape[2]/2, LQ.shape[2]/2, LQ.shape[2])
    y = np.linspace(-LQ.shape[2]/2, LQ.shape[2]/2, LQ.shape[2])
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    z = gaus2d(x, y)
    z = torch.from_numpy(z[None,:,:]).float()

    SR_img = SR_img_gauss #/ z
    input_img = input_gauss #/ z
    ref = ref_gauss #/ z
    slab_SR_img = slab_SR_img_gauss #/ z

    return SR_img, input_img, ref, slab_SR_img, avg_L1, SR_images


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

