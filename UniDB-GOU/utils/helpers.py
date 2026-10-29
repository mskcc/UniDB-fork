import numpy as np
import sys
# TODO sys.path.insert(0, str('/otazolab_ess/Schellenberg/Dynamic_MRI_Reconstruction/mr-image-metrics'))
# from medimetrics.metrics.ssim import SSIM
# from medimetrics.metrics.mse import MSE
# from medimetrics.metrics.nmi import NMI
# from medimetrics.metrics.lpips  import LPIPS

# define normalized 2D gaussian
def gaus2d(x=0, y=0, mx=0, my=0, sx=60, sy=60):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


# def extract_metrics(filename, input_all, ref_all, output_all, outputfolder):
    metrics = dict()
    metrics['SSIM'] = dict()
    metrics['MSE'] = dict()
    metrics['NMI'] = dict()
    metrics['LPIPS'] = dict()

    ssim = SSIM()
    mse = MSE()
    nmi = NMI()
    lpips = LPIPS()

    for key in metrics.keys():
        metrics[key][filename] = dict()

    for key in metrics.keys():
        metrics[key][filename]['input'] = []
        metrics[key][filename]['output'] = []


    for slice_nr, slice_idx in enumerate(range(input_all.shape[0])):
        print(slice_nr)
        # Extract study name
        input = input_all[slice_idx]
        ref = ref_all[slice_idx]
        output = output_all[slice_idx]

        # masking 
        # grasp_mean = np.mean(grasp, axis=2)
        # mask = create_mask(grasp_mean)
        # import matplotlib.pyplot as plt
        # plt.imshow(mask)
        # plt.savefig('mask.png')
        # plt.close()

        # rnufft = rnufft * mask[:,:,None]
        # output = output * mask[:,:,None]
        # grasp = grasp * mask[:,:,None]
        # output_Movienet = output_Movienet * mask[:,:,None]
        # refn = percentile_norm(ref)
        # outputn = percentile_norm(output)
        # inputn = percentile_norm(input)
        # plt.imshow(refn)
        # plt.savefig('ref.png')
        # plt.close()
        # plt.imshow(outputn)
        # plt.savefig('out.png')
        # plt.close()
        # plt.imshow(inputn)
        # plt.savefig('in.png')
        # plt.close()

        # plt.imshow(np.abs(inputn-refn), vmin=-0, vmax=1)
        # plt.savefig('in-ref.png')
        # plt.close()
        # plt.imshow(np.abs(outputn-refn), vmin=0, vmax=1)
        # plt.savefig('out-ref.png')
        # plt.close()
        # plt.imshow(np.abs(inputn-outputn), vmin=0, vmax=1)
        # plt.savefig('in-out.png')
        # plt.close()
        

        metrics['SSIM'][filename]['input'].append(ssim.compute(percentile_norm(ref), percentile_norm(input), sigma=0.01))
        metrics['SSIM'][filename]['output'].append(ssim.compute(percentile_norm(ref), percentile_norm(output), sigma=0.01))

        metrics['MSE'][filename]['input'].append(mse.compute(percentile_norm(ref), percentile_norm(input)))
        metrics['MSE'][filename]['output'].append(mse.compute(percentile_norm(ref), percentile_norm(output)))

        metrics['NMI'][filename]['input'].append(nmi.compute(percentile_norm(ref), percentile_norm(input)))
        metrics['NMI'][filename]['output'].append(nmi.compute(percentile_norm(ref), percentile_norm(output)))

        metrics['LPIPS'][filename]['input'].append(lpips.compute(percentile_norm_minus1_1(ref), percentile_norm_minus1_1(input)))
        metrics['LPIPS'][filename]['output'].append(lpips.compute(percentile_norm_minus1_1(ref), percentile_norm_minus1_1(output)))


        print('Done')

    np.savez(f'{outputfolder}/metric_{filename}.npz', **metrics)
    return metrics


def minmax_binning(data):
    """
    Min-max normalization of data to a range of [0, 255] and then it bins it to 256 bins.

    :param data: The arry to normalize.

    :param return: The normalized array. 
    """

    data = (data - np.min(data)) / (np.max(data)- np.min(data))
    data = np.uint8(np.round(data * 255, decimals=0))

    return data


def minmax_norm_minus1_1(data):
    """
    Min-max normalization of data to a range of [-1, 1]. 

    :param data: The arry to normalize.

    :param return: The normalized array. 
    """

    data = (data - np.min(data)) / (np.max(data)- np.min(data))
    data = data * 2 - 1

    return data


def percentile_norm(data):
    """
    Min-max normalization of data using 1st and 99th percentile.

    :param data: The arry to normalize.

    :param return: The normalized array. 
    """

    data = (data - np.percentile(data,1)) / (np.percentile(data,99)- np.percentile(data,1))

    return data


def percentile_5_95_norm(data):
    """
    Min-max normalization of data using 1st and 99th percentile.

    :param data: The arry to normalize.

    :param return: The normalized array. 
    """

    data = (data - np.percentile(data,5)) / (np.percentile(data,95)- np.percentile(data,5))

    return data


def percentile_norm_minus1_1(data):
    """
    Min-max normalization of data using 1st and 99th percentile.

    :param data: The arry to normalize.

    :param return: The normalized array. 
    """

    data = (data - np.percentile(data,1)) / (np.percentile(data,99)- np.percentile(data,1))
    data = data*2 -1
    data = np.clip(data, -1, 1)

    return data
