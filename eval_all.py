import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--models", default="model/lapsrn_model_%d_epoch_100.pth", type=str, help="format of model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def evaluate(model_path):

    model = torch.load(model_path)["model"]

    image_list = glob.glob(opt.dataset+"/*.*") 

    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0

    for image_name in image_list:
        print("Processing ", image_name)
        im_gt_y = sio.loadmat(image_name)['im_gt_y']
        im_b_y = sio.loadmat(image_name)['im_b_y']
        im_l_y = sio.loadmat(image_name)['im_l_y']

        im_gt_y = im_gt_y.astype(float)
        im_b_y = im_b_y.astype(float)
        im_l_y = im_l_y.astype(float)

        psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)
        avg_psnr_bicubic += psnr_bicubic

        im_input = im_l_y/255.

        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

        if cuda:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        start_time = time.time()
        HR_2x, HR_4x = model(im_input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        HR_4x = HR_4x.cpu()

        im_h_y = HR_4x.data[0].numpy().astype(np.float32)

        im_h_y = im_h_y*255.
        im_h_y[im_h_y<0] = 0
        im_h_y[im_h_y>255.] = 255.
        im_h_y = im_h_y[0,:,:]

        psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=opt.scale)
        avg_psnr_predicted += psnr_predicted

    return {
        'Scale': opt.scale,
        'Dataset': opt.dataset,
        'PSNR_predicted': avg_psnr_predicted/len(image_list),
        'PSNR_bicubic': avg_psnr_bicubic/len(image_list),
        'Avg_time': avg_elapsed_time/len(image_list)
    }

for i in range(1,1000):
    model_path = opt.model
    if os.path.exists(model_path):
        print(evaluate(model_path))
