import os
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import json
import random
import numpy as np
from PIL import Image
from PIL.Image import BICUBIC
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import DatasetFromFrames
from lapsrn import Net

parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--depth", type=int, default=4, help="depth of the network")
parser.add_argument("--nFeat", type=int, default=16, help="depth of the hide channels")
parser.add_argument("--model", default="model/model_epoch_100.pth", type=str, help="model path")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--dataset", default="frames", type=str, help="dataset name, Default: frames")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--result", default="result", type=str, help="result path")
parser.add_argument("--sample", default=10, type=int, help="How many pictures for eval?")

def PSNR(pred, gt, shave_border=0):
    imdff = np.array(pred) - np.array(gt)
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = Net(opt.depth, opt.nFeat)
if opt.cuda:
    model.cuda()
    checkpoint = torch.load(opt.model)
else:
    checkpoint = torch.load(opt.model, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model"].state_dict())

avg_psnr_2x_predicted = 0.0
avg_psnr_2x_bicubic = 0.0
avg_psnr_4x_predicted = 0.0
avg_psnr_4x_bicubic = 0.0
avg_elapsed_time = 0.0

train_set = DatasetFromFrames("frames", opt.sample)
training_data_loader = DataLoader(dataset=train_set, shuffle=True)

for i, batch in enumerate(training_data_loader, 1):
    os.makedirs(os.path.join(opt.result, str(i)), exist_ok=True)

    print("Processing epoch %d" % i)
    
    im_l_y, im_gt_2x_y, im_gt_4x_y = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)

    if opt.cuda:
        model = model.cuda()
        im_l_y = im_l_y.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    HR_2x, HR_4x, _, _, _ = model(im_l_y)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time
    HR_4x = HR_4x.cpu()
    HR_2x = HR_2x.cpu()
    
    im_h_4x_y = transforms.ToPILImage()(torch.clamp(HR_4x.data[0], 0, 1))
    im_h_4x_y.save(os.path.join(opt.result, str(i), "im_h_4x_y.png"))
    im_h_2x_y = transforms.ToPILImage()(torch.clamp(HR_2x.data[0], 0, 1))
    im_h_2x_y.save(os.path.join(opt.result, str(i), "im_h_2x_y.png"))
    im_gt_2x_y = transforms.ToPILImage()(im_gt_2x_y.cpu().data[0])
    im_gt_2x_y.save(os.path.join(opt.result, str(i), "im_gt_2x_y.png"))
    im_gt_4x_y = transforms.ToPILImage()(im_gt_4x_y.cpu().data[0])
    im_gt_4x_y.save(os.path.join(opt.result, str(i), "im_gt_4x_y.png"))

    psnr_4x_predicted = PSNR(im_gt_4x_y, im_h_4x_y,shave_border=opt.scale)
    avg_psnr_4x_predicted += psnr_4x_predicted
    print("PSNR_2x_predicted", psnr_4x_predicted)
    psnr_2x_predicted = PSNR(im_gt_2x_y, im_h_2x_y,shave_border=opt.scale)
    avg_psnr_2x_predicted += psnr_2x_predicted
    print("PSNR_4x_predicted", psnr_2x_predicted)

    im_l_y = transforms.ToPILImage()(im_l_y.cpu().data[0])
    im_b_4x_y = im_l_y.resize(im_gt_4x_y.size, resample=BICUBIC)
    im_b_4x_y.save(os.path.join(opt.result, str(i), "im_b_4x_y.png"))
    im_b_2x_y = im_l_y.resize(im_gt_2x_y.size, resample=BICUBIC)
    im_b_2x_y.save(os.path.join(opt.result, str(i), "im_b_2x_y.png"))
    psnr_4x_bicubic = PSNR(im_gt_4x_y, im_b_4x_y,shave_border=opt.scale)
    avg_psnr_4x_bicubic += psnr_4x_bicubic
    print("PSNR_4x_bicubic", psnr_4x_bicubic)
    psnr_2x_bicubic = PSNR(im_gt_2x_y, im_b_2x_y,shave_border=opt.scale)
    avg_psnr_2x_bicubic += psnr_2x_bicubic
    print("PSNR_2x_bicubic", psnr_2x_bicubic)


result = {}
def record(k,v):
    print(k, "=", v)
    result[k] = v

record("Scale", opt.scale)
record("Dataset", opt.dataset)
record("PSNR_4x_predicted", avg_psnr_4x_predicted/opt.sample)
record("PSNR_4x_bicubic", avg_psnr_4x_bicubic/opt.sample)
record("PSNR_2x_predicted", avg_psnr_2x_predicted/opt.sample)
record("PSNR_2x_bicubic", avg_psnr_2x_bicubic/opt.sample)
record("Time_per_picture", avg_elapsed_time/opt.sample)

with open(os.path.join(opt.result, 'result.json'), "w") as f:
    json.dump(result, f)