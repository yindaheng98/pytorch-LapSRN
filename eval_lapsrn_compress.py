import os
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import json
import random
import gzip
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

avg_compress_time = {}
avg_compress_size = {}
avg_decompress_time = {}
avg_compress_rate = {}

def add_result(name, compresslevel, to, data):
    if name not in to:
        to[name] = {}
    if compresslevel not in to[name]:
        to[name][compresslevel] = 0.0
    to[name][compresslevel] += data/opt.sample
    return to

def compress_test(data, name):
    for compresslevel in range(1,10):
        start_time = time.time()
        data = gzip.compress(data, compresslevel=compresslevel)
        compress_time = time.time() - start_time
        print(name, "compress time", compresslevel, compress_time)
        add_result(name, compresslevel, avg_compress_time, compress_time)

        compress_size = len(data)/1024
        print(name, "compress size", compresslevel, compress_size)
        add_result(name, compresslevel, avg_compress_size, compress_size)

        start_time = time.time()
        data = gzip.decompress(data)
        decompress_time = time.time() - start_time
        print(name, "decompress time", compresslevel, decompress_time)
        add_result(name, compresslevel, avg_decompress_time, decompress_time)

        compress_rate = compress_size/(len(data)/1024)
        print(name, "compress rate", compresslevel, compress_rate)
        add_result(name, compresslevel, avg_compress_rate, compress_rate)



result = {}
def record(k,v):
    print(k, "=", v)
    result[k] = v

record("Scale", opt.scale)
record("Dataset", opt.dataset)
record("avg_compress_time", avg_compress_time)
record("avg_compress_size", avg_compress_size)
record("avg_decompress_time", avg_decompress_time)
record("avg_compress_rate", avg_compress_rate)

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

    HR_2x, HR_4x, convt_R1, convt_R2 = model(im_l_y)
    convt_R1 = convt_R1.cpu().detach().numpy()
    convt_R2 = convt_R2.cpu().detach().numpy()

    compress_test(convt_R1, "compress_2x_time")
    compress_test(convt_R2, "compress_4x_time")



with open(os.path.join(opt.result, 'result.json'), "w") as f:
    json.dump(result, f)