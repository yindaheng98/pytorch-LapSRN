# PyTorch LapSRN
Implementation of CVPR2017 Paper: "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution"(http://vllab.ucmerced.edu/wlai24/LapSRN/) in PyTorch

## Usage

### Prepare
```
docker run --rm -it -v $(pwd)/pytorch-LapSRN:/workspace --name lapsrn --gpus all pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime bash
pip install h5py scipy matplotlib
```
### Training
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED]

PyTorch LapSRN

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=10
  --cuda                Use cuda?
  --resume RESUME       Path to checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --momentum MOMENTUM   Momentum, Default: 0.9
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay, Default: 1e-4
  --pretrained PRETRAINED
                        path to pretrained model (default: none)

```
An example of training usage is shown as follows:
```
python main_lapsrn.py --cuda
```
### Generate dataset
```
ffmpeg -i 4K.webm -g 30 -ss 00:00:00 -t 00:00:30 "frames/4K/frame%3d.png"
ffmpeg -i 4K.webm -g 30 -ss 00:00:00 -t 00:00:30 -s 1920x1080 "frames/1080p/frame%3d.png"
ffmpeg -i 4K.webm -g 30 -ss 00:00:00 -t 00:00:30 -s 960x540 "frames/540p/frame%3d.png"
```
### Generate and run jobs
```
./job_gen.sh ./job_templates/train_lapsrn.sh 64
./job_run.sh
ls checkpoint/lapsrn_model_*_epoch_100.pth
cp checkpoint/lapsrn_model_*_epoch_100.pth model/
./job_gen.sh ./job_templates/eval_lapsrn.sh 64
./job_run.sh

./job_gen.sh ./job_templates/train_frogsrn.sh 64
./job_run.sh
ls checkpoint/frogsrn_model_*_epoch_100.pth
cp checkpoint/frogsrn_model_*_epoch_100.pth model/
./job_gen.sh ./job_templates/eval_frogsrn.sh 64
./job_run.sh
```

### Evaluation
```
usage: eval.py [-h] [--cuda] [--model MODEL] [--dataset DATASET]
               [--scale SCALE]

PyTorch LapSRN Eval

optional arguments:
  -h, --help         show this help message and exit
  --cuda             use cuda?
  --model MODEL      model path
  --dataset DATASET  dataset name, Default: Set5
  --scale SCALE      scale factor, Default: 4
```

### Demo
```
usage: demo.py [-h] [--cuda] [--model MODEL] [--image IMAGE] [--scale SCALE]

PyTorch LapSRN Demo

optional arguments:
  -h, --help     show this help message and exit
  --cuda         use cuda?
  --model MODEL  model path
  --image IMAGE  image name
  --scale SCALE  scale factor, Default: 4
```

We convert Set5 test set images to mat format using Matlab, for best PSNR performance, please use Matlab

### Prepare Training dataset
  - We provide a simple hdf5 format training sample in data folder with 'data', 'label_x2', and 'label_x4' keys, the training data is generated with Matlab Bicubic Interplotation, please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-LapSRN/tree/master/data) for creating training files.

### Performance
  - We provide a pretrained LapSRN x4 model trained on T91 and BSDS200 images from [SR_training_datasets](http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip) with data augmentation as mentioned in the paper
  - No bias is used in this implementation, and another difference from paper is that Adam optimizer with 1e-4 learning is applied instead of SGD
  - Performance in PSNR on Set5, Set14, and BSD100
  
| DataSet/Method        | LapSRN Paper          | LapSRN PyTorch|
| ------------- |:-------------:| -----:|
| Set5      | 31.54      | **31.65** |
| Set14     | 28.19      | **28.27** |
| BSD100    | 27.32      | **27.36** |

### ToDos
  - LapSRN x8
  - LapGAN Evaluation
  
### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{LapSRN,
        author    = {Lai, Wei-Sheng and Huang, Jia-Bin and Ahuja, Narendra and Yang, Ming-Hsuan}, 
        title     = {Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution}, 
        booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
        year      = {2017}
    }
