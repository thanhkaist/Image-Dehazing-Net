Image Dehazing Network
======================

A GAN-based single image dehazing network that removes haze from indoor and outdoor images. The model consists of a Generator (encoder–decoder with residual blocks) and a multi-scale Discriminator, trained with a combination of GAN loss, VGG perceptual loss, and pixel-wise L2 loss.

---

## Prerequisites

### 1. Dataset

Download the dataset from:
https://www.dropbox.com/s/wc3b0q0d3querb3/Dehazing_datasets.zip?dl=0

Create the data folder and extract the archive into it:

```bash
mkdir data
unzip Dehazing_datasets.zip -d data
```

The `data` folder should contain the following subfolders:

```
data/
├── IndoorTestHazy
├── IndoorTrainGT
├── IndoorTrainHazy
├── OutdoorTestHazy
├── OutdoorTrainGT
└── OutdoorTrainHazy
```

### 2. Environment Setup

Create and activate a Python 3.6 conda environment, then install dependencies:

```bash
conda create -n dehaze python=3.6
conda activate dehaze
pip install -r requirement.txt
```

---

## How to Train

Train the network for each scene type using the provided shell scripts (10 000 epochs each):

**Indoor:**
```bash
./net_train_indoor.sh
```

**Outdoor:**
```bash
./net_train_outdoor.sh
```

Model checkpoints and training logs are saved under `resultIn/` and `resultOut/` respectively.

Key training hyperparameters (configurable in `main.py` / the shell scripts):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10000 | Number of training epochs |
| `--lr` | 1e-4 | Initial learning rate (Adam) |
| `--lrDecay` | 1000 | Epoch interval for learning rate halving |
| `--batchSize` | 2 | Mini-batch size |
| `--lamda` | 1.0 | Weight for VGG perceptual loss |
| `--alpha` | 1.0 | Weight for GAN losses |
| `--resblock` | 6 | Number of residual blocks in the generator |

---

## How to Test

### Using the Pretrained Model

Download the pretrained model from:
https://drive.google.com/file/d/1WfsmkGmo504ZI7V19_t-euKDNqwW0woC/view?usp=sharing

Unzip the archive at the root of the repository so that the following paths exist:

```
resultIn/Net1/model/model_best.pt
resultOut/Net1/model/model_best.pt
```

Run the test script to generate dehazed output images:

```bash
./net_test_in_out.sh
```

All output images are saved under the `val/` folder.

### Using Your Own Trained Model

To test a custom checkpoint, call `test.py` directly and supply the path to your model:

```bash
python test.py Normal \
    --pretrained_model <path/to/your/model_best.pt> \
    --dataset Indoor \
    --resblock 6 \
    --gpu 0
```

Replace `Indoor` with `Outdoor` for outdoor images.

---

## Evaluate NIQE

Download the MATLAB evaluation code from:
https://www.dropbox.com/s/xpcqcucxjn2y28d/evaluation_code.zip?dl=0

Copy your output images into the `Input` folder, then run `evaluate_results.m` in MATLAB to obtain the NIQE score.

---

## Results

NIQE scores (lower is better) on the test sets:

|        | Indoor (NIQE) | Outdoor (NIQE) |
|--------|---------------|----------------|
| Hazy   | 6.4564        | 4.1471         |
| Ours   | 3.6753        | 3.6608         |

Model statistics on 1× Titan X GPU:

|                                  | Indoor    | Outdoor   |
|----------------------------------|-----------|-----------|
| Generator parameters             | 34.1 M    | 34.0 M    |
| Discriminator parameters         | 5.5 M     | 5.5 M     |
| Training time (10 000 epochs)    | 52.9 h    | 61.0 h    |
| Testing time per image (seconds) | 0.0241 s  | 0.1765 s  |
