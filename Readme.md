Image Dehazing Network
======================
### Prerequisite

Dowload data set from the link: : https://www.dropbox.com/s/wc3b0q0d3querb3/Dehazing_datasets.zip?dl=0  \
Create data folder:

```mkdir data``` 

Unzip dataset to **data** folder such that we have:
- data/IndoorTestHazy 
- data/IndoorTrainGT
- data/IndoorTrainHazy
- data/OutdoorTestHazy 
- data/OutdoorTrainGT
- data/OutdoorTrainHazy

Set up environment:

```conda create -n dehaze python=3.6```\
```conda activate dehaze```\
```pip install -r requirement.txt```

### How to train 
Train the network by run corresponding command below:

Indoor:

```./net_train_indoor.sh``` 

Outdoor:

```./net_train_outdoor.sh```

### How to test 
I provide pretrained model at url: https://drive.google.com/file/d/1WfsmkGmo504ZI7V19_t-euKDNqwW0woC/view?usp=sharing

upzip the pretrained model to **src** folder such that we have these folders:
- resultIn/Net1/model/model_best.pt
- resultOut/Net1/model/model_best.pt

Run test script to generate output images:

```./net_test_in_out.sh```

All the result will be store in **val** folder

In case that you want to test your model, read the test_model.sh and modify the pretrained_model path.

### Evaluate NIQE

You can download MATLAB evaluation code at this link: https://www.dropbox.com/s/xpcqcucxjn2y28d/evaluation_code.zip?dl=0 \
Copy your output images into Input folder and run matlab file: evaluate_results.m to get NIQE score
 
### Result

|      | Indoor (NIQE) | Outdoor(NIQE) |
|------|---------------|---------------|
| HAZY | 6.4564        | 4.1471        |
| OUR  | 3.6753        | 3.6608        |

Statictis on 1 GPU Titan X

|                                | Indoor     | Outdoor   |
|--------------------------------|------------|-----------|
| Generator parameter            | 34.1M      | 34M       |
| Discriminator parameter        | 5.5M       | 5.5M      |
| Training time (10000 epoches)  | 52.9 hour  | 61.0 hour |
| Testing time                   | 0.0241     | 0.1765    |
