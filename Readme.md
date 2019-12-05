EE838 HOME WORK 3
======================
### Prerequisite

Dowload data set from the link: : https://www.dropbox.com/s/u842yorwmap7xij/GOPRO_Large.zip?dl=0 \
Create data folder:

```mkdir data``` 

Unzip GoPro dataset to **data** folder such that we have:
- data/train : train data
- data/test : test data

Set up environment:

```conda create -n deblur python=3.6```\
```conda activate deblur```\
```pip install -r requirement.txt```

### How to train 
Train the network by run corresponding command below:

One scale

```./one_scale_no_lsc.sh``` 

One scale with long skip connection

```./one_scale__lsc.sh```

Multi scale 

```./multi_scale_no_lsc.sh```

Multi scale with long skip connection 

```./multi_scale_with_lsc.sh```

### How to test 
I provide pretrained model at url: https://drive.google.com/file/d/1OrtRLABEVb-nLHf39CamDKp4ayrxIDi9/view?usp=sharing

upzip the pretrained model to **src** folder such that we have these folders:
- one_scale1
- one_scale_lsc1
- multi_scale1
- multi_scale_lsc1
- multi_scale_lsc1000

Run test:

```./test_model.sh```

All the result will be store in **val** folder

In case that you want to test your model, read the test_model.sh and modify the pretrained_model path.

### PSNR, SSIM, MS-SSIM
I used **SKIMAGE** library for calculate PSNR and SSIM 

```python
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
```

For MS-SSIM, I used **Tensorflow** code which is available at: https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py
The code is hard copy to **utils.py**, so we don't need to worry about the dependency. 

 
### Result

