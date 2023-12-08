# Cascade-Zero123: One Image to Highly Consistent 3D with Self-Prompted Nearby Views

### [Project Page](https://cascadezero123.github.io/) | [Arxiv Paper](https://arxiv.org/abs/2312.04424) | [Video](https://youtu.be/R_FfdGl_BPo)

![block](./imgs/nvs.png)   
Cascade-Zero123 progressively extracts the 3D information from one single image via self-prompted nearby views. View-consistent images can be generated by constructing the structure in a cascade manner.

![block](./imgs/method.png)
Cascade-Zero123 can be divided into two parts. The left part is Base-0123, which takes a set of R and T values as input to generate corresponding multi-view images. These output images are concatenated with the input condition image and its corresponding camera pose, forming a self-prompted input denoted as a set of c(xc, ∆R, ∆T) for the right part Refiner-0123.

## Requirements
Pytorch 2.0 for faster training and inference.
```
conda create -f environment.yml
```
or 
```
conda create -n cascade-zero123 python=3.9
conda activate cascade-zero123
pip install -r requirements.txt
```

Install [xformer](https://github.com/facebookresearch/xformers#installing-xformers) properly to enable efficient transformers.
```commandline
conda install xformers -c xformers
# from source
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

```

## Data Preparation
Download Zero123's Objaverse Renderings data:
```commandline
wget https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz
```

Configure accelerator by
```commandline
accelerate config
```

##  Training
Launch training:

Follow Original Zero123, fp32, gradient checkpointing, and EMA are turned on.
```commandline
accelerate launch train_cascade0123.py \
--train_data_dir /data/zero123/views_release \
--pretrained_model_name_or_path lambdalabs/sd-image-variations-diffusers \
--train_batch_size 192 \
--dataloader_num_workers 16 \
--output_dir logs \
--use_ema \
--gradient_checkpointing \
--mixed_precision no
```

While bf16/fp16 is also supported by running below
```commandline
accelerate launch train_cascade0123.py \
--train_data_dir /data/zero123/views_release \
--pretrained_model_name_or_path lambdalabs/sd-image-variations-diffusers \
--train_batch_size 192 \
--dataloader_num_workers 16 \
--output_dir logs \
--use_ema \
--gradient_checkpointing \
--mixed_precision bf16
```

For monitoring training progress, we recommand [wandb](https://wandb.ai/site) for its simplicity and powerful features.
```commandline
wandb login
```


##  Acknowledgement
This repository is based on original [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) and its diffuser implementation [zero123-hf](https://github.com/kxhit/zero123-hf). Thanks for their awesome works.


##  Citation
If you find this work repository/work helpful in your research, welcome to cite the paper and give a ⭐:

```
@article{Cascadezero123,
  author = {Yabo Chen, Jiemin Fang, Yuyang Huang, Taoran Yi, Xiaopeng Zhang, Lingxi Xie, Xinggang Wang, Wenrui Dai, Hongkai Xiong,and Qi Tian},
  title = {Cascade-Zero123: One Image to Highly Consistent 3D with Self-Prompted Nearby Views},
  year = {2023},
  journal={arXiv preprint arXiv:2312.04424}
```

##  On Coming
⬜ Releasing the checkpoints  
⬜ Novel View Synthesis testing code  
⬜ Single Image to 3D testing code  
⬜ Scripts of convert diffusers back to zero123 format  
