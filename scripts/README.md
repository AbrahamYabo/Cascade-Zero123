# Convert original weights to diffusers

Download original Zero123 checkpoint under `ckpts` through one of the following sources:

```
https://drive.google.com/drive/folders/1geG1IO15nWffJXsmQ_6VLih7ryNivzVs?usp=sharing
https://huggingface.co/cvlab/zero123-weights
wget https://cv.cs.columbia.edu/zero123/assets/$iteration.ckpt    # iteration = [105000, 165000, 230000, 300000]
```

Hugging Face diffusers weights are converted by script:
```commandline
cd scripts
python convert_zero123_to_diffusers.py --checkpoint_path /path/zero123/105000.ckpt --dump_path ./zero1to3 --original_config_file /path/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml
```

Hugging Face diffusers weights can convert back to Zero 1-to-3 format by script:
A parameter template_checkpoint_path needs to be specified to ensure that the parts that are not trained remain the same. 
checkpoint_path specify the output path
```commandline
cd scripts
python conver_diffusers_to_zero123.py --model_path ./zero1to3 --template_checkpoint_path ./kxic/zero123-xl --checkpoint_path ./output/output.ckpt  --original_config_file /path/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml
```

Weights are hosted here:
```commandline
https://huggingface.co/kxic/zero123-105000
https://huggingface.co/kxic/zero123-165000
https://huggingface.co/kxic/zero123-xl
```
