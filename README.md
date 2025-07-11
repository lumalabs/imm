# Inductive Moment Matching


Official Implementation of [Inductive Moment Matching](https://arxiv.org/abs/2503.07565)

<p align="center">
  <img src="assets/teaser.png" width="90%"/>
</p>

<div align="center">
  <span class="author-block">
    <a href="https://alexzhou907.github.io/">Linqi Zhou</a><sup>1</sup>,</span> 
  <span class="author-block">
    <a href="https://cs.stanford.edu/~ermon/">Stefano Ermon</a><sup>2</sup>
  </span>
  <span class="author-block">
    <a href="https://tsong.me/">Jiaming Song</a><sup>1</sup>
  </span>
</div>

<div align="center">
  <span class="author-block"><sup>1</sup>Luma AI,</span>
  <span class="author-block"><sup>2</sup>Stanford University</span>
</div>
<div align="center">
<a href="https://arxiv.org/abs/2503.07565">[Paper]</a>
<a href="https://lumalabs.ai/news/imm">[Blog]</a> 
</div>
</br>

Also check out our accompanying [position paper](https://arxiv.org/abs/2503.07154) that explains the motivation and ways of designing new generative paradigms.


# Checklist

- [x] Add model weights and model definitions.
- [x] Add inference scripts.
- [x] Add evaluation scripts.
- [x] Add training scripts.


# Dependencies

To install all packages in this codebase along with their dependencies, run
```sh
conda env create -f env.yml
``` 
For multi-node jobs, we use Slurm via `submitit`, which can be installed via
```sh
pip install submitit
```

# Pre-trained models

We provide pretrained checkpoints through our [repo](https://huggingface.co/lumaai/imm) on Hugging Face:
* IMM on CIFAR-10: [cifar10.pkl](https://huggingface.co/lumaai/imm/resolve/main/cifar10.pt).
* IMM on ImageNet-256x256:  
  1. `t-s` is passed as second time embedding, trained with `a=2`: [imagenet256_ts_a2.pkl](https://huggingface.co/lumaai/imm/resolve/main/imagenet256_ts_a2.pkl).
  2. `s` is passed as second time embedding directly, trained with `a=1`: [imagenet256_s_a1.pkl](https://huggingface.co/lumaai/imm/resolve/main/imagenet256_s_a1.pkl).


# Datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG or NPY files, along with a metadata file `dataset.json` for labels. When using latent diffusion, it is necessary to create two different versions of a given dataset: the original RGB version, used for evaluation, and a VAE-encoded latent version, used for training.

To set up CIFAR-10:

Download the (CIFAR-10 python version)[https://www.cs.toronto.edu/~kriz/cifar.html] and convert via:

```bash
python dataset_tool.py --source=YOUR_DOWNLOADED_TARGZ_FILE \
    --dest=YOUR_PATH/datasets/cifar10-32x32.zip 
```
 
To set up ImageNet-256:

1. Download the ILSVRC2012 data archive from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and extract it somewhere, e.g., `downloads/imagenet`.

2. Crop and resize the images to create the original RGB dataset:

```.bash
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \
    --dest=YOUR_PATH/datasets/img256.zip --resolution=256x256 --transform=center-crop-dhariwal
```

3. Run the images through a pre-trained VAE encoder to create the corresponding latent dataset:

```.bash
# Convert the pixel data to VAE latents
python dataset_tool.py encode --source=YOUR_PATH/datasets/img256.zip \
    --dest=YOUR_PATH/datasets/img256-sd.zip
```

4. Calculate reference statistics for the original RGB dataset, to be used with `calculate_metrics.py`:

- You can find precalculated reference stats for ImageNet-256x256 [here](https://huggingface.co/lumaai/imm/resolve/main/img256_ref_stats.pkl?download=true).

```.bash
# Compute dataset reference statistics for calculating metrics
python calculate_metrics.py ref --data=YOUR_PATH/datasets/img256.zip \
    --dest=YOUR_PATH/dataset-refs/img256.pkl
```



# Training

The default configs are in `configs/`. Before training, properly replace the `logger` and `dataset.path` fields to your own choice. You can train on a single node by

```bash
bash run_train.sh NUM_GPUS_PER_NODE CONFIG_NAME REPLACEMENT_ARGS
```
where `NUM_GPUS_PER_NODE` is number of GPUs per node, `CONFIG_NAME` is either `cifar10.yaml` or `im256.yaml`.
If you want to replace any config args, for example, `loss.a` on CIFAR-10 from `1` to `2`, you can run
```bash
bash run_train.sh 8 cifar10.yaml loss.a=2
```

To train multi-node, we use submitit and run
```bash
python launch.py --ngpus=NUM_GPUS_PER_NODE --nodes=NUM_NODES --config-name=CONFIG_NAME
```

And output folder will be created under `./outputs/`

# Sampling

The checkpoints can be tested via
```sh
python generate_images.py --config-name=CONFIG_NAME eval.resume=CKPT_PATH REPLACEMENT_ARGS
```
where `CONFIG_NAME` is `im256_generate_images.yaml` or `cifar10_generate_images.yaml` and `CKPT_PATH` is the path to your checkpoint. When loading `imagenet256_s_a1.pkl`, `REPLACEMENT_ARGS` needs to be `network.temb_type=identity`. Otherwise, `REPLACEMENT_ARGS` is empty. 


# Evaluation

The eval scripts calculate FID scores. Make sure your follow our guidelines and have reference stats files ready.

```bash
bash run_eval.sh NUM_GPUS_PER_NODE CONFIG_NAME eval.resume=YOUR_MODEL_PKL_PATH
```

For example, to evaluate on CIFAR-10 with your model saved at `outputs/cifar/network-snapshot-latest.pkl` with 8 GPUs on a single node, run
```bash
bash run_eval.sh 8 cifar10.yaml eval.resume=outputs/cifar/network-snapshot-latest.pkl
```

`YOUR_MODEL_PKL_PATH` can also be a directory containing all checkpoints of a run labeled with training iterations (i.e. your training directory). It will sort from the latest checkpoint to earliest and evaluate in that order.
 

# Acknowledgements

Some of the utility functions are based on [EDM](https://github.com/NVlabs/edm), and thus parts of the code would apply under [this license](https://github.com/NVlabs/edm/blob/main/LICENSE.txt).

# Citation

```
@inproceedings{
zhou2025inductive,
title={Inductive Moment Matching},
author={Linqi Zhou and Stefano Ermon and Jiaming Song},
booktitle={Forty-second International Conference on Machine Learning},
year={2025}
}
```