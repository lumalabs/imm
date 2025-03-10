# Inductive Moment Matching


Official Implementation of [Inductive Moment Matching]()

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
<a href="">[Paper]</a>
<a href="">[Blog]</a> 
</div>

----
 
We provide the inference script so far. The full training/evaluation code will be released at a later date TBD.
 
# Dependencies

To install all packages in this codebase along with their dependencies, run
```sh
conda env create -f env.yml
```

# Pre-trained models

We provide pretrained checkpoints through our [repo](https://huggingface.co/lumaai/imm) on Hugging Face:
* IMM on CIFAR-10: [cifar10.pkl](https://huggingface.co/lumaai/imm/resolve/main/cifar10.pt).
* IMM on ImageNet-256x256:  
  1. `t-s` is passed as second time embedding, trained with `a=2`: [imagenet256_ts_a2.pkl](https://huggingface.co/lumaai/imm/resolve/main/imagenet256_ts_a2.pkl).
  2. `s` is passed as second time embedding directly, trained with `a=1`: [imagenet256_s_a1.pkl](https://huggingface.co/lumaai/imm/resolve/main/imagenet256_s_a1.pkl).

# Sampling

The checkpoints can be tested via
```sh
python generate_images.py --config-name=CONFIG_NAME eval.resume=CKPT_PATH REPLACEMENT_ARGS
```
where `CONFIG_NAME` is `im256_generate_images.yaml` or `cifar10_generate_images.yaml` and `CKPT_PATH` is the path to your checkpoint. When loading `imagenet256_s_a1.pkl`, `REPLACEMENT_ARGS` needs to be `network.temb_type=identity`. Otherwise, `REPLACEMENT_ARGS` is empty. 