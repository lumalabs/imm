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

# Checklist

- [x] Add model weights and model definitions.
- [x] Add inference scripts.
- [ ] Add evaluation scripts.
- [ ] Add training scripts.

# Acknowledgements

Some of the utility functions are based on [EDM](https://github.com/NVlabs/edm), and thus parts of the code would apply under [this license](https://github.com/NVlabs/edm/blob/main/LICENSE.txt).

# Citation

```
@article{zhou2025inductive,
  title={Inductive Moment Matching},
  author={Zhou, Linqi and Ermon, Stefano and Song, Jiaming},
  journal={arXiv preprint arXiv:2503.07565},
  year={2025}
}
```
