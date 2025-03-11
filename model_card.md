# Model Card

These are Inductive Moment Matching (IMM) models described in the paper [Inductive Moment Matching](https://arxiv.org/abs/2503.07565). We include the following models in this release:

We provide pretrained checkpoints through our [repo](https://huggingface.co/lumaai/imm) on Hugging Face:
* IMM on CIFAR-10: [cifar10.pkl](https://huggingface.co/lumaai/imm/resolve/main/cifar10.pt).
* IMM on ImageNet-256x256:  
  1. `t-s` is passed as second time embedding, trained with `a=2`: [imagenet256_ts_a2.pkl](https://huggingface.co/lumaai/imm/resolve/main/imagenet256_ts_a2.pkl).
  2. `s` is passed as second time embedding directly, trained with `a=1`: [imagenet256_s_a1.pkl](https://huggingface.co/lumaai/imm/resolve/main/imagenet256_s_a1.pkl).


## Intended Use

This model is provided exclusively for research purposes. Acceptable uses include:

- Academic research on generative modeling techniques
- Benchmarking against other generative models
- Educational purposes to understand Inductive Moment Matching algorithms
- Exploration of model capabilities in controlled research environments

Prohibited Uses:

- Any commercial applications or commercial product development
- Integration into products or services offered to customers
- Generation of content for commercial distribution
- Any applications that could result in harm, including but not limited to:
    - Creating deceptive or misleading content
    - Generating harmful, offensive, or discriminatory outputs
    - Circumventing security systems
    - Creating deepfakes or other potentially harmful synthetic media
    - Any use case that could negatively impact individuals or society

## Limitations

The IMM models have several limitations common to image generation models:

- Limited Resolution: The models are trained on specific resolutions (CIFAR-10 and 256x256 for ImageNet), and generating images at significantly higher resolutions may result in quality degradation or artifacts.
- Computational Resources: Training and inference require substantial computational resources, which may limit their practical applications in resource-constrained environments.
- Training Data Limitations: The models are trained on specific datasets (CIFAR-10 and ImageNet), and may not generalize well to other domains or data distributions.
- Generalization to Unseen Data: The models may not generalize well to unseen data or domains, which is a common limitation for generative models.


