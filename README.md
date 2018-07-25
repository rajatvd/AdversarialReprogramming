# Adversarial Reprogramming
This repo contains the code (not very neat) for my [blog post on Adversarial Reprogramming](https://rajatvd.github.io/Exploring-Adversarial-Reprogramming/).
Based on [this paper](https://arxiv.org/pdf/1806.11146.pdf) by the Google Brain team.

Make sure to have the following packages installed:

* `pytorch` : [Get pytorch](https://pytorch.org/)
* `writefile-run` : To create python files from jupyter notebooks [writefile-run](https://pypi.org/project/writefile-run/)
* `pytorch-utils` : Set of utils I made to make training in jupyter notebooks easier [pytorch-utils](https://github.com/rajatvd/PytorchUtils)


### Description of files

* The `models.py` file contains the reprogramming modules.
* Run the `AdvReprogMNIST2.ipynb` to train an adversarial program for MNIST using the scaling input transform.
* The `LabelRemapping.ipynb` notebook implements the greedy multiple output label remapping for CIFAR.
* Run the `AdvReprogCifar.ipynb` to train an adversarial program for CIFAR using the scaling input transform and the multiple output label remappings created above.
