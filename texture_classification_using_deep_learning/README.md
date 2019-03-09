# Texture Classification using VGG Feature Correlations

The aim of my project is to come up with a texture classification scheme which utilizes correlations of feature activations of a pretrained CNN, as described in *Neural Style Transfer* paper of Gatys et. al.[1]. 

You can find the detailed explanations about the project in [project report](report.pdf)

## Getting Started

This method is tested on two texture datasets, *Textures under varying Illumination, Pose and Scale* (KTH-TIPS) [2]  and *Describable Textures Dataset* (DTD). You need to download at least one of them in order to test the algorithm.

* [KTH-TIPS2](http://www.nada.kth.se/cvap/databases/kth-tips/download.html) (11 classes)
* [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) (47 classes)

When the download completes, extract the contents to your desired location and point ```data_dir``` variable in [model_training.py](model_training.py) to the extracted folder. You also need to set ```n_class``` variable according to the number of  classes of the dataset. Then run [model_training.py](model_training.py) to perform training.

## Prerequisites

You need to have [PyTorch](https://pytorch.org/) installed and [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) enabled in order to run this project.

## References

[1] L. Gatys, A. S. Ecker, and M. Bethge. A Neural Algorithm of Artistic
Style. arXiv:1508.06576[cs.CV]. August 2015

