# Attention Based Coupled Framework for Road and  Pothole Segmentation

Few-shot learning approach using coupled framework for road and pothole segmentation to leverage accuracy even with fewer training samples, particularly in case of unstructured environments.

We extend DeepLabv3+ architecture by incorporating attention based refinement with feature fusion to improve the segmentation.

The exhaustive experimental results for road segmentation on KITTI and IDD datasets and pothole segmentation on IDD are reported.

To the best of author's knowledge, this is the first work
which explores few-shot learning approach for pothole
detection on [IDD](https://idd.insaan.iiit.ac.in/) dataset.

Check out our [paper]() for detailed description.

## Requirements
The code is written in python 3.
* Numpy
* Tensorflow
* Keras
* OpenCV
* Scikit-learn
* Matplotlib
* Seaborn
* segmentation_models

## Setup
The complete code is provided as a jupyter notebook.

## Acknowledgement
Evaluation metrics used are from the following links.
* [IDD Evaluation](https://github.com/AutoNUE/public-code)
* [KITTI Evaluation](https://github.com/MarvinTeichmann/KittiSeg)

Some of the code is adapted from [Keras implementation of DeeplabV3+](https://github.com/bonlime/keras-deeplab-v3-plus)

## Citation
If you benefit from this code, please cite our paper:
```
```