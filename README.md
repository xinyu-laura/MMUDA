# MMUDA

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-robust-semantic-segmentation-of/semantic-segmentation-on-dada-seg)](https://paperswithcode.com/sota/semantic-segmentation-on-dada-seg?p=towards-robust-semantic-segmentation-of)

This is the source code of paper: Towards Robust Semantic Segmentation of Accident Scenes via Multi-Source Mixed Sampling and Meta-Learning. Part of the code is borrowed from [TSMLDG] (https://github.com/koncle/TSMLDG) and [mmsegmentation] (https://github.com/open-mmlab/mmsegmentation).

## Environment
PyTorch (version >= 1.7.1), mmcv, and others in `requirements.txt`.

## Train the model

The model can be trained with `train.py`. If we want to train the MLDG with five source domains(W, A, B, I, C) and one target domain(D), we can parse such as these arguments.
```
python train.py --name exp --source WABIC --target D --train-num 1 --mix 2 --network O
```
For more details, please refer to `train.py`

## Test the model

The model can be trained with following command using `eval.py`. 
```
python eval.py --name exp --targets D --test-size 16
```

## Predict

```
python predict.py --name exp --targets D --test-size 16
```
## Others
Create a folder `pretrained` in the current directory and put the pretrained model (e.g. mit_b2.pth) there.

# Publications

If you find this work useful, please consider referencencing the following paper:

**Towards Robust Semantic Segmentation of Accident Scenes via Multi-Source Mixed Sampling and Meta-Learning.**
Xinyu Luo*, Jiaming Zhang*, Kailun Yang, Alina Roitberg, Kunyu Peng, Rainer Stiefelhagen.
In Workshop on Autonomous Driving (**WAD**) with IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), New Orleans, LA, United States, June 2022.
[[**PDF**](https://arxiv.org/pdf/2203.10395.pdf)]

```
@inproceedings{luo2022towards,
  title={Towards Robust Semantic Segmentation of Accident Scenes via Multi-Source Mixed Sampling and Meta-Learning},
  author={Luo, Xinyu and Zhang, Jiaming and Yang, Kailun and Roitberg, Alina and Peng, Kunyu and Stiefelhagen, Rainer},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2022}
}
```
