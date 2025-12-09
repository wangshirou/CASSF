# CASSF
This repository provides the official PyTorch implementation of the paper: **"A Class-Aware Semi-Supervised Framework for Semantic Segmentation of High-Resolution Remote Sensing Imagery"**, published in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2025. 🔗[Paper Link](https://ieeexplore.ieee.org/document/11143061)


## 📂 Repository Structure
```sh
└── CASSF/
    ├── data
    │   ├── Vaihingen
    │   ├── __init__.py
    │   └── build_data.py
    ├── model
    │   ├── DeepLabv3p.py
    │   ├── Unet.py
    │   └── __init__.py
    ├── myutils
    │   ├── __init__.py
    │   └── utils.py
    └── train_CASSF.py
```

## 📊 Dataset
We provide **cropped and pre-split semi-supervised training datasets** for the following benchmarks:

- ISPRS Vaihingen Dataset (IR-R-G + DSM & labels)
- ISPRS Potsdam Dataset(R-G-B-IR + DSM & labels)
- LoveDA Urban Dataset(R-G-B & labels)
- LoveDA Rural Dataset(R-G-B & labels)

Each dataset includes multiple labeled/unlabeled splits for semi-supervised learning: `1/2`, `1/4`, `1/8`, `1/16`, `1/32`, `1/64` (e.g., `1/8` means 12.5% labeled and 87.5% unlabeled).  

For quick start, the **Vaihingen dataset** is already included in this repo under `data/Vaihingen/`. For full experiments with all datasets, please download from: 🔗 [CASSF_data - Baidu Netdisk](https://pan.baidu.com/s/17bO0FPKYcl9o3A0ziJd0Ug?pwd=CASS).

**Note:** Test datasets are **not provided** and must be downloaded separately from the official sources:  

- 🌐 **ISPRS:** [Vaihingen & Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx)  
- 🌐 **LoveDA:** [Urban & Rural](https://zenodo.org/records/5706578)


## 🚀 Quick Start
1. Install dependencies

    It is recommended to use Python 3.8.
    Before training, please install the required dependencies (e.g., PyTorch, TorchVision, GDAL, Visdom, Scikit-learn, etc.) suitable for your environment.

2. Run training

    Once dependencies are installed, start training with:
    ```python
    python train_CASSF.py
    ```

## 🔖 Citation
If you use this code or data, please cite our paper:

```bibtex
@article{CASSF,
  author={Wang, Shirou and Su, Cheng and Zhang, Xiaocan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A Class-Aware Semi-Supervised Framework for Semantic Segmentation of High-Resolution Remote Sensing Imagery}, 
  year={2025},
  volume={18},
  pages={22372-22391},
  doi={10.1109/JSTARS.2025.3601148}}
```
