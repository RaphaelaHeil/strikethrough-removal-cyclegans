# Strikethrough Removal From Handwritten Words Using CycleGANs

[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

### [Raphaela Heil](mailto:raphaela.heil@it.uu.se):envelope:, [Ekta Vats](ekta.vats@it.uu.se) and [Anders Hast](anders.hast@it.uu.se)

Code and related resources for the [ICDAR 2021](https://icdar2021.org/) paper **Strikethrough Removal From Handwritten Words Using CycleGANs**

## Table of Contents
1. [Code](#code)
    1. [Strikethrough Removal](#strikethrough-removal)
    2. [Strikethrough Classification](#strikethrough-classification)
    3. [Strikethrough Identification](#strikethrough-identification)
    4. [Running the Code](#running-the-code)
2. [Data](#data)
3. [Citation](#citation)
4. [Acknowledgements](#acknowledgements)

## Code
Each of the following subdirectories contains the code that was used in the context of this paper. Additionally, Python requirements and the original configuration(s) are included for each. Configuration files have to be modified with local paths to input and output directories before running.

Model checkpoints are attached in the release of this repository.

### Strikethrough Removal
- code for training various forms of CycleGANs to remove strikethrough from handwritten words
- the CycleGAN code is based on [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
  > @inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

### Strikethrough Classification
- code to train a DenseNet121 to classify a struck-through word image into one of seven types of strikethrough

### Strikethrough Identification
- code to train a DenseNet121 to identify whether a given word image is struck-through or not (i.e. 'clean')

### Running the Code

#### Train
In order to train any of the three models, run:
```
python src/train.py -configfile <path to config file> -config <name of section from config file>
```

If no `configfile` is defined, the script will assume `config.cfg` in the current working directory. If no `config` is defined, the script will assume `DEFAULT`.

#### Test
For testing, run:
```
python src/train.py -configfile <path to config file> -data <path to data dir>
```
- `configfile` should point to the config file in an output directory of a train run (or one of the checkpoint config files)
- `data` should point to a directory containing `struck` and `struck_gt` sub-directories, e.g. one of the datasets presented in [Data](#data)
- an additional flag `-save` can be specified to save the cleaned images, otherwise only performance metrics (F<sub>1</sub> score and RMSE) will be logged


## Data
- Synthetic strikethrough dataset on Zenodo: [https://doi.org/10.5281/zenodo.4767094](https://doi.org/10.5281/zenodo.4767094)
  - based on the [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) database
  - multi-writer
  - generated using [https://doi.org/10.5281/zenodo.4767062](https://doi.org/10.5281/zenodo.4767062)
- Genuine strikethrough dataset on Zenodo: [https://doi.org/10.5281/zenodo.4765062](https://doi.org/10.5281/zenodo.4765062)
  - single-writer
  - blue ballpoint pen
  - clean and struck word images registered based on:
    >J. Öfverstedt, J. Lindblad and N. Sladoje, "Fast and Robust Symmetric Image Registration Based on Distances Combining Intensity and Spatial Information," in IEEE Transactions on Image Processing, vol. 28, no. 7, pp. 3584-3597, July 2019, doi: 10.1109/TIP.2019.2899947.
    ([Paper](https://ieeexplore.ieee.org/document/8643403), [Code](https://github.com/MIDA-group/py_alpha_amd_release))

## Citation
ICDAR 2021
```
@INPROCEEDINGS{heil2021strikethrough,
  author={Heil, Raphaela and Vats, Ekta and Hast, Anders},
  booktitle={2021 International Conference on Document Analysis and Recognition (ICDAR)},
  title={{Strikethrough Removal from Handwritten Words Using CycleGANs}},
  year={2021},
  pubstate={to appear}}
```

## Acknowledgements
- R.Heil would like to thank [Nicolas Pielawski](https://scholar.google.se/citations?user=MmqXB5oAAAAJ), [Håkan Wieslander](https://scholar.google.se/citations?user=PLJ8O9MAAAAJ), [Johan Öfverstedt](https://scholar.google.se/citations?user=GMminVMAAAAJ) and [Anders Brun](https://scholar.google.se/citations?user=LQ4p1qQAAAAJ) for their helpful comments and fruitful discussions.
- The computations were enabled by resources provided by the Swedish National Infrastructure for Computing ([SNIC](https://snic.se/)) at the High Performance Computing Center North ([HPC2N](https://www.hpc2n.umu.se/)) partially funded by the Swedish Research Council through grant agreement no. 2018-05973.
