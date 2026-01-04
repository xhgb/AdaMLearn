# AdaMLearn
This is the Pytorch implementation for "Engram Neuron Inspired Adaptive Memorization for Lifelong Learning".

## Environment Setup

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

## Running the Code
All experiments can be launched by directly running one of the following scripts located at the project root.
Each script corresponds to a specific benchmark dataset.

```bash
python main_cifar100.py
python main_cifar_superclass.py
python main_five_dataset.py
python main_miniimagenet.py
```

No additional command-line arguments are required by default.
Dataset-specific configurations and experimental settings are handled internally by each script.

## Datasets
The dataset for CIFAR-100, CIFAR-superclass, 5-dataset will be automatically downloaded. For the experiments on MiniImageNet, please download the [train.pkl](https://drive.google.com/file/d/1fm6TcKIwELbuoEOOdvxq72TtUlZlvGIm/view?pli=1) and [test.pkl](https://drive.google.com/file/d/1RA-MluRWM4fqxG9HQbQBBVVjDddYPCri/view). The organization it as follows:

```text
data/
└── miniimagenet/
    ├── train.pkl
    └── test.pkl
