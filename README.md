# MolGAN-pytorch

🦑This repository is a Pytorch implementation of [MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/pdf/1805.11973.pdf).

## Installation

#### Dependency

```bash
pip install -r requirements.txt
```

#### Dataset Generation

```bash
# Download gdb Dataset and NP/SA scores
bash ./data/download_dataset.sh
# Generate QM9 Dataset
python ./data/sparse_molecular_dataset.py
```

## Run

To train the MolGAN model, you could run the experiment with the following command:

```bash
python main_gan.py --lambda_wgan 0.05 --desc "lambda_wgan 0.05"
```

The `lambda_wgan` is a config parameter, which refers to the hyperparamer $\lambda$ that balances WGAN loss and RL loss in the original paper.

## Some experiment results

| lambda_wgan | Valid | Unique | Novel  | Solubility |
| :---------: | :---: | :----: | :----: | :--------: |
| 0(full RL)  | 78.21 | 93.58  | 97.51  |    0.30    |
|    0.01     | 80.01 | 95.90  | 97.79  |    0.31    |
|    0.05     | 77.04 | 97.19  | 99.24  |    0.33    |
|     0.1     | 69.51 | 96.40  | 98.93  |    0.32    |
|    0.25     | 68.90 | 95.00  | 98.88  |    0.33    |
|     0.5     | 73.81 | 96.31  | 100.00 |    0.31    |
|    0.75     | 82.60 | 95.95  | 99.50  |    0.33    |
|  1(no RL)   | 81.29 | 95.65  | 97.98  |    0.31    |

