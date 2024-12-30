# Graph-Deeplearning using GAT Model

This repository contains an implementation of Graph Attention Networks (GAT) using the Deep Graph Library (DGL). The GAT model is used for node classification tasks on various graph datasets.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [License](#license)

## Introduction

Graph Attention Networks (GAT) leverage attention mechanisms to learn the importance of neighboring nodes in a graph. This implementation supports both transductive and inductive learning settings.

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

To train the GAT model on a specific dataset, run:

```sh
python GAT/Main.py --dataset <DATASET_NAME>
```

Replace `<DATASET_NAME>` with one of the following options: `Cora`, `CiteSeer`, `Pubmed`, `PPI`.

## Datasets

The supported datasets are:
- **Cora**: Citation network dataset for node classification.
- **CiteSeer**: Citation network dataset for node classification.
- **Pubmed**: Citation network dataset for node classification.
- **PPI**: Protein-Protein Interaction dataset for multi-class node classification.

## Training

The training process varies based on the dataset:
- For transductive learning (e.g., Cora, CiteSeer, Pubmed), the model is trained on the entire graph.
- For inductive learning (e.g., PPI), the model is trained using mini-batch training.

## Evaluation

The evaluation metrics include accuracy for transductive learning and F1-score for inductive learning. The evaluation scripts are included in the `Train.py` file.

## Visualization

To visualize the learned node embeddings using UMAP, run the `umapplot.py` script:

```sh
python GAT/umapplot.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
