import argparse
import torch
from NodeClassificationData import getData  # Data loading utility
from GatModel import GAT, GATppi  # GAT models for node classification
from Train import (
    trainTransductive,
    evaluateTransductive,
    trainInductive,
)  # Training and evaluation functions

# Argument parsing to handle different datasets
parser = argparse.ArgumentParser(description="GAT for node classification")
parser.add_argument(
    "--dataset",
    type=str,
    default="Cora",  # Default dataset is "Cora"
    choices=[
        "Cora",
        "CiteSeer",
        "Pumbed",
        "PPI",
    ],  # Limit options to available datasets
    help="Dataset name ('Cora', 'CiteSeer', 'Pumbed', 'PPI').",
)
args = parser.parse_args()

# Print the type of GNN model used
print("Training with Graph Attention Network (GAT) for node classification.")

# Dataset-specific loading logic
if args.dataset == "Cora":
    features, labels, num_classes, train_mask, val_mask, test_mask, graph = getData(
        "Cora"
    )
elif args.dataset == "CiteSeer":
    features, labels, num_classes, train_mask, val_mask, test_mask, graph = getData(
        "CiteSeer"
    )
elif args.dataset == "Pumbed":
    features, labels, num_classes, train_mask, val_mask, test_mask, graph = getData(
        "Pumbed"
    )
elif args.dataset == "PPI":
    batch_train_data, batch_valid_data, batch_test_data, features, in_size, out_size = (
        getData("PPI")
    )
else:
    raise ValueError(
        f"Unknown dataset: {args.dataset}"
    )  # Raise error if an unknown dataset is passed

# Model and training setup based on the dataset
if args.dataset in ["Cora", "CiteSeer", "Pumbed"]:
    # Initialize the GAT model for transductive learning (Cora, CiteSeer, Pumbed)
    model = GAT(
        features.shape[1], 8, num_classes, [8, 1]
    )  # GAT model with 8 hidden units and [8, 1] layers
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-3, weight_decay=5e-4
    )  # Adam optimizer with weight decay

    # Train the model in a transductive setting
    trainTransductive(graph, features, labels, train_mask, val_mask, model, optimizer)

    # Evaluate the model on the test set
    test_accuracy = evaluateTransductive(graph, features, labels, test_mask, model)
    print(f"Test accuracy: {test_accuracy:.4f}")

else:
    # Initialize the GAT model for inductive learning (PPI)
    model = GATppi(
        in_size, 64, out_size, [4, 4, 6],residual=True
    )  # GATppi model with residual connections

    # Train the model in an inductive setting (using mini-batch training)
    trainInductive(batch_train_data, batch_valid_data, model)

torch.save(model.state_dict(), "gat_model_weights.pth")