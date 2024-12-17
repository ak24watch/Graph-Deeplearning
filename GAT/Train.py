import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score


def evaluateInductive(dataloader, model):
    """
    Evaluates the inductive model in mini-batches using F1 score.

    Args:
    - dataloader: The data loader that yields mini-batches.
    - model: The model to be evaluated.

    Returns:
    - The average F1 score over all batches.
    """
    total_score = 0
    for batch_id, batched_graph in enumerate(dataloader):
        features = batched_graph.ndata["feat"]  # Get features from the graph
        labels = batched_graph.ndata["label"]  # Get true labels

        # Evaluate the batch and get the F1 score
        score = evaluate_batch(batched_graph, features, labels, model)
        total_score += score

    # Return the average F1 score over all batches
    return total_score / (batch_id + 1)


def evaluate_batch(graph, features, labels, model):
    """
    Evaluates a single batch of data and computes the F1 score.

    Args:
    - graph: The input graph (e.g., DGL graph).
    - features: The feature matrix (node features).
    - labels: The true labels for nodes.
    - model: The model to be evaluated.

    Returns:
    - F1 score (micro average) between the predicted and true labels.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(graph, features)  # Get output from the model

        # Convert logits to binary predictions (0 or 1)
        pred = np.where(output.data.numpy() >= 0, 1, 0)

        # Calculate F1 score (micro average)
        score = f1_score(labels.data.numpy(), pred, average="micro")
        return score


def evaluateTransductive(graph, features, labels, mask, model):
    """
    Evaluates the transductive model (node classification) on a given graph.

    Args:
    - graph: The input graph (e.g., DGL graph).
    - features: The feature matrix (node features).
    - labels: The true labels for nodes.
    - mask: A mask indicating the nodes to be evaluated (e.g., validation mask).
    - model: The model to be evaluated.

    Returns:
    - Accuracy of the model on the masked nodes.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get the output logits from the model
        logits = model(graph, features)

        # Apply the mask to extract logits and labels for the validation set
        logits = logits[mask]
        labels = labels[mask]

        # Get predicted class (highest logit)
        _, indices = torch.max(logits, dim=1)

        # Calculate accuracy
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def trainTransductive(graph, features, labels, train_mask, val_mask, model, optimizer):
    """
    Trains the transductive model (node classification) using the given graph, features, and labels.

    Args:
    - graph: The input graph (e.g., DGL graph).
    - features: The feature matrix (node features).
    - labels: The true labels for nodes.
    - train_mask: A mask indicating the training nodes.
    - val_mask: A mask indicating the validation nodes.
    - model: The model to be trained.
    - optimizer: The optimizer used for training.

    Returns:
    - None (prints out training loss and validation accuracy during training).
    """
    loss_fcn = (
        nn.CrossEntropyLoss()
    )  # Cross-entropy loss for multi-class classification

    # Training loop for 100 epochs
    for epoch in range(100):
        model.train()  # Set the model to training mode
        logits = model(graph, features)  # Forward pass

        # Calculate loss using only the training mask
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        # Zero gradients, backpropagate, and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate on the validation set
        acc = evaluateTransductive(graph, features, labels, val_mask, model)

        # Print the current epoch loss and validation accuracy
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


def trainInductive(train_dataloader, val_dataloader, model):
    """
    Trains the inductive model (e.g., mini-batch learning with DGL) using the given data loaders.

    Args:
    - train_dataloader: The data loader for training.
    - val_dataloader: The data loader for validation.
    - model: The model to be trained.

    Returns:
    - None (prints out training loss and F1 score on validation set).
    """
    loss_fcn = (
        nn.BCEWithLogitsLoss()
    )  # Binary Cross-Entropy with logits (for binary classification)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-3, weight_decay=0
    )  # Adam optimizer

    # Training loop for 100 epochs
    for epoch in range(100):
        model.train()  # Set the model to training mode
        total_loss = 0

        # Mini-batch loop for training
        for batch_id, batched_graph in enumerate(train_dataloader):
            features = batched_graph.ndata["feat"]  # Get features
            labels = batched_graph.ndata["label"]  # Get labels

            logits = model(batched_graph, features)  # Forward pass
            loss = loss_fcn(logits, labels)  # Calculate loss

            # Backpropagation and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the average loss for the current epoch
        print("Epoch {:05d} | Loss {:.4f} |".format(epoch, total_loss / (batch_id + 1)))

        # Every 5 epochs, evaluate the model on the validation set
        if (epoch + 1) % 5 == 0:
            avg_score = evaluateInductive(val_dataloader, model)  # F1-score
            print(
                "                            Acc. (F1-score) {:.4f} ".format(avg_score)
            )
