import dgl
import dgl.data
# from dgl import AddSelfLoop


def getData(name):
    """
    Loads different graph datasets for node classification tasks.

    This function supports the following datasets:
    - 'Cora': Citation network dataset for node classification.
    - 'CiteSeer': Citation network dataset for node classification.
    - 'Pubmed': Citation network dataset for node classification.
    - 'PPI': Protein-Protein Interaction dataset for multi-class node classification.

    Args:
        name (str): The name of the dataset to load ('Cora', 'CiteSeer', 'Pubmed', 'PPI').

    Returns:
        Depending on the dataset:
            - For 'Cora', 'CiteSeer', and 'Pubmed': Features, labels, number of classes,
              training/validation/test masks, and the graph object.
            - For 'PPI': Batched graph dataloaders for training, validation, and testing,
              features, input size, and output size.
    """
    if name == "Cora":
        # Load Cora dataset with self-loops
        dataset = dgl.data.CoraGraphDataset()
        graph = dataset[0]
        num_classes = dataset.num_classes
        features = graph.ndata["feat"]
        labels = graph.ndata["label"]
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]

        return features, labels, num_classes, train_mask, val_mask, test_mask, graph

    elif name == "CiteSeer":
        # Load CiteSeer dataset
        dataset = dgl.data.CiteseerGraphDataset()
        graph = dataset[0]
        num_classes = dataset.num_classes
        features = graph.ndata["feat"]
        labels = graph.ndata["label"]
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]

        return features, labels, num_classes, train_mask, val_mask, test_mask, graph

    elif name == "Pubmed":
        # Load Pubmed dataset
        dataset = dgl.data.PubmedGraphDataset()
        graph = dataset[0]
        num_classes = dataset.num_classes
        features = graph.ndata["feat"]
        labels = graph.ndata["label"]
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]

        return features, labels, num_classes, train_mask, val_mask, test_mask, graph

    elif name == "PPI":
        # Load PPI dataset with batched graphs
        train_data = dgl.data.PPIDataset(mode="train")
        valid_data = dgl.data.PPIDataset(mode="valid")
        test_data = dgl.data.PPIDataset(mode="test")

        # Create batched graph dataloaders
        batched_train_graphs = dgl.dataloading.GraphDataLoader(
            train_data, batch_size=2, shuffle=True
        )
        batched_valid_graphs = dgl.dataloading.GraphDataLoader(
            valid_data, batch_size=2, shuffle=True
        )
        batched_test_graphs = dgl.dataloading.GraphDataLoader(
            test_data, batch_size=2, shuffle=True
        )

        # Extract features and the input/output size
        features = train_data[0].ndata["feat"]
        in_size = features.shape[1]
        out_size = train_data.num_classes

        return (
            batched_train_graphs,
            batched_valid_graphs,
            batched_test_graphs,
            features,
            in_size,
            out_size,
        )
