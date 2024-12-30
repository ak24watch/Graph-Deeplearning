import matplotlib.pyplot as plt
import umap
import torch
from NodeClassificationData import getData
from GatModel import GAT


# Load the Cora dataset
# features, labels, num_classes, train_mask, val_mask, test_mask, graph = getData("Cora")
features, labels, num_classes, train_mask, val_mask, test_mask, graph = getData("CiteSeer")

# Initialize the GAT model
model = GAT(features.shape[1], 8, num_classes, [8, 1])

# Load the trained model weights
model.load_state_dict(torch.load("gat_model_weights_citeseer.pth", weights_only=True))
model.eval()

# Get the output embeddings from the model
with torch.no_grad():
    output_embeddings = model(graph, features)

# Convert output embeddings to numpy array
output_embeddings_np = output_embeddings.numpy()

# Apply UMAP to reduce dimensionality to 2D
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(output_embeddings_np)

# Plot the UMAP embedding
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0], embedding[:, 1], c=labels.numpy(), cmap="tab20", s=5
)
plt.colorbar(scatter, label="Node Labels")
plt.title("UMAP projection of CiteSeer dataset (Post-Training)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

# Save the plot
# plt.savefig("umap_projection_cora.png")
plt.savefig("umap_projection_citeseer.png")
