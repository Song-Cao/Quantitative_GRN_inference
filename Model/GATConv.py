import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    """
    Graph Attention Network for learning local dependencies in spatial single-cell data with GRN input.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass of the GAT model."""
        x = self.gat1(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)
        x = self.gat2(x, edge_index, edge_weight=edge_weight)
        return x

def build_gat_model(input_dim, hidden_dim, output_dim):
    """
    Constructs the GAT model with specified dimensions.
    
    Args:
        input_dim (int): Input feature dimensionality.
        hidden_dim (int): Hidden layer dimensionality.
        output_dim (int): Output feature dimensionality.
    
    Returns:
        nn.Module: Instantiated GAT model.
    """
    model = GATModel(
        in_channels=input_dim,
        hidden_channels=hidden_dim,
        out_channels=output_dim,
    )
    return model

if __name__ == "__main__":
    import argparse
    import pickle
    import pandas as pd
    from torch_geometric.data import Data, DataLoader

    parser = argparse.ArgumentParser(description="Train GAT model with GRN input for spatial single-cell data")
    parser.add_argument("--train_data", required=True, help="Path to training data CSV")
    parser.add_argument("--grn_file", required=True, help="Path to GRN adjacency matrix numpy file")
    parser.add_argument("--output", required=True, help="Path to save the GAT model")
    parser.add_argument("--input_dim", type=int, required=True, help="Input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden feature dimension")
    parser.add_argument("--output_dim", type=int, required=True, help="Output feature dimension")
    args = parser.parse_args()

    # Load training data and GRN adjacency matrix
    print("Loading training data and GRN adjacency matrix...")
    train_features = torch.tensor(pd.read_csv(args.train_data).values, dtype=torch.float32)
    grn_adjacency_matrix = torch.tensor(np.load(args.grn_file), dtype=torch.float32)
    
    # Convert adjacency matrix to edge index and edge weights
    edge_index = (grn_adjacency_matrix > 0).nonzero(as_tuple=False).t()
    edge_weight = grn_adjacency_matrix[edge_index[0], edge_index[1]]

    # Initialize GAT model
    print("Building GAT model...")
    model = build_gat_model(args.input_dim, args.hidden_dim, args.output_dim)
    print(model)

    # Create a simple dataset and dataloader
    data = Data(x=train_features, edge_index=edge_index, edge_weight=edge_weight)
    dataloader = DataLoader([data], batch_size=1, shuffle=True)

    # Save the initialized model
    torch.save(model.state_dict(), args.output)
    print(f"GAT model saved to {args.output}")
