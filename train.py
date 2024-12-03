import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from Model.GATConv import GATModel
from Model.graph_transformer import TransformerModel

def train_model(gat_model, transformer_model, train_loader, val_loader, num_epochs, gat_lr, transformer_lr, device):
    """
    Train the integrated GAT and Transformer model pipeline with GRN support.
    
    Args:
        gat_model (nn.Module): Graph Attention Network model.
        transformer_model (nn.Module): Transformer model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        gat_lr (float): Learning rate for GAT.
        transformer_lr (float): Learning rate for Transformer.
        device (torch.device): Device to train the model on.
    
    Returns:
        None
    """
    gat_model.to(device)
    transformer_model.to(device)
    
    # Optimizers
    gat_optimizer = optim.Adam(gat_model.parameters(), lr=gat_lr)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=transformer_lr)
    
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        gat_model.train()
        transformer_model.train()
        
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            
            # Forward pass through GAT
            gat_out = gat_model(data.x, data.edge_index, data.edge_weight)
            
            # Forward pass through Transformer
            transformer_out = transformer_model(gat_out)
            
            # Compute loss
            loss = criterion(transformer_out, data.y)
            
            # Backward pass and optimization
            gat_optimizer.zero_grad()
            transformer_optimizer.zero_grad()
            loss.backward()
            gat_optimizer.step()
            transformer_optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
        
        # Validation
        val_loss = evaluate_model(gat_model, transformer_model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss}")

def evaluate_model(gat_model, transformer_model, loader, criterion, device):
    """
    Evaluate the integrated model with GRN support.
    
    Args:
        gat_model (nn.Module): GAT model.
        transformer_model (nn.Module): Transformer model.
        loader (DataLoader): DataLoader for evaluation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.
    
    Returns:
        float: Average loss on the evaluation data.
    """
    gat_model.eval()
    transformer_model.eval()
    
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass
            gat_out = gat_model(data.x, data.edge_index, data.edge_weight)
            transformer_out = transformer_model(gat_out)
            
            # Compute loss
            loss = criterion(transformer_out, data.y)
            total_loss += loss.item()
    
    return total_loss / len(loader)

if __name__ == "__main__":
    import argparse
    import pickle
    import pandas as pd
    from torch_geometric.data import Data, DataLoader
    
    parser = argparse.ArgumentParser(description="Train integrated GAT and Transformer model with GRN support")
    parser.add_argument("--train_data", required=True, help="Path to training data CSV")
    parser.add_argument("--train_grn", required=True, help="Path to GRN adjacency matrix numpy file for training")
    parser.add_argument("--val_data", required=True, help="Path to validation data CSV")
    parser.add_argument("--val_grn", required=True, help="Path to GRN adjacency matrix numpy file for validation")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--gat_lr", type=float, default=1e-3, help="Learning rate for GAT")
    parser.add_argument("--transformer_lr", type=float, default=1e-4, help="Learning rate for Transformer")
    parser.add_argument("--gat_input_dim", type=int, required=True, help="GAT input dimension")
    parser.add_argument("--gat_hidden_dim", type=int, default=128, help="GAT hidden dimension")
    parser.add_argument("--gat_output_dim", type=int, required=True, help="GAT output dimension")
    parser.add_argument("--transformer_heads", type=int, default=8, help="Number of attention heads in Transformer")
    parser.add_argument("--transformer_hidden_dim", type=int, default=128, help="Transformer hidden dimension")
    parser.add_argument("--transformer_layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--device", default="cuda", help="Device to train the model on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load datasets and GRNs
    print("Loading training and validation data with GRN adjacency matrices...")
    train_features = torch.tensor(pd.read_csv(args.train_data).values, dtype=torch.float32)
    train_grn = torch.tensor(np.load(args.train_grn), dtype=torch.float32)
    train_edge_index = (train_grn > 0).nonzero(as_tuple=False).t()
    train_edge_weight = train_grn[train_edge_index[0], train_edge_index[1]]

    val_features = torch.tensor(pd.read_csv(args.val_data).values, dtype=torch.float32)
    val_grn = torch.tensor(np.load(args.val_grn), dtype=torch.float32)
    val_edge_index = (val_grn > 0).nonzero(as_tuple=False).t()
    val_edge_weight = val_grn[val_edge_index[0], val_edge_index[1]]

    # Create DataLoaders
    train_data = Data(x=train_features, edge_index=train_edge_index, edge_weight=train_edge_weight, y=train_features)
    val_data = Data(x=val_features, edge_index=val_edge_index, edge_weight=val_edge_weight, y=val_features)
    train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=1, shuffle=False)

    # Initialize models
    print("Initializing models...")
    gat_model = GATModel(args.gat_input_dim, args.gat_hidden_dim, args.gat_output_dim)
    transformer_model = TransformerModel(args.gat_output_dim, args.transformer_heads, args.transformer_hidden_dim, args.transformer_layers)

    # Train the models
    print("Starting training...")
    train_model(
        gat_model, transformer_model, train_loader, val_loader,
        args.num_epochs, args.gat_lr, args.transformer_lr, device
    )
