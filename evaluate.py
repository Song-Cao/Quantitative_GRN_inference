import torch
from torch_geometric.data import DataLoader
from Model.GATConv import GATModel
from Model.graph_transformer import TransformerModel

def evaluate_model(gat_model, transformer_model, test_loader, criterion, device):
    """
    Evaluate the integrated model on test data.
    
    Args:
        gat_model (nn.Module): Trained GAT model.
        transformer_model (nn.Module): Trained Transformer model.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function for evaluation.
        device (torch.device): Device to perform evaluation on.
    
    Returns:
        float: Average loss on the test data.
    """
    gat_model.eval()
    transformer_model.eval()
    
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Forward pass through GAT
            gat_out = gat_model(data.x, data.edge_index)
            
            # Forward pass through Transformer
            transformer_out = transformer_model(gat_out)
            
            # Compute loss
            loss = criterion(transformer_out, data.y)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

if __name__ == "__main__":
    import argparse
    import pickle
    import pandas as pd
    from torch_geometric.data import Data, DataLoader
    
    parser = argparse.ArgumentParser(description="Evaluate the trained GAT-Transformer model")
    parser.add_argument("--test_data", required=True, help="Path to test data CSV")
    parser.add_argument("--test_edges", required=True, help="Path to test edge index pickle")
    parser.add_argument("--gat_model_path", required=True, help="Path to the trained GAT model")
    parser.add_argument("--transformer_model_path", required=True, help="Path to the trained Transformer model")
    parser.add_argument("--gat_input_dim", type=int, required=True, help="GAT input dimension")
    parser.add_argument("--gat_hidden_dim", type=int, default=128, help="GAT hidden dimension")
    parser.add_argument("--gat_output_dim", type=int, required=True, help="GAT output dimension")
    parser.add_argument("--transformer_heads", type=int, default=8, help="Number of attention heads in Transformer")
    parser.add_argument("--transformer_hidden_dim", type=int, default=128, help="Transformer hidden dimension")
    parser.add_argument("--transformer_layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--device", default="cuda", help="Device to evaluate the model on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load test data
    print("Loading test data...")
    test_features = torch.tensor(pd.read_csv(args.test_data).values, dtype=torch.float32)
    with open(args.test_edges, "rb") as f:
        test_edges = pickle.load(f)
    test_edge_index = torch.tensor(test_edges, dtype=torch.long)

    test_data = Data(x=test_features, edge_index=test_edge_index, y=test_features)  # Placeholder for y
    test_loader = DataLoader([test_data], batch_size=1, shuffle=False)

    # Load trained models
    print("Loading trained models...")
    gat_model = GATModel(args.gat_input_dim, args.gat_hidden_dim, args.gat_output_dim)
    transformer_model = TransformerModel(args.gat_output_dim, args.transformer_heads, args.transformer_hidden_dim, args.transformer_layers)
    
    gat_model.load_state_dict(torch.load(args.gat_model_path))
    transformer_model.load_state_dict(torch.load(args.transformer_model_path))
    
    gat_model.to(device)
    transformer_model.to(device)

    # Define evaluation criterion
    criterion = torch.nn.MSELoss()

    # Evaluate the model
    print("Evaluating the model...")
    test_loss = evaluate_model(gat_model, transformer_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
