import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    """
    Transformer model for global dependency modeling and refining GAT embeddings.
    """
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """Forward pass of the Transformer model."""
        x = self.embedding(x)  # Project to hidden dimension
        x = self.transformer(x)  # Apply Transformer layers
        x = self.output_layer(x)  # Map back to input dimension
        return x

def build_transformer_model(input_dim, num_heads, hidden_dim, num_layers):
    """
    Constructs the Transformer model with specified dimensions.
    
    Args:
        input_dim (int): Input feature dimensionality.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden layer dimensionality.
        num_layers (int): Number of Transformer encoder layers.
    
    Returns:
        nn.Module: Instantiated Transformer model.
    """
    model = TransformerModel(
        input_dim=input_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Transformer model for spatial single-cell data")
    parser.add_argument("--input_dim", type=int, required=True, help="Input feature dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden feature dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer encoder layers")
    parser.add_argument("--output", required=True, help="Path to save the Transformer model")
    args = parser.parse_args()

    # Initialize Transformer model
    print("Building Transformer model...")
    model = build_transformer_model(
        input_dim=args.input_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    print(model)

    # Save the initialized model
    torch.save(model.state_dict(), args.output)
    print(f"Transformer model saved to {args.output}")
