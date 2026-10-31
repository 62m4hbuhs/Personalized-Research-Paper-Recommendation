import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


print("=" * 70)
print("GRAPHSAGE GPU TEST")
print("=" * 70)


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ------------------------------------------------------------
# SMALL TEST GRAPH
# ------------------------------------------------------------

# 1,000 nodes with 768-dimensional features
num_nodes = 1000
num_features = 768

x = torch.randn(
    num_nodes,
    num_features
)

# Simple bidirectional chain-like graph
src = torch.arange(
    0,
    num_nodes - 1
)

dst = src + 1

edge_index = torch.cat(
    [
        torch.stack([src, dst]),
        torch.stack([dst, src])
    ],
    dim=1
)

data = Data(
    x=x,
    edge_index=edge_index
)

data = data.to(device)

print(f"\nNodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}")


# ------------------------------------------------------------
# GRAPHSAGE MODEL
# ------------------------------------------------------------

class GraphSAGE(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ------------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------------

model = GraphSAGE(
    in_channels=768,
    hidden_channels=256,
    out_channels=128
).to(device)

print("\nModel:")
print(model)


# ------------------------------------------------------------
# FORWARD PASS
# ------------------------------------------------------------

print("\nRunning forward pass...")

with torch.no_grad():

    output = model(
        data.x,
        data.edge_index
    )


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GPU TEST COMPLETE")
print("=" * 70)

print(
    f"Input shape:  {tuple(data.x.shape)}"
)

print(
    f"Output shape: {tuple(output.shape)}"
)

print(
    f"Output device: {output.device}"
)

print(
    f"Output dtype:  {output.dtype}"
)

print(
    f"GPU memory allocated: "
    f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
)

print("\nGraphSAGE GPU test PASSED.")