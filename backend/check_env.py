import torch
import torch_geometric
import sklearn

print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("PyG:", torch_geometric.__version__)
print("Sklearn:", sklearn.__version__)

try:
    import fastapi
    print("FastAPI:", fastapi.__version__)
except ImportError:
    print("FastAPI: Not installed")

try:
    import uvicorn
    print("Uvicorn:", uvicorn.__version__)
except ImportError:
    print("Uvicorn: Not installed")

