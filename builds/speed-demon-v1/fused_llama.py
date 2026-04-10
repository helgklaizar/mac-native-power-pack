import mlx.core as mx
import mlx.nn as nn
from fused_ops_mlx.infra.ops.rms_norm import fused_rms_norm

class FusedRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        # We use our custom C++ (Metal) fused kernel instead of standard logic
        return fused_rms_norm(x, self.weight, self.eps)

def patch_model_with_mnpp(model):
    """
    Replaces standard RMSNorm layers with MNPP Fused kernels.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.RMSNorm):
            # Dynamic replacement
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if not parent_name else model.get_module(parent_name)
            
            # Create our fused version
            fused_norm = FusedRMSNorm(module.weight.shape[0], module.eps)
            fused_norm.weight = module.weight # Copy weights
            
            setattr(parent, attr_name, fused_norm)
            print(f"✅ Patched {name} with MNPP Fused Kernels")

if __name__ == "__main__":
    # Example: Pseudo-model patching
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.RMSNorm(4096)
            self.norm2 = nn.RMSNorm(4096)
    
    model = MockModel()
    print("Before patching:", type(model.norm1))
    patch_model_with_mnpp(model)
    print("After patching:", type(model.norm1))
