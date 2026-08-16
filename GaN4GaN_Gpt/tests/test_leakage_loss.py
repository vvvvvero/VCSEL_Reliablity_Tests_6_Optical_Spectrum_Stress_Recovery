import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("_07_losses_test", ROOT / "07_losses.py")
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)


def test_leakage_confidence_weights_reduce_near_floor_emphasis():
    x_true = torch.tensor([[[0.0, 0.2]], [[0.0, 2.0]]], dtype=torch.float32)
    feature_mask = torch.ones_like(x_true, dtype=torch.bool)
    floors = torch.tensor([0.1, 0.1], dtype=torch.float32)

    weights = losses.leakage_confidence_weights(x_true, feature_mask, floors, floor_margin=0.2)
    assert weights.shape == x_true.shape
    assert torch.all(weights[:, :, 0] <= weights[:, :, 1])
    assert torch.all(weights >= 0.2)
    assert torch.all(weights <= 1.0)
