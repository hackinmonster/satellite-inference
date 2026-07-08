from model import CLASSES, LandCoverNet
import torch


def test_forward_shape():
    m = LandCoverNet()
    m.eval()
    x = torch.randn(4, 3, 64, 64)
    y = m(x)
    assert y.shape == (4, len(CLASSES))
