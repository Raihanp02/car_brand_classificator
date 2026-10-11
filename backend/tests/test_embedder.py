# backend/tests/test_embedder.py
import torch
def test_batch_equals_singles():
    from app.embedder import CarEmbedder
    e = CarEmbedder(method="cls_proj256", checkpoint=None)  # random-proj baseline for CI
    from PIL import Image
    imgs = [Image.new("RGB", (224, 224), (i * 20, 0, 0)) for i in range(3)]
    batched = e.embed_batch(imgs)
    singles = torch.stack([e.embed_batch([im]) for im in imgs]).squeeze(1)
    assert torch.allclose(batched, singles, atol=1e-5)

def test_method_mismatch_refuses():
    from app.embedder import load_index
    import pytest
    with pytest.raises(ValueError, match="method/dim mismatch"):
        load_index("models/index_cls_patch_concat.faiss", expected_method="cls_proj256", expected_dim=256)
