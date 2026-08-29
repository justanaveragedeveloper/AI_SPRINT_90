"""
Complete test suite for transforms and dataloader using pytest.
Covers happy path, edge cases, defensive checks, and reproducibility.
"""

from __future__ import annotations

import random

import pytest
import torch
from dataloader import DataLoader, SimpleImageDataset
from torch import Tensor
from transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip


# ---------- Helpers ----------
def random_image(c: int = 3, h: int = 32, w: int = 32) -> Tensor:
    return torch.rand(c, h, w)


# ---------- Custom non-commutative transforms for testing ordering ----------
class AddOne:
    def __call__(self, img: Tensor) -> Tensor:
        return img + 1.0


class MultiplyTwo:
    def __call__(self, img: Tensor) -> Tensor:
        return img * 2.0


# ---------- Normalize ----------
def test_normalize_basic() -> None:
    norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = torch.ones(3, 2, 2) * 0.5
    out = norm(img)
    expected = torch.zeros(3, 2, 2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_normalize_channel_mismatch() -> None:
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = torch.rand(1, 28, 28)
    with pytest.raises(ValueError, match="Channel mismatch"):
        norm(img)


def test_normalize_wrong_ndim() -> None:
    norm = Normalize(mean=[0.5], std=[0.5])
    img = torch.rand(4, 4)
    with pytest.raises(ValueError, match="Expected 3D tensor"):
        norm(img)


def test_normalize_zero_std() -> None:
    norm = Normalize(mean=[0.0], std=[0.0], eps=1e-6)
    img = torch.ones(1, 2, 2) * 5.0
    out = norm(img)
    expected = (5.0 - 0.0) / (0.0 + 1e-6)
    assert torch.allclose(out, torch.tensor(expected))


def test_normalize_negative_eps() -> None:
    with pytest.raises(ValueError, match="eps must be non-negative"):
        Normalize([0.0], [1.0], eps=-1e-6)


def test_normalize_negative_std() -> None:
    with pytest.raises(ValueError, match="std values must be non-negative"):
        Normalize([0.0], [-0.5])


def test_normalize_integer_input() -> None:
    """Check that integer images are converted to float and normalized correctly."""
    norm = Normalize(mean=[0.5], std=[0.5])
    img = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int64)  # shape (1,2,2)
    out = norm(img)
    # After conversion to float, (1-0.5)/0.5 = 1, (2-0.5)/0.5 = 3, etc.
    expected = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    assert torch.allclose(out, expected)


# ---------- RandomHorizontalFlip ----------
def test_random_horizontal_flip_deterministic() -> None:
    flip = RandomHorizontalFlip(p=1.0)
    img = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    out = flip(img)
    expected = torch.tensor([[[2.0, 1.0], [4.0, 3.0]]])
    assert torch.equal(out, expected)


def test_random_horizontal_flip_never() -> None:
    flip = RandomHorizontalFlip(p=0.0)
    img = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    out = flip(img)
    assert torch.equal(out, img)


def test_random_horizontal_flip_probability() -> None:
    rng = random.Random(42)
    flip = RandomHorizontalFlip(p=0.5, rng=rng)
    pattern = torch.arange(10).float().view(1, 1, 10)
    flipped_count = 0
    n_trials = 200
    for _ in range(n_trials):
        out = flip(pattern)
        if torch.equal(out, torch.flip(pattern, dims=[2])):
            flipped_count += 1
    assert 70 < flipped_count < 130


# ---------- RandomCrop ----------
def test_random_crop_without_padding() -> None:
    crop = RandomCrop(crop_size=(2, 2), padding=0, rng=random.Random(123))
    img = torch.arange(16).float().view(1, 4, 4)
    out = crop(img)
    assert out.shape == (1, 2, 2)
    # Verify all values in the crop are from the original image.
    assert torch.all((out >= 0) & (out <= 15))
    # Verify that the crop is a contiguous sub-window.
    found = False
    for start_h in range(3):
        for start_w in range(3):
            if torch.equal(img[:, start_h:start_h+2, start_w:start_w+2], out):
                found = True
                break
        if found:
            break
    assert found, "Crop is not a valid sub-window of the original image."


def test_random_crop_with_padding() -> None:
    crop = RandomCrop(crop_size=(4, 4), padding=2, rng=random.Random(42))
    img = torch.ones(1, 2, 2) * 5.0
    out = crop(img)
    assert out.shape == (1, 4, 4)
    assert torch.all((out == 5.0) | (out == 0.0))


def test_random_crop_invalid_size() -> None:
    crop = RandomCrop(crop_size=(10, 10), padding=0)
    img = torch.rand(1, 4, 4)
    with pytest.raises(ValueError, match="exceeds image dimensions"):
        crop(img)


def test_random_crop_negative_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        RandomCrop(crop_size=(-1, 2))


def test_random_crop_malformed_tuple() -> None:
    with pytest.raises(ValueError, match="must be an int or a tuple of two ints"):
        RandomCrop(crop_size=(2, 3, 4))


def test_random_crop_tuple_non_integer() -> None:
    with pytest.raises(ValueError, match="contain two integers"):
        RandomCrop(crop_size=(2.5, 3.5))


# ---------- Compose ----------
def test_compose_ordering() -> None:
    """Check that Compose applies transforms in the correct order using non-commutative transforms."""
    # AddOne then MultiplyTwo => (x+1)*2
    compose1 = Compose([AddOne(), MultiplyTwo()])
    # MultiplyTwo then AddOne => x*2 + 1
    compose2 = Compose([MultiplyTwo(), AddOne()])

    x = torch.tensor([1.0, 2.0, 3.0])  # shape (3,)
    # We need to test on 3D images; we can use 1x1x3 tensors for simplicity.
    x_img = x.view(1, 1, 3)

    out1 = compose1(x_img)
    out2 = compose2(x_img)

    # (x+1)*2 vs x*2+1 are different for most x.
    assert not torch.equal(out1, out2)
    # Check specific values.
    expected1 = (x + 1) * 2
    expected2 = x * 2 + 1
    assert torch.allclose(out1.view(-1), expected1)
    assert torch.allclose(out2.view(-1), expected2)


def test_compose_empty() -> None:
    img = torch.rand(3, 4, 4)
    out = Compose([])(img)
    assert torch.equal(out, img)


# ---------- Dataset ----------
def test_simple_image_dataset_basic() -> None:
    images = [torch.rand(1, 2, 2) for _ in range(5)]
    labels = [0, 1, 0, 1, 0]
    ds = SimpleImageDataset(images, labels)
    assert len(ds) == 5
    img, lab = ds[2]
    assert lab == 0
    assert isinstance(img, Tensor)
    assert img.shape == (1, 2, 2)


def test_dataset_transform_integration() -> None:
    transform = Compose([Normalize(mean=[0.5], std=[0.5])])
    img = torch.ones(1, 2, 2) * 0.5
    ds = SimpleImageDataset([img], [0], transform=transform)
    out, _ = ds[0]
    expected = torch.zeros(1, 2, 2)
    assert torch.allclose(out, expected)


def test_dataset_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        SimpleImageDataset([torch.rand(1, 2, 2)], [0, 1])


def test_dataset_non_3d_image() -> None:
    img = torch.rand(2, 2)
    with pytest.raises(ValueError, match="must have shape"):
        SimpleImageDataset([img], [0])


def test_dataset_shape_validation() -> None:
    images = [torch.rand(3, 32, 32), torch.rand(3, 64, 64)]
    with pytest.raises(ValueError, match="expected"):
        SimpleImageDataset(images, [0, 1], validate_shapes=True)

    ds = SimpleImageDataset(images, [0, 1], validate_shapes=False)
    assert len(ds) == 2


def test_dataset_channel_consistency() -> None:
    images = [torch.rand(3, 32, 32), torch.rand(1, 32, 32)]
    with pytest.raises(ValueError, match="expected"):
        SimpleImageDataset(images, [0, 1], validate_shapes=True)


# ---------- DataLoader ----------
def test_dataloader_batching_drop_last() -> None:
    images = [torch.rand(1, 2, 2) for _ in range(5)]
    labels = [0, 1, 0, 1, 0]
    ds = SimpleImageDataset(images, labels)
    loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=True)
    assert len(loader) == 2
    batches = list(loader)
    assert len(batches) == 2
    for batch_imgs, batch_labs in batches:
        assert batch_imgs.shape[0] == 2
        assert batch_labs.shape[0] == 2


def test_dataloader_no_drop_last() -> None:
    images = [torch.rand(1, 2, 2) for _ in range(5)]
    labels = [0, 1, 0, 1, 0]
    ds = SimpleImageDataset(images, labels)
    loader = DataLoader(ds, batch_size=2, shuffle=False, drop_last=False)
    assert len(loader) == 3
    batches = list(loader)
    assert batches[0][0].shape[0] == 2
    assert batches[1][0].shape[0] == 2
    assert batches[2][0].shape[0] == 1


def test_dataloader_shuffle_preserves_alignment() -> None:
    images = [torch.full((1, 2, 2), float(i)) for i in range(5)]
    labels = [10, 20, 30, 40, 50]
    ds = SimpleImageDataset(images, labels, validate_shapes=False)
    rng = random.Random(42)
    loader = DataLoader(ds, batch_size=5, shuffle=True, rng=rng)
    batch_imgs, batch_labels = next(iter(loader))

    for img, label in zip(batch_imgs, batch_labels):
        img_value = int(img[0, 0, 0].item())
        expected_label = [10, 20, 30, 40, 50][img_value]
        assert label.item() == expected_label


def test_dataloader_len() -> None:
    ds = SimpleImageDataset([torch.rand(1, 2, 2)] * 10, [0] * 10)
    loader = DataLoader(ds, batch_size=3, drop_last=False)
    assert len(loader) == 4
    loader2 = DataLoader(ds, batch_size=3, drop_last=True)
    assert len(loader2) == 3


def test_dataloader_empty_dataset() -> None:
    ds = SimpleImageDataset([], [])
    loader = DataLoader(ds, batch_size=2)
    assert len(loader) == 0
    batches = list(loader)
    assert len(batches) == 0


def test_dataloader_invalid_batch_size() -> None:
    ds = SimpleImageDataset([torch.rand(1, 2, 2)], [0])
    with pytest.raises(ValueError, match="positive integer"):
        DataLoader(ds, batch_size=0)
    with pytest.raises(ValueError, match="positive integer"):
        DataLoader(ds, batch_size=-1)


def test_dataloader_shape_mismatch() -> None:
    images = [torch.rand(1, 2, 2), torch.rand(1, 3, 3), torch.rand(1, 2, 2)]
    labels = [0, 1, 0]
    ds = SimpleImageDataset(images, labels, validate_shapes=False)
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    with pytest.raises(ValueError, match="Inconsistent shapes"):
        list(loader)


def test_dataloader_channel_mismatch() -> None:
    images = [torch.rand(3, 4, 4), torch.rand(1, 4, 4)]
    labels = [0, 1]
    ds = SimpleImageDataset(images, labels, validate_shapes=False)
    loader = DataLoader(ds, batch_size=2)
    with pytest.raises(ValueError, match="Inconsistent shapes"):
        list(loader)


def test_reproducibility_with_seed() -> None:
    images = [torch.rand(1, 4, 4) for _ in range(10)]
    labels = list(range(10))
    ds = SimpleImageDataset(images, labels)

    rng1 = random.Random(42)
    loader1 = DataLoader(ds, batch_size=3, shuffle=True, rng=rng1)
    batches1 = list(loader1)

    rng2 = random.Random(42)
    loader2 = DataLoader(ds, batch_size=3, shuffle=True, rng=rng2)
    batches2 = list(loader2)

    for (imgs1, labs1), (imgs2, labs2) in zip(batches1, batches2):
        assert torch.equal(imgs1, imgs2)
        assert torch.equal(labs1, labs2)


def test_shuffle_permutation() -> None:
    images = [torch.full((1, 1, 1), float(i)) for i in range(10)]
    labels = list(range(10))
    ds = SimpleImageDataset(images, labels)
    loader = DataLoader(ds, batch_size=10, shuffle=True, rng=random.Random(42))
    _, batch_labels = next(iter(loader))
    assert torch.allclose(batch_labels.sort().values, torch.tensor(list(range(10))))
    assert not torch.equal(batch_labels, torch.tensor(list(range(10))))