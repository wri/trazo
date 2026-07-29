"""Step 3 fine-tuning: LoRA adapters, UPGD, and model merging.

These are the properties that were silently wrong in the experiment code this
package is derived from, so they are asserted rather than assumed.
"""

import pytest

pytestmark = pytest.mark.heavy

torch = pytest.importorskip("torch", reason="needs the pt3 extra")


@pytest.fixture
def tiny_unet():
    smp = pytest.importorskip("segmentation_models_pytorch", reason="needs the pt3 extra")
    return smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=8, classes=3)


# ---------------------------------------------------------------- LoRA


def test_inject_lora_wraps_layers_and_freezes_base(tiny_unet):
    from trazo.pt3_finetune.lora import inject_lora, mark_only_lora_as_trainable

    total_before = sum(p.numel() for p in tiny_unet.parameters())
    inject_lora(tiny_unet, r=4, lora_alpha=1.0, lora_dropout=0.1)
    trainable = mark_only_lora_as_trainable(tiny_unet, bias="none")

    assert trainable > 0, "LoRA injection produced no trainable parameters"
    assert trainable < total_before * 0.5, "LoRA should train a small fraction of weights"
    for name, param in tiny_unet.named_parameters():
        if "lora_" not in name:
            assert not param.requires_grad, f"base weight {name} is still trainable"


def test_lora_starts_as_identity(tiny_unet):
    """lora_B initializes to zero, so the wrapped model must start unchanged."""
    from trazo.pt3_finetune.lora import inject_lora

    x = torch.randn(1, 8, 64, 64)
    tiny_unet.eval()
    with torch.no_grad():
        before = tiny_unet(x)

    inject_lora(tiny_unet, r=4, lora_alpha=1.0)
    tiny_unet.eval()
    with torch.no_grad():
        after = tiny_unet(x)

    assert torch.allclose(before, after, atol=1e-5), (
        "LoRA changed the model output before any training step"
    )


def test_lora_does_not_duplicate_base_parameters(tiny_unet):
    """The wrapper must not re-register the wrapped conv's parameters."""
    from trazo.pt3_finetune.lora import inject_lora

    inject_lora(tiny_unet, r=4)
    ids = [id(p) for p in tiny_unet.parameters()]
    assert len(ids) == len(set(ids)), "a parameter is registered more than once"


def test_lora_dropout_is_applied(tiny_unet):
    """Dropout was accepted and ignored in the original implementation."""
    from trazo.pt3_finetune.lora import ConvLoRA, inject_lora

    inject_lora(tiny_unet, r=4, lora_dropout=0.5)
    wrappers = [m for m in tiny_unet.modules() if isinstance(m, ConvLoRA)]
    assert wrappers
    assert any(isinstance(w.lora_dropout, torch.nn.Dropout) for w in wrappers)


def test_inject_lora_raises_when_nothing_wrapped():
    from trazo.pt3_finetune.lora import inject_lora

    empty = torch.nn.Module()
    with pytest.raises(RuntimeError, match="wrapped no layers"):
        inject_lora(empty, r=4)


# ---------------------------------------------------------------- UPGD


def test_upgd_step_changes_weights_and_stays_finite():
    from trazo.pt3_finetune.optimizers import UPGD

    layer = torch.nn.Linear(8, 4)
    before = layer.weight.detach().clone()
    opt = UPGD(layer.parameters(), lr=1e-2)

    loss = layer(torch.randn(16, 8)).pow(2).mean()
    loss.backward()
    opt.step()

    assert torch.isfinite(layer.weight).all(), "UPGD produced non-finite weights"
    assert not torch.equal(before, layer.weight.detach()), "UPGD did not update anything"


def test_upgd_survives_zero_gradients():
    """A zero global utility maximum used to divide by zero and yield NaNs."""
    from trazo.pt3_finetune.optimizers import UPGD

    layer = torch.nn.Linear(4, 4)
    opt = UPGD(layer.parameters(), lr=1e-3, sigma=0.0)
    layer.weight.grad = torch.zeros_like(layer.weight)
    layer.bias.grad = torch.zeros_like(layer.bias)

    opt.step()
    assert torch.isfinite(layer.weight).all()


def test_upgd_skips_frozen_parameters():
    from trazo.pt3_finetune.optimizers import UPGD

    layer = torch.nn.Linear(4, 4)
    layer.bias.requires_grad = False
    frozen = layer.bias.detach().clone()

    opt = UPGD([p for p in layer.parameters()], lr=1e-2)
    layer(torch.randn(8, 4)).sum().backward()
    opt.step()

    assert torch.equal(frozen, layer.bias.detach()), "UPGD updated a frozen parameter"


# ---------------------------------------------------------------- merging


@pytest.fixture
def base_and_finetuned(tmp_path):
    base = {"w": torch.zeros(4, 4), "b": torch.zeros(4), "num_batches_tracked": torch.tensor(3)}
    ft1 = {"w": torch.full((4, 4), 2.0), "b": torch.full((4,), 1.0),
           "num_batches_tracked": torch.tensor(9)}
    ft2 = {"w": torch.full((4, 4), -3.0), "b": torch.full((4,), 0.5),
           "num_batches_tracked": torch.tensor(9)}
    paths = []
    for name, state in (("base", base), ("ft1", ft1), ("ft2", ft2)):
        p = tmp_path / f"{name}.ckpt"
        torch.save({"state_dict": state}, p)
        paths.append(p)
    return paths


def test_task_vector_is_the_delta(base_and_finetuned):
    from trazo.pt3_finetune.merge import TaskVector

    base, ft1, _ = base_and_finetuned
    tv = TaskVector(base, ft1)
    assert torch.allclose(tv.vector["w"], torch.full((4, 4), 2.0))
    assert "num_batches_tracked" not in tv.vector, "integer buffers must be excluded"


def test_magmax_keeps_largest_magnitude(base_and_finetuned):
    from trazo.pt3_finetune.merge import TaskVector, magmax

    base, ft1, ft2 = base_and_finetuned
    merged = magmax([TaskVector(base, ft1), TaskVector(base, ft2)])

    # w: |-3| > |2| so the negative delta wins; b: 1.0 > 0.5 so the first wins.
    assert torch.allclose(merged.vector["w"], torch.full((4, 4), -3.0))
    assert torch.allclose(merged.vector["b"], torch.full((4,), 1.0))


def test_merge_checkpoints_applies_scaling(base_and_finetuned):
    from trazo.pt3_finetune.merge import merge_checkpoints

    base, ft1, ft2 = base_and_finetuned
    state = merge_checkpoints(base, [ft1, ft2], method="magmax", scaling_coef=0.5)
    assert torch.allclose(state["w"], torch.full((4, 4), -1.5))
    assert torch.equal(state["num_batches_tracked"], torch.tensor(3))


def test_task_vector_addition_and_negation(base_and_finetuned):
    from trazo.pt3_finetune.merge import TaskVector

    base, ft1, ft2 = base_and_finetuned
    a, b = TaskVector(base, ft1), TaskVector(base, ft2)
    assert torch.allclose((a + b).vector["w"], torch.full((4, 4), -1.0))
    assert torch.allclose((-a).vector["w"], torch.full((4, 4), -2.0))


def test_apply_to_rejects_unrelated_checkpoint(tmp_path, base_and_finetuned):
    from trazo.pt3_finetune.merge import TaskVector

    base, ft1, _ = base_and_finetuned
    other = tmp_path / "other.ckpt"
    torch.save({"state_dict": {"totally_different": torch.zeros(2)}}, other)

    with pytest.raises(ValueError, match="shares no keys"):
        TaskVector(base, ft1).apply_to(other)


# ---------------------------------------------------------------- CLI wiring


def test_finetune_cli_forwards_every_flag():
    from trazo.pt3_finetune.cli import build_finetune_args

    args = build_finetune_args(
        "cfg.yaml",
        strategy="lora",
        checkpoint="ftw.ckpt",
        data_dir="/data/region",
        output_dir="/models/ft",
        max_epochs=5,
        lr=1e-4,
        extra=["--data.init_args.batch_size=8"],
    )
    assert args[0] == "fit"
    assert "--model.init_args.strategy=lora" in args
    assert "--model.init_args.pretrained_ckpt=ftw.ckpt" in args
    assert "--data.init_args.root=/data/region" in args
    assert "--trainer.default_root_dir=/models/ft" in args
    assert "--trainer.max_epochs=5" in args
    assert "--model.init_args.lr=0.0001" in args
    assert "--data.init_args.batch_size=8" in args


def test_bundled_configs_exist_for_every_strategy():
    import yaml

    from trazo.pt3_finetune.cli import STRATEGY_CONFIGS, default_config_for

    for strategy in STRATEGY_CONFIGS:
        path = default_config_for(strategy)
        assert path.is_file(), f"missing bundled config for {strategy}: {path}"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["model"]["init_args"]["strategy"] == strategy
        assert cfg["model"]["class_path"].endswith("FineTuneTask")


def test_lora_preserves_output_shape_on_efficientnet():
    """The original wrapper broke stride-2 same-padding convs.

    EfficientNet keeps its padding in a separate module, so re-running the
    convolution with `padding=self.conv.padding` dropped a pixel and the U-Net
    skip connections failed. Guard the shape explicitly.
    """
    smp = pytest.importorskip("segmentation_models_pytorch", reason="needs the pt3 extra")
    pytest.importorskip("efficientnet_pytorch", reason="needs the efficientnet encoder")
    from trazo.pt3_finetune.lora import inject_lora

    model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None,
                     in_channels=8, classes=3).eval()
    x = torch.randn(1, 8, 256, 256)
    with torch.no_grad():
        before = model(x)

    inject_lora(model, r=4, lora_alpha=1.0, lora_dropout=0.1)
    model.eval()
    with torch.no_grad():
        after = model(x)

    assert after.shape == before.shape
    assert torch.allclose(before, after, atol=1e-4)
