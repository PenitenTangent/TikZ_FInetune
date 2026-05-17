import json
from pathlib import Path

import pytest

from tikz_mlx.adapter_config_io import (
    adapter_lora_input_dim_rewrites,
    infer_adapter_lora_rank,
    materialize_lora_handoff_adapter,
    validate_resumed_adapter_model_and_shape,
)

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")
from safetensors.torch import load_file, save_file  # noqa: E402


def _write_adapter(path: Path, *, rank: int, alpha: int, dropout: float, input_dim: int = 4) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    tensors = {
        "language_model.model.layers.0.self_attn.q_proj.A": torch.arange(
            input_dim * rank, dtype=torch.float32
        ).reshape(input_dim, rank),
        "language_model.model.layers.0.self_attn.q_proj.B": torch.arange(
            rank * 3, dtype=torch.float32
        ).reshape(rank, 3),
        "language_model.model.layers.0.norm.weight": torch.ones(input_dim, dtype=torch.float32),
    }
    save_file(tensors, str(path / "adapters.safetensors"))
    (path / "adapter_config.json").write_text(
        json.dumps({"rank": rank, "alpha": alpha, "dropout": dropout}),
        encoding="utf-8",
    )
    return path


def test_infer_adapter_lora_rank_from_tensor_shapes(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "adapter", rank=16, alpha=32, dropout=0.03)

    assert infer_adapter_lora_rank(adapter) == 16


def test_materialize_lora_handoff_expands_rank_without_changing_delta(tmp_path: Path) -> None:
    source = _write_adapter(tmp_path / "source", rank=16, alpha=32, dropout=0.03)
    target = tmp_path / "target"

    result = materialize_lora_handoff_adapter(
        source_adapter_path=source,
        target_dir=target,
        target_rank=24,
        target_alpha=48,
        target_dropout=0.05,
        seed=123,
    )

    assert result["expanded"] is True
    assert json.loads((target / "adapter_config.json").read_text(encoding="utf-8")) == {
        "rank": 24,
        "alpha": 48,
        "dropout": 0.05,
    }
    old = load_file(str(source / "adapters.safetensors"))
    new = load_file(str(target / "adapters.safetensors"))
    old_a = old["language_model.model.layers.0.self_attn.q_proj.A"]
    old_b = old["language_model.model.layers.0.self_attn.q_proj.B"]
    new_a = new["language_model.model.layers.0.self_attn.q_proj.A"]
    new_b = new["language_model.model.layers.0.self_attn.q_proj.B"]

    assert tuple(new_a.shape) == (4, 24)
    assert tuple(new_b.shape) == (24, 3)
    torch.testing.assert_close(new_a[:, :16], old_a)
    torch.testing.assert_close(new_b[:16, :], old_b)
    torch.testing.assert_close(new_b[16:, :], torch.zeros_like(new_b[16:, :]))
    torch.testing.assert_close((old_a @ old_b) * (32 / 16), (new_a @ new_b) * (48 / 24))
    torch.testing.assert_close(
        new["language_model.model.layers.0.norm.weight"],
        old["language_model.model.layers.0.norm.weight"],
    )


def test_materialize_lora_handoff_rescales_alpha_to_preserve_delta(tmp_path: Path) -> None:
    source = _write_adapter(tmp_path / "source", rank=16, alpha=16, dropout=0.03)
    target = tmp_path / "target"

    result = materialize_lora_handoff_adapter(
        source_adapter_path=source,
        target_dir=target,
        target_rank=16,
        target_alpha=32,
        target_dropout=0.03,
    )

    assert result["expanded"] is False
    assert result["alpha_rescaled"] is True
    old = load_file(str(source / "adapters.safetensors"))
    new = load_file(str(target / "adapters.safetensors"))
    old_a = old["language_model.model.layers.0.self_attn.q_proj.A"]
    old_b = old["language_model.model.layers.0.self_attn.q_proj.B"]
    new_a = new["language_model.model.layers.0.self_attn.q_proj.A"]
    new_b = new["language_model.model.layers.0.self_attn.q_proj.B"]
    torch.testing.assert_close((old_a @ old_b) * (16 / 16), (new_a @ new_b) * (32 / 16))


def test_materialize_lora_handoff_rejects_rank_shrink(tmp_path: Path) -> None:
    source = _write_adapter(tmp_path / "source", rank=24, alpha=48, dropout=0.05)

    with pytest.raises(RuntimeError, match="higher-rank LoRA adapter"):
        materialize_lora_handoff_adapter(
            source_adapter_path=source,
            target_dir=tmp_path / "target",
            target_rank=16,
            target_alpha=32,
            target_dropout=0.03,
        )


def test_materialize_lora_handoff_rewrites_known_gemma_input_dim(tmp_path: Path) -> None:
    source = _write_adapter(tmp_path / "source", rank=16, alpha=32, dropout=0.03, input_dim=2400)
    target = tmp_path / "target"

    rewrites = adapter_lora_input_dim_rewrites(
        source,
        expected_model_id="mlx-community/gemma-4-e4b-it-6bit",
    )
    result = materialize_lora_handoff_adapter(
        source_adapter_path=source,
        target_dir=target,
        target_rank=16,
        target_alpha=32,
        target_dropout=0.03,
        input_dim_rewrites=rewrites,
    )

    old = load_file(str(source / "adapters.safetensors"))
    new = load_file(str(target / "adapters.safetensors"))
    old_a = old["language_model.model.layers.0.self_attn.q_proj.A"]
    new_a = new["language_model.model.layers.0.self_attn.q_proj.A"]

    assert rewrites == {2400: 2560}
    assert result["input_dim_rewritten"] is True
    assert tuple(new_a.shape) == (2560, 16)
    torch.testing.assert_close(new_a[:2400, :], old_a)
    torch.testing.assert_close(new_a[2400:, :], torch.zeros_like(new_a[2400:, :]))


def test_validate_resumed_adapter_allows_known_gemma_input_dim_rewrite(tmp_path: Path) -> None:
    source = _write_adapter(tmp_path / "source", rank=16, alpha=32, dropout=0.03, input_dim=2400)

    validate_resumed_adapter_model_and_shape(
        adapter_path=source,
        expected_model_id="mlx-community/gemma-4-e4b-it-6bit",
    )


def test_mlx_vlm_lora_patch_uses_quantized_scale_shape_for_input_dim() -> None:
    nn = pytest.importorskip("mlx.nn")
    pytest.importorskip("mlx_vlm.trainer.lora")
    from mlx_vlm.trainer.lora import LoRaLayer

    linear = nn.QuantizedLinear(2560, 2048, bias=False, group_size=64, bits=6)
    layer = LoRaLayer(linear, rank=16, alpha=32, dropout=0.0)

    assert tuple(layer.A.shape) == (2560, 16)
    assert tuple(layer.B.shape) == (16, 2048)
