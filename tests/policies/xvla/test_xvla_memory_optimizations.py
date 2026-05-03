#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAModel
from lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _xvla_config(**kwargs) -> XVLAConfig:
    return XVLAConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(20,)),
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        florence_config={
            "vision_config": {
                "dim_embed": [8, 8, 8, 8],
                "num_heads": [1, 1, 1, 1],
                "num_groups": [1, 1, 1, 1],
                "depths": [1, 1, 1, 1],
                "projection_dim": 8,
            },
            "text_config": {
                "vocab_size": 32,
                "d_model": 8,
                "encoder_layers": 1,
                "encoder_attention_heads": 1,
                "encoder_ffn_dim": 16,
                "decoder_layers": 1,
                "decoder_attention_heads": 1,
                "decoder_ffn_dim": 16,
            },
            "projection_dim": 8,
        },
        hidden_size=8,
        depth=1,
        num_heads=1,
        max_len_seq=32,
        **kwargs,
    )


def _uninitialized_xvla_model(config: XVLAConfig) -> XVLAModel:
    model = XVLAModel.__new__(XVLAModel)
    nn.Module.__init__(model)
    model.config = config
    return model


def test_xvla_auto_dtype_uses_float32_on_cpu():
    model = _uninitialized_xvla_model(_xvla_config(dtype="auto", device="cpu"))

    assert model._get_target_dtype() is torch.float32


def test_xvla_accepts_float16_dtype():
    model = _uninitialized_xvla_model(_xvla_config(dtype="float16"))

    assert model._get_target_dtype() is torch.float16


def test_xvla_rejects_invalid_dtype():
    with pytest.raises(ValueError, match="Invalid dtype"):
        _xvla_config(dtype="int8")


def test_xvla_vlm_gradient_checkpointing_updates_florence_config():
    florence_config = _xvla_config(vlm_gradient_checkpointing=True).get_florence_config()

    assert florence_config.vision_config.enable_checkpoint is True
    assert florence_config.text_config.use_cache is False


def test_xvla_freeze_vlm_disables_vlm_gradients_only():
    model = _uninitialized_xvla_model(_xvla_config(freeze_vlm=True))
    model.vlm = nn.Linear(4, 4)
    model.transformer = nn.Linear(4, 4)

    model._apply_freezing()

    assert all(not param.requires_grad for param in model.vlm.parameters())
    assert all(param.requires_grad for param in model.transformer.parameters())


class _DummyEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, attention_mask, inputs_embeds):  # noqa: ARG002
        return (self.proj(inputs_embeds),)


class _DummyLanguageModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.encoder = _DummyEncoder(dim)


class _DummyVLM(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.image_encoder = nn.Linear(3, dim)
        self.embedding = nn.Embedding(16, dim)
        self.language_model = _DummyLanguageModel(dim)
        self.num_encoded_images = 0

    def _encode_image(self, pixel_values):
        self.num_encoded_images = pixel_values.shape[0]
        return self.image_encoder(pixel_values.mean(dim=(-1, -2))).unsqueeze(1)

    def get_input_embeddings(self):
        return self.embedding

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds):
        merged = torch.cat([image_features, inputs_embeds], dim=1)
        attention_mask = torch.ones(merged.shape[:2], device=merged.device, dtype=merged.dtype)
        return merged, attention_mask


def test_xvla_frozen_vlm_forward_skips_autograd_and_padded_images():
    model = _uninitialized_xvla_model(_xvla_config(freeze_vlm=True))
    model.vlm = _DummyVLM()
    model.transformer = nn.Linear(4, 4)
    model._apply_freezing()
    model.train()

    enc = model.forward_vlm(
        input_ids=torch.ones(2, 3, dtype=torch.long),
        pixel_values=torch.randn(2, 3, 3, 8, 8),
        image_mask=torch.tensor([[True, False, True], [True, False, False]]),
    )

    assert model.vlm.num_encoded_images == 3
    assert not enc["vlm_features"].requires_grad
    assert not enc["aux_visual_inputs"].requires_grad


def test_soft_prompted_transformer_uses_checkpointing_in_training(monkeypatch):
    calls = 0

    def fake_checkpoint(function, *args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["use_reentrant"] is False
        return function(*args)

    monkeypatch.setattr(
        "lerobot.policies.xvla.soft_transformer.checkpoint.checkpoint",
        fake_checkpoint,
    )
    transformer = SoftPromptedTransformer(
        hidden_size=8,
        multi_modal_input_size=8,
        depth=2,
        num_heads=1,
        mlp_ratio=1,
        dim_action=4,
        dim_propio=4,
        dim_time=4,
        len_soft_prompts=0,
        max_len_seq=16,
        gradient_checkpointing=True,
    )
    transformer.train()

    out = transformer(
        domain_id=torch.zeros(2, dtype=torch.long),
        vlm_features=torch.randn(2, 2, 8),
        aux_visual_inputs=torch.randn(2, 1, 8),
        action_with_noise=torch.randn(2, 3, 4),
        proprio=torch.randn(2, 4),
        t=torch.rand(2),
    )
    out.sum().backward()

    assert calls == 2
