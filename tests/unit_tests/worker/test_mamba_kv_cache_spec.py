import sys
from unittest.mock import MagicMock
sys.modules["vllm.attention.layer"] = MagicMock()

import torch
import pytest
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
from vllm.config import VllmConfig
from vllm.platforms import current_platform


DEVICE = current_platform.device_type

def get_vllm_config():
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=False,  # Added required field
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        task="generate",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=128,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config

def test_get_kv_cache_spec_with_mamba():
    class FakeMamba(MambaBase):
        def get_state_shape(self):
            return [(2, 3), (4, 5)]
        @property
        def mamba_type(self):
            return "fake_mamba"
        def get_state_dtype(self):
            return (torch.bfloat16, torch.bfloat16)
    vllm_config = get_vllm_config()
    vllm_config.cache_config.mamba_block_size = 42
    vllm_config.cache_config.mamba_page_size_padded = 128
    vllm_config.compilation_config.static_forward_context = {"layer.mamba": FakeMamba()}
    runner = HPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    assert "layer.mamba" in kv_cache_spec
    spec = kv_cache_spec["layer.mamba"]
    assert isinstance(spec, MambaSpec)
    assert spec.shapes == [(2, 3), (4, 5)]
    assert spec.dtypes == (torch.bfloat16, torch.bfloat16)
    assert spec.block_size == 42
    assert spec.page_size_padded == 128
    assert spec.mamba_type == "fake_mamba"
    assert spec.num_speculative_blocks == 0
