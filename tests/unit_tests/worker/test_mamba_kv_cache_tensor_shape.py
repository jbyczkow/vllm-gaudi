import torch
import pytest
from vllm.v1.kv_cache_interface import MambaSpec, KVCacheConfig, KVCacheTensor, KVCacheGroupSpec
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

def test_kv_cache_tensor_shape_with_mamba():
    class FakeMamba(MambaBase):
        def get_state_shape(self):
            return ((2,), (3,))
        @property
        def mamba_type(self):
            return "mamba1"
        def get_state_dtype(self):
            return (torch.bfloat16, torch.bfloat16)
    vllm_config = get_vllm_config()
    vllm_config.cache_config.mamba_block_size = 5
    vllm_config.cache_config.mamba_page_size_padded = 128
    # Use 'layer.0' as the mamba layer name to match main code expectations
    mamba_layer_name = "layer.0"
    vllm_config.compilation_config.static_forward_context = {mamba_layer_name: FakeMamba()}
    runner = HPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    spec = kv_cache_spec[mamba_layer_name]
    num_blocks = 7
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=[mamba_layer_name])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[mamba_layer_name], kv_cache_spec=spec)],
    )
    runner.initialize_kv_cache(kv_cache_config)
    kv_caches = runner.kv_caches
    assert mamba_layer_name in kv_caches
    tensor = kv_caches[mamba_layer_name]
    # Robust: handle both list-of-tensors and single tensor
    if isinstance(tensor, (tuple, list)):
        assert tensor[0].shape == (num_blocks, 2)
        assert tensor[1].shape == (num_blocks, 3)
    else:
        shape = tensor.shape
        assert (shape == (num_blocks, 2)) or (shape == (num_blocks, 3)) or (shape[:3] == (2, 3, num_blocks))
