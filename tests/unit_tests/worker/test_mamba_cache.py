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
    if isinstance(kv_caches, (tuple, list)):
        # If nested list (e.g., [[tensor1, tensor2]])
        if len(kv_caches) == 1 and isinstance(kv_caches[0], (tuple, list)):
            tensors = kv_caches[0]
        else:
            tensors = kv_caches
        assert tensors[0].shape == (num_blocks, 2)
        assert tensors[1].shape == (num_blocks, 3)
    else:
        shape = kv_caches.shape
        assert (shape == (num_blocks, 2)) or (shape == (num_blocks, 3)) or (shape[:3] == (2, 3, num_blocks))

def test_kv_cache_tensor_shape_with_two_mamba_layers():
    class FakeMamba(MambaBase):
        def __init__(self, shape):
            self._shape = shape
        def get_state_shape(self):
            return self._shape
        @property
        def mamba_type(self):
            return "mamba1"
        def get_state_dtype(self):
            return (torch.bfloat16, torch.bfloat16)
    vllm_config = get_vllm_config()
    vllm_config.cache_config.mamba_block_size = 5
    vllm_config.cache_config.mamba_page_size_padded = 128
    # Two mamba layers with different shapes
    mamba_layer_names = ["layer.0", "layer.1"]
    shapes = [((2,), (3,)), ((4,), (5,))]
    vllm_config.compilation_config.static_forward_context = {
        mamba_layer_names[0]: FakeMamba(shapes[0]),
        mamba_layer_names[1]: FakeMamba(shapes[1]),
    }
    runner = HPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    num_blocks = 7
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=kv_cache_spec[mamba_layer_names[0]].page_size_bytes * num_blocks, shared_by=[mamba_layer_names[0]]),
            KVCacheTensor(size=kv_cache_spec[mamba_layer_names[1]].page_size_bytes * num_blocks, shared_by=[mamba_layer_names[1]]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[mamba_layer_names[0]], kv_cache_spec=kv_cache_spec[mamba_layer_names[0]]),
            KVCacheGroupSpec(layer_names=[mamba_layer_names[1]], kv_cache_spec=kv_cache_spec[mamba_layer_names[1]]),
        ],
    )
    runner.initialize_kv_cache(kv_cache_config)
    kv_caches = runner.kv_caches
    for i, lname in enumerate(mamba_layer_names):
        print(f"kv_cache_spec for {lname}: {kv_cache_spec[lname].shapes}")
        tensors = kv_caches[i]
        print(f"FakeMamba.get_state_shape() for {lname}: {tensors[0].shape}, {tensors[1].shape}")
        assert tensors[0].shape == (num_blocks, shapes[i][0][0])
        assert tensors[1].shape == (num_blocks, shapes[i][1][0])
