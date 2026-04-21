# SPDX-License-Identifier: Apache-2.0
"""Dynamic batch size mixin for HPU schedulers.

When enabled via VLLM_DYNAMIC_BATCH_SIZE=1, dynamically adjusts the maximum
number of concurrent requests (max_num_running_reqs) based on KV cache
utilization. This allows running more concurrent requests when sequences are
short (low KV cache usage) and fewer when sequences are long (high usage).

Configuration via environment variables:
    VLLM_DYNAMIC_BATCH_SIZE: Set to "1" to enable (default: "0", disabled)
    VLLM_DYNAMIC_BS_MIN: Minimum batch size under high memory pressure
                         (default: 1)
    VLLM_DYNAMIC_BS_LOW_WATERMARK: KV cache usage below which full batch
                                    size is used (default: 0.3)
    VLLM_DYNAMIC_BS_HIGH_WATERMARK: KV cache usage above which minimum
                                     batch size is used (default: 0.7)
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)


class DynamicBatchSizeMixin:
    """Mixin that overrides max_num_running_reqs as a dynamic property.

    Must be listed BEFORE the Scheduler base class in the MRO so the
    property descriptor takes priority over the instance attribute set
    by Scheduler.__init__.

    When disabled (default), the property simply returns the value that
    Scheduler.__init__ stored, so there is zero overhead.
    """

    _dynamic_bs_enabled: bool = (
        os.environ.get("VLLM_DYNAMIC_BATCH_SIZE", "0") == "1"
    )
    _dynamic_bs_min: int = int(
        os.environ.get("VLLM_DYNAMIC_BS_MIN", "1")
    )
    _dynamic_bs_low_watermark: float = float(
        os.environ.get("VLLM_DYNAMIC_BS_LOW_WATERMARK", "0.3")
    )
    _dynamic_bs_high_watermark: float = float(
        os.environ.get("VLLM_DYNAMIC_BS_HIGH_WATERMARK", "0.7")
    )
    _dynamic_bs_log_interval: int = 50

    @property
    def max_num_running_reqs(self) -> int:
        base = self._base_max_num_running_reqs

        if not self._dynamic_bs_enabled:
            return base

        # Before kv_cache_manager is initialized, return base value.
        kv_mgr = getattr(self, "kv_cache_manager", None)
        if kv_mgr is None:
            return base

        usage = kv_mgr.usage  # 0.0 to 1.0
        low = self._dynamic_bs_low_watermark
        high = self._dynamic_bs_high_watermark
        min_bs = max(self._dynamic_bs_min, 1)

        if usage <= low:
            effective = base
        elif usage >= high:
            effective = min_bs
        else:
            ratio = (usage - low) / (high - low)
            effective = max(min_bs, round(base - ratio * (base - min_bs)))

        # Never return less than current running count so the end-of-schedule
        # assertion  (len(self.running) <= max_num_running_reqs)  always holds.
        # This also means already-running requests are not forcibly removed;
        # instead the standard preemption path handles memory pressure.
        running = getattr(self, "running", None)
        if running is not None:
            effective = max(effective, len(running))

        # Periodic logging so users can observe the adaptation.
        counter = getattr(self, "_dynamic_bs_counter", 0) + 1
        object.__setattr__(self, "_dynamic_bs_counter", counter)
        if counter % self._dynamic_bs_log_interval == 1:
            logger.warning(
                "Dynamic BS: kv_usage=%.2f effective_max=%d base_max=%d "
                "running=%d min_bs=%d watermarks=[%.2f, %.2f]",
                usage,
                effective,
                base,
                len(running) if running else 0,
                min_bs,
                low,
                high,
            )

        return effective

    @max_num_running_reqs.setter
    def max_num_running_reqs(self, value: int) -> None:
        # Store via object.__setattr__ to avoid recursion with the property.
        object.__setattr__(self, "_base_max_num_running_reqs", value)

    if _dynamic_bs_enabled:
        logger.info(
            "Dynamic batch size ENABLED: min=%s low_wm=%s high_wm=%s",
            os.environ.get("VLLM_DYNAMIC_BS_MIN", "1"),
            os.environ.get("VLLM_DYNAMIC_BS_LOW_WATERMARK", "0.3"),
            os.environ.get("VLLM_DYNAMIC_BS_HIGH_WATERMARK", "0.7"),
        )
