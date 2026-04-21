# SPDX-License-Identifier: Apache-2.0
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_gaudi.v1.core.sched.dynamic_batch_size import DynamicBatchSizeMixin


class HPUScheduler(DynamicBatchSizeMixin, Scheduler):
    """HPU scheduler with dynamic batch size support."""
    pass
