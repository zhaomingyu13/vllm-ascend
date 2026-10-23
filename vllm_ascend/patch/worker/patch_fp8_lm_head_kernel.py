#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Register the Ascend FP8 scaled-MM kernel for PTPC online FP8 quantization.

Upstream ``init_fp8_linear_kernel`` selects a platform FP8 kernel from
``_POSSIBLE_FP8_KERNELS``, which has no entry for the OOT (Ascend) platform.
Without one, ``Fp8PtpcOnlineLinearMethod`` (used by the quantized draft
lm_head copy, `draft_lm_head_quantization="fp8"`) raises ``ValueError``.

This patch uses upstream's own OOT registration API (``register_linear_kernel``
with ``kernel_type="fp8"`` on ``PlatformEnum.OOT``) to register
``AscendFP8ScaledMMLinearKernel``. Its ``can_implement`` only matches the
PTPC QuantKey pair (static per-channel weight + dynamic per-token activation),
so other online FP8 methods are never captured.

It also makes ``fp8_scaled_mm_is_batch_invariant`` return True for this
kernel. Under ``VLLM_BATCH_INVARIANT``, ``Fp8PtpcOnlineLinearMethod.apply``
otherwise falls back to a BF16 dequant path that assumes the PTPC ``[K, N]``
weight + ``[N, 1]`` scale layout, which no longer holds after the kernel
rewrites the layer to the MXFP8 layout. MXFP8 per-token activation scales are
batch-invariant, so the kernel may safely run directly.

Both the ``vllm.model_executor.kernels.linear`` module attribute and the
``online.fp8`` module binding must be replaced: ``online.fp8`` imports the
function at module import time, which happens before worker patches run.
"""

import logging

from vllm.platforms.interface import PlatformEnum

from vllm_ascend.quantization.methods.w8a8.ascend_fp8_scaled_mm import (
    AscendFP8ScaledMMLinearKernel,
)

logger = logging.getLogger(__name__)


# register_linear_kernel exists since vLLM 0.26.0 ([OOT] Add OOT support for
# linear kernel, #37989); it is present in both v0.31.0 and main.
from vllm.model_executor.kernels.linear import (  # noqa: E402
    register_linear_kernel,
)

register_linear_kernel(
    AscendFP8ScaledMMLinearKernel,
    PlatformEnum.OOT,
    kernel_type="fp8",
)

# ``fp8_scaled_mm_is_batch_invariant`` only exists on vLLM builds that ship the
# batch-invariance branch in ``online/fp8.py`` (v0.32.0+/main). On v0.31.0
# PTPC's apply() never consults it, so nothing needs patching there.
import vllm.model_executor.kernels.linear as _kernels_linear  # noqa: E402

_fp8_batch_invariant = getattr(_kernels_linear, "fp8_scaled_mm_is_batch_invariant", None)
if _fp8_batch_invariant is not None:

    def _ascend_fp8_batch_invariant(kernel):
        if isinstance(kernel, AscendFP8ScaledMMLinearKernel):
            return True
        return _fp8_batch_invariant(kernel)

    _kernels_linear.fp8_scaled_mm_is_batch_invariant = _ascend_fp8_batch_invariant
    try:
        import vllm.model_executor.layers.quantization.online.fp8 as _online_fp8  # noqa: E402

        if hasattr(_online_fp8, "fp8_scaled_mm_is_batch_invariant"):
            _online_fp8.fp8_scaled_mm_is_batch_invariant = _ascend_fp8_batch_invariant
    except ImportError:
        # online/fp8.py may not be imported yet (or at all) on this build;
        # when it is imported later it resolves the function from the
        # kernels.linear module attribute patched above.
        pass

logger.info(
    "Registered %s on the OOT platform for FP8 PTPC online quantization "
    "(draft_lm_head_quantization='fp8').",
    AscendFP8ScaledMMLinearKernel.__name__,
)
