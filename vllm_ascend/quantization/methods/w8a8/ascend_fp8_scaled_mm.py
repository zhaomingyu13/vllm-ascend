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
"""FP8 scaled-MM kernel for Ascend, backed by the MXFP8 execution path.

Upstream vLLM's ``Fp8PtpcOnlineLinearMethod`` (per-output-channel weight scale
+ dynamic per-token activation scale) needs a platform FP8 scaled-MM kernel.
Ascend has no native scaled-MM kernel that consumes that layout, so this kernel
maps the PTPC layout onto the MXFP8 path already used by
``AscendW8A8MXFP8DynamicLinearMethod``:

* weights: PTPC produces ``weight [K, N]`` fp8 + ``weight_scale [N, 1]`` fp32.
  The scale is resolved into a dense model-dtype matrix, re-quantized to MXFP8
  (one E8M0 scale per 32 elements along the reduction dim), and handed to the
  MXFP8 weight layout transform.
* activations: quantized dynamically per token with ``npu_dynamic_mx_quant``,
  matching MXFP8's per-token scale semantics.

This kernel only implements the PTPC QuantKey pair, so registering it on the
``OOT`` platform cannot accidentally capture other online FP8 methods (their
per-tensor / per-block keys fail ``can_implement``).
"""

import torch
import torch_npu
from vllm.model_executor.kernels.linear.scaled_mm import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
)

_MXFP8_GROUP_SIZE = 32


class AscendFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """PTPC FP8 scaled-MM kernel executed through the Ascend MXFP8 path.

    The kernel is selected by upstream ``init_fp8_linear_kernel`` when the
    platform FP8 kernel table (``_POSSIBLE_FP8_KERNELS``) lists it on the OOT
    platform. ``process_weights_after_loading`` re-quantizes the PTPC-produced
    weights to MXFP8 and delegates the layout transform to
    ``AscendW8A8MXFP8DynamicLinearMethod``; ``apply_weights`` delegates the
    per-token MXFP8 activation quantization and quantized matmul.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_out_of_tree():
            return False, "requires an out-of-tree (Ascend NPU) platform."
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key != kFp8StaticChannelSym:
            return False, "requires a static per-channel weight scale."
        if c.activation_quant_key != kFp8DynamicTokenSym:
            return False, "requires a dynamic per-token activation scale."
        if c.out_dtype not in (torch.bfloat16, torch.float16):
            return False, "Only bfloat16/float16 output dtype supported."
        n, k = c.weight_shape
        if k % _MXFP8_GROUP_SIZE != 0:
            return False, (
                f"requires the weight reduction dim (K={k}) to be a multiple "
                f"of the MXFP8 group size {_MXFP8_GROUP_SIZE}."
            )
        return True, None

    def __init__(self, c: FP8ScaledMMLinearLayerConfig, layer_param_names):
        # The base class builds a QuantFP8 for the activation key. It is not
        # used here (in-kernel MXFP8 quantization), but is cheap and harmless.
        super().__init__(c, layer_param_names)
        self.mxfp8_method = AscendW8A8MXFP8DynamicLinearMethod()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Re-quantize the PTPC layout to MXFP8, then apply the MXFP8 layout.

        PTPC's ``process_weights_after_loading`` has already replaced the
        layer's weight with ``[K, N]`` fp8 and ``weight_scale`` with ``[N, 1]``
        fp32 (per-output-channel). Resolve those back into a dense
        model-dtype matrix, MX-quantize it, and delegate the transpose/NZ
        weight and scale layout to the MXFP8 method.
        """
        weight = getattr(layer, self.layer_param_names[0])
        weight_scale = getattr(layer, self.layer_param_names[1])
        # weight: [K, N] fp8 * weight_scale: [N, 1] -> dense [N, K] model dtype.
        resolved = (
            weight.t().to(torch.float32) * weight_scale.to(torch.float32)
        ).to(self.config.input_dtype)

        quantized, mx_scale = torch_npu.npu_dynamic_mx_quant(
            resolved,
            dst_type=torch.float8_e4m3fn,
            scale_alg=self.mxfp8_method.dynamic_mx_quant_scale_alg,
        )
        # CANN 9.1 packs two E8M0 scales per entry on a 2D weight, giving
        # [N, K // 64, 2]; the MXFP8 method consumes [N, K // 32]. view() keeps
        # the bytes, reshape() restores the expected layout (see fp8_block).
        mx_scale = mx_scale.view(torch.uint8).reshape(resolved.shape[0], -1)
        replace_parameter(layer, "weight", torch.nn.Parameter(quantized, requires_grad=False))
        replace_parameter(layer, "weight_scale", torch.nn.Parameter(mx_scale, requires_grad=False))
        self.mxfp8_method.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mxfp8_method.apply(layer, x, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "AscendFP8ScaledMMLinearKernel overrides apply_weights directly."
        )
