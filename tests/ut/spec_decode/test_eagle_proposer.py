import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CacheConfig, CompilationMode, CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.spec_decode.draft_model import DraftModelProposer

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

BLOCK_SIZE = 16


@dataclass
class BatchSpec:
    """Specification for a batch configuration (workload shape only)."""

    seq_lens: list[int]
    query_lens: list[int]

    name: str = "unnamed"

    @property
    def batch_size(self):
        return len(self.seq_lens)

    def __post_init__(self):
        assert len(self.seq_lens) == len(self.query_lens)

    def compute_num_tokens(self):
        return sum(self.query_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
    max_block_idx: int = 1000,
    arange_block_indices: bool = False,
) -> AscendCommonAttentionMetadata:
    """Create AscendCommonAttentionMetadata from a BatchSpec and ModelParams."""
    # Create query start locations
    query_start_loc = torch.zeros(batch_spec.batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(batch_spec.query_lens, dtype=torch.int32, device=device).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = batch_spec.compute_num_tokens()

    # Create sequence lengths
    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())

    # Create computed tokens (context length for each sequence)
    context_lens = [batch_spec.seq_lens[i] - batch_spec.query_lens[i] for i in range(batch_spec.batch_size)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)

    # Create block table and slot mapping
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    if arange_block_indices:
        num_blocks = batch_spec.batch_size * max_blocks
        block_table_tensor = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
            batch_spec.batch_size, max_blocks
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device).view(num_tokens)
    else:
        block_table_tensor = torch.randint(
            0,
            max_block_idx,
            (batch_spec.batch_size, max_blocks),
            dtype=torch.int32,
            device=device,
        )
        slot_mapping = torch.randint(0, max_block_idx, (num_tokens,), dtype=torch.int64, device=device)

    # Calculate max query length
    max_query_len = max(batch_spec.query_lens)

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


class TestEagleProposerInitialization(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.cache_config = MagicMock(spec=CacheConfig)
        self.vllm_config.scheduler_config = MagicMock()
        self.vllm_config.model_config = MagicMock()
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.compilation_config = MagicMock()
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    def test_initialization_eagle_graph(self):
        self.vllm_config.speculative_config.method = "eagle"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 4096
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = False
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 4096)
            self.assertTrue(proposer.use_cuda_graph)

            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.input_ids.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.positions.shape, (expected_max_num_tokens,))
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 4096))
            self.assertEqual(proposer.arange.shape, (expected_max_num_tokens,))

    def test_initialization_eagle3_enforce_eager(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.NONE
        self.vllm_config.compilation_config.pass_config = MagicMock()
        self.vllm_config.compilation_config.pass_config.enable_sp = False
        self.vllm_config.model_config.enforce_eager = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertFalse(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_eagle3_full_graph_async(self):
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertTrue(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_mtp_full_graph_async(self):
        self.vllm_config.speculative_config.method = "mtp"
        self.vllm_config.speculative_config.draft_model_config.get_hidden_size.return_value = 2048
        self.vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        self.vllm_config.model_config.enforce_eager = False
        self.vllm_config.speculative_config.enforce_eager = False
        self.vllm_config.scheduler_config.async_scheduling = True
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertEqual(proposer.hidden_size, 2048)
            self.assertTrue(proposer.use_cuda_graph)
            expected_max_num_tokens = proposer.max_num_tokens
            self.assertEqual(proposer.hidden_states.shape, (expected_max_num_tokens, 2048))

    def test_initialization_draft_model(self):
        self.vllm_config.speculative_config.method = "draft_model"
        self.vllm_config.speculative_config.parallel_drafting = False
        # TODO(klyzhenko-vadim): remove when target_tp != draft_tp will be supported.
        self.vllm_config.speculative_config.draft_parallel_config.tensor_parallel_size = 1
        self.vllm_config.speculative_config.target_parallel_config.tensor_parallel_size = 1
        init_ascend_config(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            proposer = AscendDraftModelProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

            self.assertTrue(isinstance(proposer, DraftModelProposer))
            self.assertFalse(proposer.pass_hidden_states_to_model)
            self.assertTrue(proposer.needs_extra_input_slots)


@unittest.skip("Skip due to the changes in #7153, fix me later")
class TestEagleProposerLoadModel(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.method = "eagle"
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.parallel_drafting = False

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 1
        mock_target_layer1 = MagicMock()
        mock_target_layer2 = MagicMock()
        mock_draft_layer1 = MagicMock()
        mock_draft_layer3 = MagicMock()
        mock_get_layers.side_effect = [
            {"layer1": mock_target_layer1, "layer2": mock_target_layer2},
            {},
            {},
            {"layer1": mock_draft_layer1, "layer3": mock_draft_layer3},
        ]

        weight = torch.zeros(0)

        mock_model = MagicMock()
        mock_model.supports_multimodal = False
        mock_model.lm_head = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_model.model.embed_tokens = MagicMock()
        mock_model.model.embed_tokens.weight = weight

        mock_get_model.return_value = MagicMock()
        mock_get_model.return_value.model.embed_tokens.weight = weight

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            mock_get_model.assert_called_once()
            self.assertEqual(self.proposer.attn_layer_names, ["layer3"])
            self.assertIs(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    def test_load_model_pp_gt1(self, mock_pp_group, mock_get_model, mock_get_layers):
        mock_pp_group.return_value.world_size = 2
        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]

        mock_model = MagicMock()
        original_embed = MagicMock()
        mock_model.multimodal_cpu_fields = None
        mock_model.merge_by_field_config = None
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)

            self.assertIsNot(self.proposer.model.model.embed_tokens, mock_model.model.embed_tokens)
            self.assertEqual(self.proposer.attn_layer_names, ["layer2"])

    @patch("vllm_ascend.spec_decode.eagle_proposer.get_layers_from_vllm_config")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_model")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_pp_group")
    @patch("vllm_ascend.spec_decode.eagle_proposer.supports_multimodal")
    def test_load_model_multimodal(self, mock_supports_multi, mock_pp_group, mock_get_model, mock_get_layers):
        mock_model = MagicMock()
        mock_model.get_language_model.return_value.lm_head = MagicMock()
        mock_supports_multi.return_value = True
        original_embed = MagicMock()
        mock_get_model.return_value = MagicMock(model=MagicMock(embed_tokens=original_embed))

        mock_target_layer1 = MagicMock()
        mock_draft_layer2 = MagicMock()

        mock_get_layers.side_effect = [{"layer1": mock_target_layer1}, {}, {}, {"layer2": mock_draft_layer2}]
        mock_pp_group.return_value.world_size = 2

        self.proposer.model = MagicMock()

        with set_current_vllm_config(self.vllm_config):
            self.proposer.load_model(mock_model)
            self.assertEqual(mock_model.get_language_model.call_count, 2)
            self.assertIs(self.proposer.model.lm_head, mock_model.get_language_model.return_value.lm_head)


class TestEagleProposerDummyRun(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.num_speculative_tokens = 4
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1
        self.runner.pin_memory = False
        self.runner._sync_metadata_across_dp.return_value = (8, torch.tensor([8]), CUDAGraphMode.NONE)

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.model_config.use_mla = False
        self.vllm_config.model_config.hf_text_config = MagicMock(
            spec=[]
        )  # Empty spec to prevent hasattr from returning True
        self.vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(4)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Mock parallel state functions
        self.mock_tp_world_size = patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1
        )
        self.mock_tp_world_size.start()

        mock_dp_group = MagicMock()
        mock_dp_group.world_size = 1
        self.mock_dp_group = patch("vllm_ascend.ascend_forward_context.get_dp_group", return_value=mock_dp_group)
        self.mock_dp_group.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)
        self.proposer.model = MagicMock()
        self.proposer._runnable = MagicMock()
        self.proposer.update_stream = MagicMock()

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        self.mock_tp_world_size.stop()
        self.mock_dp_group.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # cpu does not support parallel-group, let alone `sp`
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_forward_context", **{"return_value.flash_comm_v1_enabled": False}
    )
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_basic(self, mock_context, mock_get_context, mock_get_context_2):
        num_tokens = 32
        with_prefill = False

        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=num_tokens, with_prefill=with_prefill)

            self.assertTrue(self.proposer._runnable.call_count == 1)

    # cpu does not support parallel-group, let alone `sp`
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch(
        "vllm_ascend.spec_decode.eagle_proposer.get_forward_context", **{"return_value.flash_comm_v1_enabled": False}
    )
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_with_prefill(self, mock_context, mock_get_context, mock_get_context_2):
        mock_context.return_value.__enter__.return_value = None
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, with_prefill=True, num_reqs=4)
            self.assertTrue(self.proposer._runnable.call_count == 1)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_capture(
        self, mock_context, mock_get_context, mock_update_full_graph_params, mock_get_context_2
    ):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = True
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=True, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            mock_update_full_graph_params.assert_not_called()
            self.proposer.use_cuda_graph = last_use_cuda_graph

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.update_full_graph_params")
    @patch("vllm_ascend.spec_decode.eagle_proposer.get_forward_context")
    @patch("vllm_ascend.spec_decode.eagle_proposer.set_ascend_forward_context")
    def test_dummy_run_in_graph_run(
        self, mock_context, mock_get_context, mock_update_full_graph_params, mock_get_context_2
    ):
        last_use_cuda_graph = self.proposer.use_cuda_graph
        mock_return_context = MagicMock()
        mock_return_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        mock_return_context.capturing = False
        # cpu does not support parallel-group, let alone `sp`
        mock_return_context.flash_comm_v1_enabled = False
        mock_get_context.return_value = mock_return_context
        mock_get_context_2.return_value = mock_return_context
        self.proposer.use_cuda_graph = True
        self.proposer.draft_attn_groups = [MagicMock()]
        # cpu does not support `torch.ops.vllm.maybe_pad_and_reduce`
        with set_current_vllm_config(self.vllm_config):
            self.proposer.enable_shared_expert_dp = False
            self.proposer.dummy_run(num_tokens=64, in_graph_capturing=False, aclgraph_runtime_mode=CUDAGraphMode.FULL)
            self.assertTrue(self.proposer._runnable.call_count == 1)
            self.assertTrue(mock_update_full_graph_params.call_count == 1)
            self.proposer.use_cuda_graph = last_use_cuda_graph


class TestEagleProposerHelperMethods(TestBase):
    # TODO: Can add some tests about prepare_next_token_ids in future.

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.scheduler_config = MagicMock(max_num_seqs=3)
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = [0, 1, 2]
        self.runner.arange_np = np.arange(10)
        self.runner.input_batch.num_reqs = 3
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1

        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 2048
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.enable_expert_parallel = False
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.num_speculative_tokens = 2
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(2)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # TODO: This is equivalent to disable_padded_drafter_batch=True.
    # We need to add a test_prepare_inputs_padded in future.
    def test_prepare_inputs(self):
        self.proposer.token_arange_np = np.arange(10)
        mock_attn = MagicMock()
        mock_attn.slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
        num_rejected = torch.tensor([1, 0, 1], device=self.device)
        mock_return_attn = MagicMock()

        with (
            set_current_vllm_config(self.vllm_config),
            patch.object(self.proposer, "prepare_inputs", return_value=(mock_return_attn, torch.tensor([1, 2, 4]))),
        ):
            return_attn, indices = self.proposer.prepare_inputs(mock_attn, num_rejected)
            self.assertEqual(indices.tolist(), [1, 2, 4])


class TestEagleProposerPropose:
    @pytest.fixture(autouse=True)
    def setUp_and_tearDown(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.speculative_config = MagicMock()
        self.vllm_config.speculative_config.num_speculative_tokens = 3
        self.vllm_config.speculative_config.method = "eagle3"
        self.vllm_config.speculative_config.parallel_drafting = False
        self.device = torch.device("cpu")
        self.runner = MagicMock()
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1
        self.runner.max_num_tokens = 8192
        self.runner.max_num_reqs = 256
        self.runner.pin_memory = False

        self.vllm_config.scheduler_config.max_num_batched_tokens = 1024
        self.vllm_config.scheduler_config.max_num_seqs = 32
        self.vllm_config.model_config.dtype = torch.float16
        self.vllm_config.model_config.max_model_len = 32768
        self.vllm_config.model_config.uses_mrope = False
        self.vllm_config.model_config.uses_xdrope_dim = 0
        self.vllm_config.model_config.use_mla = False
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.parallel_config.data_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.speculative_config.draft_tensor_parallel_size = 1
        self.vllm_config.speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(4)])
        self.vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
        self.vllm_config.speculative_config.draft_model_config.uses_mrope = False
        self.vllm_config.speculative_config.disable_padded_drafter_batch = False
        self.vllm_config.additional_config = None
        init_ascend_config(self.vllm_config)

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

        # Mock parallel state functions
        self.mock_tp_world_size = patch(
            "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1
        )
        self.mock_tp_world_size.start()

        mock_dp_group = MagicMock()
        mock_dp_group.world_size = 1
        self.mock_dp_group = patch("vllm_ascend.ascend_forward_context.get_dp_group", return_value=mock_dp_group)
        self.mock_dp_group.start()

        # Mock sp
        self.mock_enable_sp = patch("vllm_ascend.utils.enable_sp", return_value=False)
        self.mock_enable_sp.start()

        # Set the current vllm config
        set_current_vllm_config(self.vllm_config)
        self.proposer = AscendEagleProposer(vllm_config=self.vllm_config, device=self.device, runner=self.runner)

        yield

        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()
        self.mock_tp_world_size.stop()
        self.mock_dp_group.stop()
        # Clear the current vllm config
        set_current_vllm_config(None)

    # config: prefill and decode, Qwen3-8B, tp1, enforce_eager,
    # no_async_scheduling, eagle3, k=3, "disable_padded_drafter_batch": False
    @pytest.mark.parametrize(
        "flag_prefill_decode, query_start_loc, query_start_loc_cpu, seq_lens, num_reqs,"
        "num_actual_tokens, max_query_len, max_seq_len, block_table_tensor,"
        "slot_mapping, causal, logits_indices_padded, num_logits_indices,"
        "encoder_seq_lens, encoder_seq_lens_cpu, dcp_local_seq_lens,"
        "dcp_local_seq_lens_cpu, _seq_lens_cpu, _num_computed_tokens_cpu,"
        "_num_computed_tokens_cache, seq_lens_cpu, num_computed_tokens_cpu,"
        "decode_token_per_req, actual_seq_lengths_q, positions, attn_state,"
        "graph_pad_size, num_input_tokens, prefill_context_parallel_metadata",
        [
            (
                "prefill",
                torch.tensor([0, 13], device=torch.device("cpu"), dtype=torch.int32),
                torch.tensor([0, 13], dtype=torch.int32),
                torch.tensor([13], device=torch.device("cpu"), dtype=torch.int32),
                1,
                13,
                13,
                13,
                torch.eye(256, device=torch.device("cpu"), dtype=torch.int32)[0].unsqueeze(0),
                torch.tensor(
                    [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140],
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                ),
                True,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                torch.tensor([13], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                4,
                [],
                torch.cat([torch.arange(13), torch.zeros(8704 - 13)]),
                AscendAttentionState.PrefillNoCache,
                -1,
                13,
                None,
            ),
            (
                "decode",
                torch.tensor([0, 4, 8, 12], device=torch.device("cpu"), dtype=torch.int32),
                torch.tensor([0, 4, 8, 12], dtype=torch.int32),
                torch.tensor([21, 17, 17], device=torch.device("cpu"), dtype=torch.int32),
                3,
                12,
                4,
                0,
                torch.cat(
                    [torch.eye(256, device="cpu", dtype=torch.int32)[0].unsqueeze(0) * i for i in [1, 2, 3]], dim=0
                ),
                torch.tensor(
                    [145, 146, 147, 148, 269, 270, 271, 272, 397, 398, 399, 400],
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                ),
                True,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                torch.tensor([21, 17, 17], dtype=torch.int32),
                torch.tensor([17, 13, 13], dtype=torch.int32),
                4,
                [],
                torch.cat(
                    [
                        torch.tensor(
                            [
                                17,
                                18,
                                19,
                                20,
                                13,
                                14,
                                15,
                                16,
                                13,
                                14,
                                15,
                                16,
                                8,
                                9,
                                10,
                                11,
                                12,
                                0,
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                11,
                                12,
                            ]
                        ),
                        torch.zeros(8704 - 30),
                    ]
                ),
                AscendAttentionState.ChunkedPrefill,
                -1,
                12,
                None,
            ),
        ],
    )
    # config: prefill and decode, Qwen3-30B, tp2, ep_enable, enforce_eager,
    # no_async_scheduling, eagle3, k=3, "disable_padded_drafter_batch": False
    @pytest.mark.parametrize("model_type", ["qwen_dense", "qwen_moe", "deepseek"])
    @patch("vllm_ascend.spec_decode.eagle_proposer.AscendEagleProposer.get_model")
    def test_propose(
        self,
        mock_get_model,
        model_type,
        flag_prefill_decode,
        query_start_loc,
        query_start_loc_cpu,
        seq_lens,
        num_reqs,
        num_actual_tokens,
        max_query_len,
        max_seq_len,
        block_table_tensor,
        slot_mapping,
        causal,
        logits_indices_padded,
        num_logits_indices,
        encoder_seq_lens,
        encoder_seq_lens_cpu,
        dcp_local_seq_lens,
        dcp_local_seq_lens_cpu,
        _seq_lens_cpu,
        _num_computed_tokens_cpu,
        _num_computed_tokens_cache,
        seq_lens_cpu,
        num_computed_tokens_cpu,
        decode_token_per_req,
        actual_seq_lengths_q,
        positions,
        attn_state,
        graph_pad_size,
        num_input_tokens,
        prefill_context_parallel_metadata,
    ):
        # mock and adjust functions and var in propose
        if model_type == "deepseek":
            self.proposer.method = "mtp"
            if not self.is_decode(flag_prefill_decode):
                num_actual_tokens = 9
        self.runner._sync_metadata_across_dp.return_value = (num_actual_tokens, None, CUDAGraphMode.NONE)
        self.proposer.model = MagicMock(spec=Eagle3LlamaForCausalLM)
        custom_combined_hidden_states = torch.zeros(num_actual_tokens, 4096, device=self.device, dtype=torch.bfloat16)
        self.proposer.model.combine_hidden_states.return_value = custom_combined_hidden_states
        mock_get_model.return_value = self.proposer.model
        self.proposer.hidden_size = 4096
        if model_type == "deepseek":
            self.proposer.hidden_states = torch.zeros(8192, 7168, device=self.device, dtype=torch.bfloat16)
        else:
            self.proposer.hidden_states = torch.zeros(8192, 4096, device=self.device, dtype=torch.bfloat16)
        mock_attn_group = MagicMock()
        mock_builder = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_builder.build.return_value = mock_attn_metadata
        mock_attn_group.get_metadata_builder.return_value = mock_builder
        self.proposer.draft_attn_groups = [mock_attn_group]
        self.proposer.attn_layer_names = ["model.layers.36.self_attn.attn"]
        self.proposer.kernel_block_size = 128
        self.proposer._runnable = MagicMock()
        self.proposer._runnable.return_value = [0, 0, 0]
        captured_common_attn_metadata = None
        original_method = self.proposer.attn_update_stack_num_spec_norm

        def side_effect(*args, **kwargs):
            nonlocal captured_common_attn_metadata
            res_common, res_attn = original_method(*args, **kwargs)
            captured_common_attn_metadata = res_common
            return res_common, res_attn

        # create common_attn_metadata
        mock_common_attn_metadata = MagicMock()
        if not self.is_decode(flag_prefill_decode):
            mock_common_attn_metadata.batch_size.return_value = 1
            if model_type == "qwen_moe":
                _seq_lens_cpu = torch.tensor([13], dtype=torch.int32)
            if model_type == "deepseek":
                query_start_loc = torch.tensor([0, 9], device=torch.device("cpu"), dtype=torch.int32)
                query_start_loc_cpu = torch.tensor([0, 9], device=torch.device("cpu"), dtype=torch.int32)
                seq_lens = torch.tensor([9], device=torch.device("cpu"), dtype=torch.int32)
                max_query_len = 9
                max_seq_len = 9
                slot_mapping = torch.tensor(
                    [128, 129, 130, 131, 132, 133, 134, 135, 136], device=torch.device("cpu"), dtype=torch.int32
                )
                _seq_lens_cpu = torch.tensor([9], dtype=torch.int32)
                seq_lens_cpu = torch.tensor([9], dtype=torch.int32)
                positions = torch.cat([torch.arange(9), torch.zeros(8704 - 9)])
                num_input_tokens = 9
        if self.is_decode(flag_prefill_decode):
            mock_common_attn_metadata.batch_size.return_value = 3
            if model_type == "qwen_moe":
                seq_lens = torch.tensor([19, 17, 17], device=torch.device("cpu"), dtype=torch.int32)
                slot_mapping = torch.tensor(
                    [143, 144, 145, 146, 269, 270, 271, 272, 397, 398, 399, 400],
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )
                seq_lens_cpu = torch.tensor([19, 17, 17], dtype=torch.int32)
                num_computed_tokens_cpu = torch.tensor([15, 13, 13], dtype=torch.int32)
                positions = torch.cat(
                    [
                        torch.tensor(
                            [
                                15,
                                16,
                                17,
                                18,
                                13,
                                14,
                                15,
                                16,
                                13,
                                14,
                                15,
                                16,
                                8,
                                9,
                                10,
                                11,
                                12,
                                0,
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                11,
                                12,
                            ]
                        ),
                        torch.zeros(8704 - 30),
                    ]
                )
            if model_type == "deepseek":
                seq_lens = torch.tensor([14, 13, 14], device=torch.device("cpu"), dtype=torch.int32)
                slot_mapping = torch.tensor(
                    [138, 139, 140, 141, 265, 266, 267, 268, 394, 395, 396, 397],
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )
                seq_lens_cpu = torch.tensor([14, 13, 14], dtype=torch.int32)
                num_computed_tokens_cpu = torch.tensor([10, 9, 10], dtype=torch.int32)
                positions = torch.cat(
                    [
                        torch.tensor([10, 11, 12, 13, 9, 10, 11, 12, 10, 11, 12, 13, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                        torch.zeros(8704 - 23),
                    ]
                )
                attn_state = AscendAttentionState.SpecDecoding
        self.value_mock_common_attn_metadata(
            mock_common_attn_metadata,
            query_start_loc,
            query_start_loc_cpu,
            seq_lens,
            num_reqs,
            num_actual_tokens,
            max_query_len,
            max_seq_len,
            block_table_tensor,
            slot_mapping,
            causal,
            logits_indices_padded,
            num_logits_indices,
            encoder_seq_lens,
            encoder_seq_lens_cpu,
            dcp_local_seq_lens,
            dcp_local_seq_lens_cpu,
            _seq_lens_cpu,
            _num_computed_tokens_cpu,
            _num_computed_tokens_cache,
            seq_lens_cpu,
            num_computed_tokens_cpu,
            decode_token_per_req,
            actual_seq_lengths_q,
            positions,
            attn_state,
            graph_pad_size,
            num_input_tokens,
            prefill_context_parallel_metadata,
        )

        # create other parameters
        if not self.is_decode(flag_prefill_decode):
            if model_type == "qwen_dense" or model_type == "qwen_moe":
                target_token_ids = torch.tensor(
                    [151644, 872, 198, 5501, 7512, 14678, 51765, 30, 151645, 198, 151644, 77091, 198],
                    device=self.device,
                    dtype=torch.int32,
                )
                target_positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], device=self.device)
                next_token_ids = torch.tensor([151667], device=self.device, dtype=torch.int32)
                req_scheduled_tokens = {"0-8222703c": 13}
            if model_type == "deepseek":
                target_token_ids = torch.tensor(
                    [0, 0, 128803, 12473, 9734, 19991, 50096, 33, 128804], device=self.device, dtype=torch.int32
                )
                target_positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], device=self.device)
                target_hidden_states = torch.zeros(num_actual_tokens, 7168, device=self.device, dtype=torch.bfloat16)
                next_token_ids = torch.tensor([128798], device=self.device, dtype=torch.int32)
                req_scheduled_tokens = {"0-b4ed8210": 9}
            if model_type == "qwen_dense":
                target_hidden_states = torch.zeros(num_actual_tokens, 12288, device=self.device, dtype=torch.bfloat16)
            if model_type == "qwen_moe":
                target_hidden_states = torch.zeros(num_actual_tokens, 6144, device=self.device, dtype=torch.bfloat16)
            token_indices_to_sample = None
            target_model_batch_desc = BatchDescriptor(
                num_tokens=num_actual_tokens, num_reqs=None, uniform=False, has_lora=False, num_active_loras=0
            )
            mock_sampling_metadata = MagicMock()
            mm_embed_inputs = None
            long_seq_metadata = None
            num_prefill_reqs = 0
            num_decode_reqs = 0
            scheduler_output = MagicMock()
            num_scheduled_tokens = num_actual_tokens
            num_rejected_tokens_gpu = None

        if self.is_decode(flag_prefill_decode):
            if model_type == "qwen_dense":
                target_token_ids = torch.tensor(
                    [279, 1196, 374, 8014, 151667, 198, 32313, 11, 151667, 198, 32313, 11],
                    device=self.device,
                    dtype=torch.int32,
                )
                target_positions = torch.tensor([17, 18, 19, 20, 13, 14, 15, 16, 13, 14, 15, 16], device=self.device)
                target_hidden_states = torch.zeros(num_actual_tokens, 12288, device=self.device, dtype=torch.bfloat16)
                next_token_ids = torch.tensor([4588, 279, 279], device=self.device, dtype=torch.int32)
                token_indices_to_sample = torch.tensor([1, 7, 11], device=self.device, dtype=torch.int32)
                num_rejected_tokens_gpu = torch.tensor([2, 0, 0], device=self.device, dtype=torch.int32)
            if model_type == "qwen_moe":
                target_token_ids = torch.tensor(
                    [32313, 2776, 198, 198, 151667, 198, 198, 198, 151667, 198, 198, 198],
                    device=self.device,
                    dtype=torch.int32,
                )
                target_positions = torch.tensor([15, 16, 17, 18, 13, 14, 15, 16, 13, 14, 15, 16], device=self.device)
                target_hidden_states = torch.zeros(num_actual_tokens, 6144, device=self.device, dtype=torch.bfloat16)
                next_token_ids = torch.tensor([11, 32313, 32313], device=self.device, dtype=torch.int32)
                token_indices_to_sample = torch.tensor([0, 5, 9], device=self.device, dtype=torch.int32)
                num_rejected_tokens_gpu = torch.tensor([3, 2, 2], device=self.device, dtype=torch.int32)
            if model_type == "deepseek":
                target_token_ids = torch.tensor(
                    [201, 33001, 14, 832, 128798, 271, 5, 128798, 128798, 271, 5, 128798],
                    device=self.device,
                    dtype=torch.int32,
                )
                target_positions = torch.tensor([10, 11, 12, 13, 9, 10, 11, 12, 10, 11, 12, 13], device=self.device)
                target_hidden_states = torch.zeros(num_actual_tokens, 7168, device=self.device, dtype=torch.bfloat16)
                next_token_ids = torch.tensor([270, 128799, 201], device=self.device, dtype=torch.int32)
                token_indices_to_sample = torch.tensor([2, 5, 8], device=self.device, dtype=torch.int32)
                num_rejected_tokens_gpu = torch.tensor([1, 2, 3], device=self.device, dtype=torch.int32)
            target_model_batch_desc = BatchDescriptor(
                num_tokens=num_actual_tokens, num_reqs=None, uniform=False, has_lora=False, num_active_loras=0
            )
            mock_sampling_metadata = MagicMock()
            mm_embed_inputs = None
            req_scheduled_tokens = {"0-b69afbe5": 4, "1-b60368b9": 4, "2-82281e95": 4}
            long_seq_metadata = None
            num_prefill_reqs = 0
            num_decode_reqs = 0
            scheduler_output = MagicMock()
            num_scheduled_tokens = num_actual_tokens

        # run
        with (
            patch.object(self.proposer, "attn_update_stack_num_spec_norm", side_effect=side_effect),
            set_current_vllm_config(self.vllm_config),
        ):
            self.proposer._propose(
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
                mock_common_attn_metadata,
                target_model_batch_desc,
                mock_sampling_metadata,
                mm_embed_inputs,
                req_scheduled_tokens,
                long_seq_metadata,
                num_prefill_reqs,
                num_decode_reqs,
                scheduler_output,
                num_scheduled_tokens,
                num_rejected_tokens_gpu,
            )
            self.assert_value_common_attn_metadata(captured_common_attn_metadata, flag_prefill_decode, model_type)

    # give common_attn_metadata value
    def value_mock_common_attn_metadata(
        self,
        mock_common_attn_metadata,
        query_start_loc,
        query_start_loc_cpu,
        seq_lens,
        num_reqs,
        num_actual_tokens,
        max_query_len,
        max_seq_len,
        block_table_tensor,
        slot_mapping,
        causal,
        logits_indices_padded,
        num_logits_indices,
        encoder_seq_lens,
        encoder_seq_lens_cpu,
        dcp_local_seq_lens,
        dcp_local_seq_lens_cpu,
        _seq_lens_cpu,
        _num_computed_tokens_cpu,
        _num_computed_tokens_cache,
        seq_lens_cpu,
        num_computed_tokens_cpu,
        decode_token_per_req,
        actual_seq_lengths_q,
        positions,
        attn_state,
        graph_pad_size,
        num_input_tokens,
        prefill_context_parallel_metadata,
    ):
        mock_common_attn_metadata.query_start_loc = query_start_loc
        mock_common_attn_metadata.query_start_loc_cpu = query_start_loc_cpu
        mock_common_attn_metadata.seq_lens = seq_lens
        mock_common_attn_metadata.num_reqs = num_reqs
        mock_common_attn_metadata.num_actual_tokens = num_actual_tokens
        mock_common_attn_metadata.max_query_len = max_query_len
        mock_common_attn_metadata.max_seq_len = max_seq_len
        mock_common_attn_metadata.block_table_tensor = block_table_tensor
        mock_common_attn_metadata.slot_mapping = slot_mapping
        mock_common_attn_metadata.causal = causal
        mock_common_attn_metadata.logits_indices_padded = logits_indices_padded
        mock_common_attn_metadata.num_logits_indices = num_logits_indices
        mock_common_attn_metadata.encoder_seq_lens = encoder_seq_lens
        mock_common_attn_metadata.encoder_seq_lens_cpu = encoder_seq_lens_cpu
        mock_common_attn_metadata.dcp_local_seq_lens = dcp_local_seq_lens
        mock_common_attn_metadata.dcp_local_seq_lens_cpu = dcp_local_seq_lens_cpu
        mock_common_attn_metadata._seq_lens_cpu = _seq_lens_cpu
        mock_common_attn_metadata._num_computed_tokens_cpu = _num_computed_tokens_cpu
        mock_common_attn_metadata._num_computed_tokens_cache = _num_computed_tokens_cache
        mock_common_attn_metadata.seq_lens_cpu = seq_lens_cpu
        mock_common_attn_metadata.num_computed_tokens_cpu = num_computed_tokens_cpu
        mock_common_attn_metadata.decode_token_per_req = decode_token_per_req
        mock_common_attn_metadata.actual_seq_lengths_q = actual_seq_lengths_q
        mock_common_attn_metadata.positions = positions
        mock_common_attn_metadata.attn_state = attn_state
        mock_common_attn_metadata.graph_pad_size = graph_pad_size
        mock_common_attn_metadata.num_input_tokens = num_input_tokens
        mock_common_attn_metadata.prefill_context_parallel_metadata = prefill_context_parallel_metadata

    # assert the value common_attn_metadata
    def assert_value_common_attn_metadata(self, captured_common_attn_metadata, flag_prefill_decode, model_type):
        if not self.is_decode(flag_prefill_decode):
            assert torch.equal(captured_common_attn_metadata.query_start_loc, torch.tensor([0, 1]))
            assert torch.equal(captured_common_attn_metadata.query_start_loc_cpu, torch.tensor([0, 1]))
            assert captured_common_attn_metadata.num_reqs == 1
            assert captured_common_attn_metadata.num_actual_tokens == 1
            assert captured_common_attn_metadata.max_query_len == 1
            assert torch.equal(
                captured_common_attn_metadata.block_table_tensor, torch.eye(256, dtype=torch.int32)[0].unsqueeze(0)
            )
            assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([2]))
            if model_type == "qwen_moe":
                assert captured_common_attn_metadata._seq_lens_cpu == torch.tensor([15])
            if model_type == "qwen_dense":
                assert captured_common_attn_metadata._seq_lens_cpu is None
            if model_type == "deepseek":
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([11]))
                assert captured_common_attn_metadata.max_seq_len == 9
                assert torch.equal(
                    captured_common_attn_metadata.slot_mapping,
                    torch.cat([torch.tensor([138]), torch.full((8703,), -1)]),
                )
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([11]))
                assert captured_common_attn_metadata._seq_lens_cpu == torch.tensor([11])
                assert torch.equal(
                    captured_common_attn_metadata.positions,
                    torch.tensor([10, 1, 2, 3, 4, 5, 6, 7, 8] + [0] * (8704 - 9), dtype=torch.int64),
                )
                assert captured_common_attn_metadata.num_input_tokens == 9
            if model_type == "qwen_dense" or model_type == "qwen_moe":
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([15]))
                assert captured_common_attn_metadata.max_seq_len == 13
                assert torch.equal(
                    captured_common_attn_metadata.slot_mapping,
                    torch.cat([torch.tensor([142]), torch.full((8703,), -1)]),
                )
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([15]))
                assert torch.equal(
                    captured_common_attn_metadata.positions,
                    torch.tensor([14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + [0] * (8704 - 13), dtype=torch.int64),
                )
                assert captured_common_attn_metadata.num_input_tokens == 13

        if self.is_decode(flag_prefill_decode):
            assert torch.equal(captured_common_attn_metadata.query_start_loc, torch.tensor([0, 1, 2, 3]))
            assert torch.equal(captured_common_attn_metadata.query_start_loc_cpu, torch.tensor([0, 1, 2, 3]))
            assert captured_common_attn_metadata.num_input_tokens == 12
            assert captured_common_attn_metadata.num_reqs == 3
            assert captured_common_attn_metadata.num_actual_tokens == 3
            assert captured_common_attn_metadata.max_query_len == 1
            assert captured_common_attn_metadata.max_seq_len == 0
            assert torch.equal(
                captured_common_attn_metadata.block_table_tensor,
                torch.cat(
                    [torch.eye(256, device="cpu", dtype=torch.int32)[0].unsqueeze(0) * i for i in [1, 2, 3]], dim=0
                ),
            )
            assert captured_common_attn_metadata._seq_lens_cpu is None
            if model_type == "qwen_dense":
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([23, 19, 19]))
                assert torch.equal(
                    captured_common_attn_metadata.slot_mapping,
                    torch.cat([torch.tensor([148, 274, 402]), torch.full((8701,), -1)]),
                )
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([23, 19, 19]))
                assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([19, 15, 15]))
                assert torch.equal(
                    captured_common_attn_metadata.positions,
                    torch.tensor(
                        [
                            20,
                            18,
                            18,
                            20,
                            13,
                            14,
                            15,
                            16,
                            13,
                            14,
                            15,
                            16,
                            8,
                            9,
                            10,
                            11,
                            12,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                        ]
                        + [0] * (8704 - 30),
                        dtype=torch.int64,
                    ),
                )
            if model_type == "qwen_moe":
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([21, 19, 19]))
                assert torch.equal(
                    captured_common_attn_metadata.slot_mapping,
                    torch.cat([torch.tensor([145, 272, 400]), torch.full((8701,), -1)]),
                )
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([21, 19, 19]))
                assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([17, 15, 15]))
                assert torch.equal(
                    captured_common_attn_metadata.positions,
                    torch.tensor(
                        [
                            17,
                            16,
                            16,
                            18,
                            13,
                            14,
                            15,
                            16,
                            13,
                            14,
                            15,
                            16,
                            8,
                            9,
                            10,
                            11,
                            12,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                        ]
                        + [0] * (8704 - 30),
                        dtype=torch.int64,
                    ),
                )
            if model_type == "deepseek":
                assert torch.equal(captured_common_attn_metadata.seq_lens, torch.tensor([16, 15, 16]))
                assert torch.equal(
                    captured_common_attn_metadata.slot_mapping,
                    torch.cat([torch.tensor([142, 268, 396]), torch.full((8701,), -1)]),
                )
                assert torch.equal(captured_common_attn_metadata.seq_lens_cpu, torch.tensor([16, 15, 16]))
                assert torch.equal(captured_common_attn_metadata.num_computed_tokens_cpu, torch.tensor([12, 11, 12]))
                assert torch.equal(
                    captured_common_attn_metadata.positions,
                    torch.tensor(
                        [14, 12, 12, 13, 9, 10, 11, 12, 10, 11, 12, 13, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                        + [0] * (8704 - 23),
                        dtype=torch.int64,
                    ),
                )
        assert captured_common_attn_metadata.causal
        assert captured_common_attn_metadata.logits_indices_padded is None
        assert captured_common_attn_metadata.num_logits_indices is None
        assert captured_common_attn_metadata.encoder_seq_lens is None
        assert captured_common_attn_metadata.encoder_seq_lens_cpu is None
        assert captured_common_attn_metadata.dcp_local_seq_lens is None
        assert captured_common_attn_metadata.dcp_local_seq_lens_cpu is None
        assert captured_common_attn_metadata._num_computed_tokens_cpu is None
        assert captured_common_attn_metadata._num_computed_tokens_cache is None
        assert captured_common_attn_metadata.decode_token_per_req == 1
        assert captured_common_attn_metadata.actual_seq_lengths_q == []
        if model_type == "deepseek":
            assert captured_common_attn_metadata.attn_state == AscendAttentionState.SpecDecoding
        else:
            assert captured_common_attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
        assert captured_common_attn_metadata.graph_pad_size == -1
        assert captured_common_attn_metadata.prefill_context_parallel_metadata is None

    # prefill or decode
    def is_decode(self, flag_prefill_decode):
        if flag_prefill_decode == "decode":
            return True
        if flag_prefill_decode == "prefill":
            return False


class TestEagleProposerSetInputsFirstPass(TestBase):
    """Test set_inputs_first_pass for AscendEagleProposer.

    This test class covers all branches of set_inputs_first_pass:

    Branch coverage:
    - Branch 1 (needs_extra_input_slots=False): Default EAGLE pathway
      - Branch 1.1: multiple requests
      - Branch 1.2: pcp_size > 1 (PCP split logic) - vllm-ascend specific
    - Branch 2 (needs_extra_input_slots=True): Draft model / Parallel drafting
      - Branch 2.1: shift_input_ids=False (draft_model)
      - Branch 2.2: shift_input_ids=True (parallel_drafting)

    """

    def setUp(self):
        self.device = torch.device(current_platform.device_type)
        self.runner = MagicMock()
        self.runner.pin_memory = False
        self.runner.pcp_size = 1
        self.runner.dcp_size = 1
        self.runner.max_num_tokens = 8192
        self.runner.max_num_reqs = 256

        self.mock_cpugpubuffer = patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer")
        self.mock_cpugpubuffer.start()
        self.mock_supports_multimodal_inputs = patch(
            "vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False
        )
        self.mock_supports_multimodal_inputs.start()

    def tearDown(self):
        self.mock_cpugpubuffer.stop()
        self.mock_supports_multimodal_inputs.stop()

    def _create_base_vllm_config(self):
        """Create base vllm_config with common settings shared across all tests."""
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.cache_config = MagicMock(spec=CacheConfig)
        vllm_config.cache_config.block_size = BLOCK_SIZE
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.max_num_batched_tokens = 1024
        vllm_config.scheduler_config.max_num_seqs = 32
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.hf_text_config = MagicMock(spec=[])
        vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.max_model_len = 2048
        vllm_config.model_config.uses_mrope = False
        vllm_config.model_config.uses_xdrope_dim = 0
        vllm_config.compilation_config = MagicMock()
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.data_parallel_rank = 0
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.enable_expert_parallel = False
        vllm_config.additional_config = None
        return vllm_config

    def _create_speculative_config(
        self,
        method: str,
        num_speculative_tokens: int,
        parallel_drafting: bool = False,
    ):
        """Create speculative_config for specific method."""
        speculative_config = MagicMock()
        speculative_config.method = method
        speculative_config.parallel_drafting = parallel_drafting
        speculative_config.num_speculative_tokens = num_speculative_tokens
        speculative_config.speculative_token_tree = str([(i + 1) * (0,) for i in range(num_speculative_tokens)])
        speculative_config.draft_tensor_parallel_size = 1
        speculative_config.disable_padded_drafter_batch = False
        speculative_config.draft_model_config = MagicMock()
        speculative_config.draft_model_config.get_hidden_size.return_value = 4096
        speculative_config.draft_model_config.uses_xdrope_dim = 0
        speculative_config.draft_model_config.uses_mrope = False
        speculative_config.target_parallel_config = MagicMock()
        speculative_config.target_parallel_config.tensor_parallel_size = 1
        speculative_config.draft_parallel_config = MagicMock()
        speculative_config.draft_parallel_config.tensor_parallel_size = 1
        return speculative_config

    def _create_proposer(
        self,
        method: str,
        num_speculative_tokens: int,
        parallel_drafting: bool = False,
        device: torch.device = None,
        runner=None,
    ):
        """Create a proposer instance for testing."""
        if device is None:
            device = torch.device(current_platform.device_type)
        vllm_config = self._create_base_vllm_config()
        vllm_config.speculative_config = self._create_speculative_config(
            method=method,
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=parallel_drafting,
        )

        init_ascend_config(vllm_config)

        with (
            patch("vllm.v1.spec_decode.eagle.CpuGpuBuffer"),
            patch("vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False),
            set_current_vllm_config(vllm_config),
        ):
            if method == "eagle":
                proposer = AscendEagleProposer(
                    vllm_config=vllm_config,
                    device=device,
                    runner=runner,
                )
            elif method == "draft_model":
                proposer = AscendDraftModelProposer(
                    vllm_config=vllm_config,
                    device=device,
                    runner=runner,
                )
            proposer.block_size = BLOCK_SIZE
            return proposer, vllm_config

    def test_set_inputs_first_pass_default_eagle(self):
        """
        Test for set_inputs_first_pass without extra input slots (default EAGLE).

        This tests the path where needs_extra_input_slots=False, which is the
        default EAGLE pathway. In this case:
        - Input IDs are rotated (shifted by one)
        - The next_token_ids are inserted at the last position of each request
        - Positions are copied as-is
        - Hidden states are copied as-is
        - The CommonAttentionMetadata is returned unchanged

        Setup:
        - 3 requests with query_lens [3, 2, 4]
        - Tokens: [a1, a2, a3, b1, b2, c1, c2, c3, c4]
        - After rotation: [a2, a3, -, b2, -, c2, c3, c4, -]
        - After inserting next_tokens [100, 200, 300]:
            [a2, a3, 100, b2, 200, c2, c3, c4, 300]
        """
        num_speculative_tokens = 3
        block_size = BLOCK_SIZE

        self.proposer, self.vllm_config = self._create_proposer(
            method="eagle",
            num_speculative_tokens=num_speculative_tokens,
            device=self.device,
            runner=self.runner,
        )

        batch_spec = BatchSpec(
            seq_lens=[10, 8, 12],
            query_lens=[3, 2, 4],
        )

        common_attn_metadata = create_common_attn_metadata(
            batch_spec,
            block_size=block_size,
            device=self.device,
        )

        self.proposer.needs_extra_input_slots = False

        target_token_ids = torch.tensor([10, 11, 12, 20, 21, 30, 31, 32, 33], dtype=torch.int32, device=self.device)
        target_positions = torch.tensor([7, 8, 9, 6, 7, 8, 9, 10, 11], dtype=torch.int64, device=self.device)
        target_hidden_states = torch.randn(9, self.proposer.hidden_size, dtype=self.proposer.dtype, device=self.device)
        next_token_ids = torch.tensor([100, 200, 300], dtype=torch.int32, device=self.device)

        with set_current_vllm_config(self.vllm_config):
            out_num_tokens, out_token_indices, out_cad, long_seq_args = self.proposer.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=None,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=None,
            )

        self.assertEqual(out_num_tokens, 9)

        expected_token_indices = torch.tensor([2, 4, 8], dtype=torch.int32, device=self.device)
        self.assertTrue(torch.equal(out_token_indices, expected_token_indices))

        self.assertIs(out_cad, common_attn_metadata)

        expected_input_ids = torch.tensor(
            [11, 12, 100, 21, 200, 31, 32, 33, 300], dtype=torch.int32, device=self.device
        )
        self.assertTrue(torch.equal(self.proposer.input_ids[:out_num_tokens], expected_input_ids))
        self.assertTrue(torch.equal(self.proposer.positions[:out_num_tokens], target_positions))
        self.assertTrue(torch.equal(self.proposer.hidden_states[:out_num_tokens], target_hidden_states))

    def test_set_inputs_first_pass_pcp_dcp_mixed(self):
        """
        Test Default pcp_dcp_mixed scenario
        """
        self.proposer, self.vllm_config = self._create_proposer(
            method="eagle",
            num_speculative_tokens=3,
            device=self.device,
            runner=self.runner,
        )

        self.proposer.pcp_size = 2
        self.proposer.dcp_size = 2
        self.proposer.pcp_rank = 0
        self.proposer.needs_extra_input_slots = False

        num_decode_reqs = 2
        num_prefill_reqs = 2

        req_ids = ["req-0", "req-1", "req-2", "req-3"]
        req_scheduled_tokens = {"req-0": 3, "req-1": 2, "req-2": 4, "req-3": 3}
        query_lens = [3, 2, 4, 3]

        self.runner.query_lens = torch.tensor(query_lens[:num_decode_reqs], dtype=torch.int32, device=self.device)
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.req_ids = req_ids
        self.runner.logits_indices = torch.arange(12, dtype=torch.int32, device=self.device)

        query_start_loc = torch.tensor([0] + list(np.cumsum(query_lens)), dtype=torch.int32, device=self.device)
        mock_common_attn_metadata = MagicMock()
        mock_common_attn_metadata.query_start_loc = query_start_loc
        mock_common_attn_metadata.query_start_loc_cpu = query_start_loc.clone()
        mock_common_attn_metadata.num_reqs = 4
        mock_common_attn_metadata.num_actual_tokens = 12

        mock_common_attn_metadata.seq_lens = torch.tensor([10, 8, 12, 6], dtype=torch.int32, device=self.device)
        mock_common_attn_metadata.seq_lens_cpu = mock_common_attn_metadata.seq_lens.clone()

        mock_common_attn_metadata.slot_mapping = torch.zeros(12, dtype=torch.int32, device=self.device)

        target_token_ids = torch.tensor(
            [10, 11, 12, 20, 21, 30, 31, 32, 33, 40, 41, 42], dtype=torch.int32, device=self.device
        )

        target_positions = torch.tensor([7, 8, 9, 6, 7, 8, 9, 10, 11, 5, 6, 7], dtype=torch.int64, device=self.device)

        next_token_ids = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=self.device)

        target_hidden_states = torch.randn(18, self.proposer.hidden_size, dtype=self.proposer.dtype, device=self.device)

        with set_current_vllm_config(self.vllm_config):
            out_num_tokens, out_token_indices, out_cad, (query_lens_d, ori_token_indices_to_sample) = (
                self.proposer.set_inputs_first_pass(
                    target_token_ids=target_token_ids,
                    next_token_ids=next_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    token_indices_to_sample=None,
                    cad=mock_common_attn_metadata,
                    num_rejected_tokens_gpu=None,
                    req_scheduled_tokens=req_scheduled_tokens,
                    long_seq_metadata=MagicMock(),
                    num_prefill_reqs=num_prefill_reqs,
                    num_decode_reqs=num_decode_reqs,
                )
            )

        self.assertEqual(out_num_tokens, 9)

        expected_token_indices = torch.tensor([2, 4, 10, 11], dtype=torch.int32, device=self.device)
        self.assertTrue(torch.equal(out_token_indices, expected_token_indices))

        expected_seq_lens = torch.tensor([10, 8, 2, 2], dtype=torch.int32, device=self.device)
        expected_query_start_loc = torch.tensor([0, 3, 5, 7, 9], dtype=torch.int32, device=self.device)
        self.assertIs(out_cad, mock_common_attn_metadata)
        self.assertEqual(out_cad.num_actual_tokens, 9)
        self.assertTrue(torch.equal(out_cad.seq_lens, expected_seq_lens))
        self.assertTrue(torch.equal(out_cad.query_start_loc, expected_query_start_loc))
        self.assertEqual(out_cad.max_query_len, 4)

        expected_query_lens_d = torch.tensor([3, 2], dtype=torch.int32, device=self.device)
        expected_ori_token_indices_to_sample = torch.tensor([2, 4, 8, 11], dtype=torch.int32, device=self.device)
        self.assertTrue(torch.equal(query_lens_d, expected_query_lens_d))
        self.assertTrue(torch.equal(ori_token_indices_to_sample, expected_ori_token_indices_to_sample))

        expected_input_ids = torch.tensor([11, 12, 100, 21, 200, 31, 300, 41, 0], dtype=torch.int32, device=self.device)
        indices = torch.tensor([0, 1, 2, 6, 7, 10, 13, 14, 17], dtype=torch.long, device=self.device)
        expected_target_hidden_states = target_hidden_states[indices]
        self.assertTrue(torch.equal(self.proposer.input_ids[:out_num_tokens], expected_input_ids))
        self.assertTrue(torch.equal(self.proposer.positions[:out_num_tokens], target_positions[:out_num_tokens]))
        self.assertTrue(torch.equal(self.proposer.hidden_states[:out_num_tokens], expected_target_hidden_states))

    def test_set_inputs_first_pass_draft_model(self):
        """
        Test for set_inputs_first_pass with a draft model (extra input slots,
        no shift).

        This tests the path where needs_extra_input_slots=True and
        shift_input_ids=False (draft model case). In this case:
        - Input IDs are NOT shifted
        - Each request gets extra_slots_per_request (1) new slots
        - The kernel handles copying tokens and inserting bonus/padding tokens
        - A new CommonAttentionMetadata is returned with updated query_start_loc

        Setup:
        - 2 requests
        - Request 0: tokens [10, 11, 12] at positions [0, 1, 2]
        - Only tokens [10, 11] are "valid" (query_end_loc=1),
            token 12 is a rejected token from previous speculation
        - Request 1: tokens [20, 21] at positions [0, 1], both valid.
        - Note: this is less than num_speculative_tokens (2) to ensure
            we handle variable lengths correctly.
        - next_token_ids: [100, 200] (bonus tokens)

        With extra_slots_per_request=1 and shift=False:
        Expected output layout:
        Request 0 (indices 0-3):
        - idx 0: token 10, pos 0
        - idx 1: token 11, pos 1
        - idx 2: token 100, pos 2 (bonus token)
        - idx 3: padding_token_id, is_rejected=True
        Request 1 (indices 4-6):
        - idx 4: token 20, pos 0
        - idx 5: token 21, pos 1
        - idx 6: token 200, pos 2 (bonus token)
        """
        num_speculative_tokens = 2
        block_size = BLOCK_SIZE

        proposer, vllm_config = self._create_proposer(
            method="draft_model",
            num_speculative_tokens=num_speculative_tokens,
            device=self.device,
            runner=self.runner,
        )
        proposer.net_num_new_slots_per_request = 1
        proposer.needs_extra_input_slots = True

        proposer.parallel_drafting_token_id = 0
        proposer.is_rejected_token_mask = torch.zeros(proposer.max_num_tokens, dtype=torch.bool, device=self.device)
        proposer.is_masked_token_mask = torch.zeros(proposer.max_num_tokens, dtype=torch.bool, device=self.device)

        mock_kv_cache_spec = MagicMock()
        mock_kv_cache_spec.block_size = block_size
        mock_attn_group = MagicMock()
        mock_attn_group.kv_cache_spec = mock_kv_cache_spec
        proposer.draft_attn_groups = [mock_attn_group]

        batch_spec = BatchSpec(
            seq_lens=[3, 2],
            query_lens=[3, 2],
        )

        common_attn_metadata = create_common_attn_metadata(
            batch_spec,
            block_size=block_size,
            device=self.device,
        )

        target_token_ids = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32, device=self.device)
        target_positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64, device=self.device)
        target_hidden_states = torch.randn(5, proposer.hidden_size, dtype=proposer.dtype, device=self.device)
        next_token_ids = torch.tensor([100, 200], dtype=torch.int32, device=self.device)
        num_rejected_tokens_gpu = torch.tensor([1, 0], dtype=torch.int32, device=self.device)

        with set_current_vllm_config(vllm_config):
            out_num_tokens, out_token_indices, out_cad, long_seq_args = proposer.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=None,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )

        self.assertEqual(proposer.net_num_new_slots_per_request, 1)
        self.assertTrue(proposer.needs_extra_input_slots)
        self.assertEqual(out_num_tokens, 7)

        expected_input_ids = torch.tensor([10, 11, 100, 0, 20, 21, 200], dtype=torch.int32, device=self.device)
        self.assertTrue(torch.equal(proposer.input_ids[:out_num_tokens], expected_input_ids))

        expected_positions = torch.tensor([0, 1, 2, 0, 0, 1, 2], dtype=torch.int64, device=self.device)
        self.assertTrue(
            torch.equal(
                proposer.positions[:out_num_tokens],
                expected_positions,
            )
        )

        expected_is_rejected = torch.zeros(7, dtype=torch.bool, device=self.device)
        expected_is_rejected[3] = True
        self.assertTrue(torch.equal(proposer.is_rejected_token_mask[:out_num_tokens], expected_is_rejected))

        expected_is_masked = torch.zeros(7, dtype=torch.bool, device=self.device)
        self.assertTrue(torch.equal(proposer.is_masked_token_mask[:out_num_tokens], expected_is_masked))

        expected_out_token_indices = torch.tensor([2, 6], dtype=torch.int32, device=self.device)
        self.assertTrue(torch.equal(out_token_indices, expected_out_token_indices))

        expected_query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32, device=self.device)
        self.assertTrue(torch.equal(out_cad.query_start_loc, expected_query_start_loc))

    def test_set_inputs_first_pass_parallel_drafting(self):
        """
        Test for set_inputs_first_pass with parallel drafting (extra input slots,
        with shift).

        This tests the path where needs_extra_input_slots=True and
        shift_input_ids=True (parallel drafting case). In this case:
        - Input IDs ARE shifted (like default EAGLE)
        - Each request gets extra_slots_per_request (3) new slots
        - Parallel drafting tokens are inserted and marked as masked
        - Hidden states are mapped correctly

        Setup:
        - 2 requests with query_lens [4, 4] (1 bonus + 3 spec tokens each)
        - Request 0: tokens [10, 11, 12, 13] at positions [5, 6, 7, 8]
        - Only tokens [10, 11, 12] are "valid", token 13 is rejected
        - Request 1: tokens [20, 21, 22, 23] at positions [10, 11, 12, 13], all valid.
        - next_token_ids: [100, 200] (bonus tokens)

        With shift_input_ids=True, extra_slots_per_request=3:
        Expected output layout:
        Request 0 (6 output slots = 4 - 1 + 3):
        - idx 0-2: shifted tokens [11, 12, 100]
        - idx 3-4: parallel_drafting_tokens, is_masked=True
        - idx 5: padding_token, is_rejected=True
        Request 1 (6 output slots = 4 - 1 + 3):
        - idx 6-8: shifted tokens [21, 22, 23]
        - idx 9: bonus token 200
        - idx 10-11: parallel_drafting_tokens, is_masked=True
        """
        num_speculative_tokens = 3
        block_size = BLOCK_SIZE

        proposer, vllm_config = self._create_proposer(
            method="eagle",
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=True,
            device=self.device,
            runner=self.runner,
        )

        self.assertTrue(proposer.pass_hidden_states_to_model)
        self.assertTrue(proposer.needs_extra_input_slots)

        proposer.parallel_drafting_token_id = -2
        proposer.parallel_drafting_hidden_state_tensor = torch.zeros(
            proposer.hidden_size, dtype=proposer.dtype, device=self.device
        )
        proposer.is_rejected_token_mask = torch.zeros(proposer.max_num_tokens, dtype=torch.bool, device=self.device)
        proposer.is_masked_token_mask = torch.zeros(proposer.max_num_tokens, dtype=torch.bool, device=self.device)

        mock_kv_cache_spec = MagicMock()
        mock_kv_cache_spec.block_size = block_size
        mock_attn_group = MagicMock()
        mock_attn_group.kv_cache_spec = mock_kv_cache_spec
        proposer.draft_attn_groups = [mock_attn_group]

        batch_spec = BatchSpec(
            seq_lens=[9, 14],
            query_lens=[4, 4],
        )

        common_attn_metadata = create_common_attn_metadata(
            batch_spec,
            block_size=block_size,
            device=self.device,
            arange_block_indices=True,
        )

        target_token_ids = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int32, device=self.device)
        target_positions = torch.tensor([5, 6, 7, 8, 10, 11, 12, 13], dtype=torch.int64, device=self.device)
        target_hidden_states = torch.arange(8 * proposer.hidden_size, dtype=proposer.dtype, device=self.device).view(
            8, proposer.hidden_size
        )
        next_token_ids = torch.tensor([100, 200], dtype=torch.int32, device=self.device)
        num_rejected_tokens_gpu = torch.tensor([1, 0], dtype=torch.int32, device=self.device)

        with set_current_vllm_config(vllm_config):
            out_num_tokens, out_token_indices, out_cad, long_seq_args = proposer.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=None,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )

        self.assertEqual(out_num_tokens, 12)

        expected_input_ids = torch.tensor(
            [11, 12, 100, -2, -2, 0, 21, 22, 23, 200, -2, -2],
            dtype=torch.int32,
            device=self.device,
        )
        self.assertTrue(torch.equal(proposer.input_ids[:out_num_tokens], expected_input_ids))

        expected_positions = torch.tensor(
            [5, 6, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15], dtype=torch.int64, device=self.device
        )
        self.assertTrue(
            torch.equal(
                proposer.positions[:out_num_tokens],
                expected_positions,
            )
        )

        expected_is_rejected = torch.zeros(12, dtype=torch.bool, device=self.device)
        expected_is_rejected[5] = True
        self.assertTrue(torch.equal(proposer.is_rejected_token_mask[:out_num_tokens], expected_is_rejected))

        expected_is_masked = torch.zeros(12, dtype=torch.bool, device=self.device)
        expected_is_masked[3] = True
        expected_is_masked[4] = True
        expected_is_masked[10] = True
        expected_is_masked[11] = True
        self.assertTrue(torch.equal(proposer.is_masked_token_mask[:out_num_tokens], expected_is_masked))

        expected_out_token_indices = torch.tensor([2, 3, 4, 9, 10, 11], dtype=torch.int32, device=self.device)
        self.assertTrue(torch.equal(out_token_indices, expected_out_token_indices))

        expected_query_start_loc = torch.tensor([0, 6, 12], dtype=torch.int32, device=self.device)
        self.assertTrue(torch.equal(out_cad.query_start_loc, expected_query_start_loc))

        parallel_drafting_hs = proposer.parallel_drafting_hidden_state_tensor
        for i in range(out_num_tokens):
            if expected_is_masked[i]:
                self.assertTrue(
                    torch.equal(proposer.hidden_states[i], parallel_drafting_hs),
                    f"Masked position {i} should have parallel drafting hidden state",
                )
