"""SGLang custom model: SMLR fused multi-lane VLA served from ONE shared Llama backbone.

ONE backbone forward per step; each request is routed (by sampling_params.custom_params["lane"])
to its own decoder head. Greedy per-lane decode shares the frame prompt prefix via RadixAttention.

Registration: this module lives inside `sglang.srt.models`, so the registry's package scan
(import_model_classes) picks up `EntryClass` in every worker process -> no cross-process bug (#11578).
Checkpoint config must set architectures=["SmlrMultiLaneForCausalLM"] and smlr_lanes=[...].
"""
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.layers.logits_processor import LogitsProcessorOutput, LogitsMetadata
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)

# fixed policy label order (must match runtime.vla_client.LABELS)
LABELS = ["WAIT", "NOTE", "SUMMARY", "QUESTION", "VERIFY",
          "QUERY_TOOL", "WARN", "ALERT", "RESOLVE", "REVISE"]


class SmlrMultiLaneForCausalLM(LlamaForCausalLM):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.lane_names = list(getattr(config, "smlr_lanes", ["reasoning", "state_patch"]))
        self.lane_index = {ln: i for i, ln in enumerate(self.lane_names)}
        # per-lane decoder heads (ParallelLMHead -> same weight loader path as lm_head)
        self.lane_heads = nn.ModuleList([
            ParallelLMHead(config.vocab_size, config.hidden_size,
                           quant_config=quant_config, prefix=f"head_{ln}")
            for ln in self.lane_names
        ])
        # policy head (read once off the prompt-end hidden; bias included for production parity)
        self.policy_head = nn.Linear(config.hidden_size, len(LABELS), bias=True)
        logger.info(f"[smlr] multilane heads={self.lane_names} vocab={config.vocab_size}")

    # ---- last-token hidden pruning (mirrors LogitsProcessor.forward, greedy/no-logprob only) ----
    def _prune_last(self, hidden_states, logits_metadata):
        fm = logits_metadata.forward_mode
        if fm.is_decode_or_idle() or fm.is_target_verify():
            return hidden_states                       # one row per running seq already
        if fm.is_extend() and not logits_metadata.extend_return_logprob:
            if logits_metadata.padded_static_len < 0:
                last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            else:
                idx = torch.arange(len(logits_metadata.extend_seq_lens),
                                   device=logits_metadata.extend_seq_lens.device)
                last_index = (idx * logits_metadata.padded_static_len
                              + logits_metadata.extend_seq_lens - 1)
            return hidden_states[last_index]
        # extend WITH input logprobs: fall back to gathering each seq's final token only
        # (we don't request input logprobs in the VLA decode path, so this is a safety net)
        last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
        return hidden_states[last_index]

    def _lane_ids(self, forward_batch, n):
        """Per-row lane index (batch order, host list). Reads custom_params[i]['lane']; default 0."""
        si = getattr(forward_batch, "sampling_info", None)
        cps = getattr(si, "custom_params", None) if si is not None else None
        out = [0] * n
        if cps is not None:
            for i in range(min(n, len(cps))):
                cp = cps[i]
                if cp:
                    ln = cp.get("lane")
                    if isinstance(ln, str):
                        out[i] = self.lane_index.get(ln, 0)
                    elif isinstance(ln, int):
                        out[i] = ln
        return out

    def smlr_fill_lane_ids(self, forward_batch, buf):
        """Fill buf[:bs] (a pre-allocated DEVICE int64 tensor) with per-row lane ids from
        custom_params. Called by the CUDA graph runner in replay_prepare (OUTSIDE the graph), so the
        captured graph reads current lane routing from a stable buffer on every replay."""
        n = forward_batch.batch_size
        ids = self._lane_ids(forward_batch, n)
        buf[:n].copy_(torch.tensor(ids, dtype=torch.long, device=buf.device))

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch: ForwardBatch,
                input_embeds=None, get_embedding: bool = False,
                pp_proxy_tensors: Optional[PPProxyTensors] = None):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds,
                                   pp_proxy_tensors=pp_proxy_tensors)
        if not self.pp_group.is_last_rank:
            return hidden_states

        lm = LogitsMetadata.from_forward_batch(forward_batch)
        pruned = self._prune_last(hidden_states, lm)          # [N, H] last-token hidden per seq
        N = pruned.shape[0]

        # per-row lane id as a DEVICE tensor. Under CUDA graph the runner pre-fills
        # forward_batch.smlr_lane_ids (a stable buffer); otherwise (prefill / uncaptured decode)
        # build it from custom_params on the host.
        lane_buf = getattr(forward_batch, "smlr_lane_ids", None)
        if lane_buf is not None:
            lane_ids = lane_buf[:N]
        else:
            lane_ids = torch.tensor(self._lane_ids(forward_batch, N),
                                    dtype=torch.long, device=pruned.device)

        # GRAPH-SAFE head select: compute ALL lane heads for ALL rows (static shapes), then gather
        # each row's own lane. n_lanes x head matmul (fine for 2, acceptable for 6). No host sync,
        # no data-dependent .nonzero() -> capturable.
        V = self.config.vocab_size
        all_logits = torch.stack(
            [torch.matmul(pruned.to(h.weight.dtype), h.weight.T) for h in self.lane_heads], dim=0
        )                                                     # [L, N, V]
        logits = all_logits.gather(0, lane_ids.view(1, N, 1).expand(1, N, V)).squeeze(0)

        out = LogitsProcessorOutput(next_token_logits=logits)
        # POLICY read: when the request asks for hidden states (a 1-token policy probe), return the
        # FULL per-token hidden (capture mode is FULL); sglang slices it per request and the client
        # takes the LAST row = prompt-end hidden, then applies policy_head offline (exact 10-way
        # softmax + bias). Prefill only -> never inside the captured decode graph.
        if lm.capture_hidden_mode.need_capture():
            out.hidden_states = hidden_states
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        backbone = []
        for name, w in weights:
            if name.startswith("head_"):                      # head_<lane>.weight -> lane_heads[i]
                lane = name[len("head_"):].rsplit(".weight", 1)[0]
                if lane in self.lane_index:
                    p = self.lane_heads[self.lane_index[lane]].weight
                    default_weight_loader(p, w)
                continue
            if name.startswith("policy_head."):
                p = params.get(name)
                if p is not None:
                    default_weight_loader(p, w)
                continue
            backbone.append((name, w))                         # model.* + lm_head -> parent loader
        # parent load_weights handles qkv/gate_up fusion + skips tied lm_head
        super().load_weights(backbone)


EntryClass = SmlrMultiLaneForCausalLM
