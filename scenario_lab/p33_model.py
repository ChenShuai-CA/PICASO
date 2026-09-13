"""AR-Scene-v1: minimal autoregressive scene model per frozen spec p33.0-v1.1.

Architecture decisions the frozen spec leaves open, forced by the strict
hidden-information boundary (query logits must be exactly invariant to changes
in invisible source-agent history):

* **Static-key isolation.**  Agent-agent attention uses keys/values that are
  functions of the *source agent's own raw history only* (encoder) or of the
  *token identity only* (decoder).  Layer-updated states are never shared as
  keys: agent ``b``'s evolving representation could otherwise absorb hidden
  agent ``s`` and relay it to ``a`` even though ``s`` is invisible to ``a``.
  With static keys the only path from ``s`` into query ``a`` is a direct
  attention edge, which the per-row visibility mask removes.
* **NULL keys.**  Attention that can be fully masked for a row (agent-map,
  decoder inter-agent) carries one extra always-attendable learned NULL key so
  softmax never sees an all -inf row (map-less scenes, agents with no visible
  neighbours).
* **Decode anchor.**  ``scene-shard-v1`` motion vectors are consecutive
  differences of ``concat(history[-1].xy, future_xy)`` (see
  ``p33_pipeline.batched_motion_vectors``).  Decoding cumulatively sums
  ``centroid + residual`` from the raw ``t = 0`` history position, and
  trajectory-loss validity is cumulative chain validity (current-valid AND all
  intermediate future steps valid), matching token chunk validity.

Loss follows ``training.loss`` weights from the frozen config; the Huber delta
(1.0 m) is an engineering choice recorded in the model smoke manifest.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HUBER_DELTA_M = 1.0
CENTROID_SCALE_M = 5.0  # centroid/residual input normalization scale (meters)
BOS_TOKEN = 128  # virtual "begin of chunk sequence" id in the padded codebook


def _sinusoidal(length: int, dimension: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / dimension))
    table = torch.zeros(length, dimension)
    table[:, 0::2] = torch.sin(position * divisor)
    table[:, 1::2] = torch.cos(position * divisor)
    return table


def to_torch_batch(batch: dict, device: str | torch.device = "cpu") -> dict:
    """Convert a collated numpy batch into torch tensors (lists stay lists)."""
    converted = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            dtype = torch.int64 if key == "motion_token_target" else None
            converted[key] = torch.from_numpy(value.copy()).to(device=device, dtype=dtype)
        else:
            converted[key] = value
    return converted


class MaskedPool(nn.Module):
    """Mean-pool point encodings under a validity mask; invalid steps contribute zero."""

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # features [B,N,P,d], valid [B,N,P] -> [B,N,d]
        weight = valid.to(features.dtype).unsqueeze(-1)
        summed = (features * weight).sum(dim=2)
        count = weight.sum(dim=2).clamp(min=1.0)
        return summed / count


class EncoderLayer(nn.Module):
    """Pre-LN: agent-agent (static own-history keys) + agent-map + FFN."""

    def __init__(self, config: dict):
        super().__init__()
        d = config["model"]["d_model"]
        heads = config["model"]["num_attention_heads"]
        dropout = config["model"]["dropout"]
        self.agent_attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.map_attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, config["model"]["ffn_dimension"]), nn.GELU(),
                                 nn.Linear(config["model"]["ffn_dimension"], d), nn.Dropout(dropout))
        self.norm_query = nn.LayerNorm(d)
        self.norm_static = nn.LayerNorm(d)
        self.norm_map = nn.LayerNorm(d)
        self.norm_ffn = nn.LayerNorm(d)

    def forward(self, x, static_agents, agent_mask, map_tokens, map_pad):
        # x [B,A,d]; static_agents [B,A,d]; agent_mask [B*H,A,A] bool True=blocked;
        # map_tokens [B,map+1,d] incl. NULL; map_pad [B,map+1] True=pad
        query = self.norm_query(x)
        static = self.norm_static(static_agents)
        attended, _ = self.agent_attention(query, static, static, attn_mask=agent_mask,
                                           need_weights=False)
        x = x + attended
        attended, _ = self.map_attention(self.norm_map(x), map_tokens, map_tokens,
                                         key_padding_mask=map_pad, need_weights=False)
        x = x + attended
        x = x + self.ffn(self.norm_ffn(x))
        return x


class DecoderLayer(nn.Module):
    """Pre-LN: per-agent causal chunk self-attn + inter-agent static-token cross-attn + FFN."""

    def __init__(self, config: dict):
        super().__init__()
        d = config["model"]["d_model"]
        heads = config["model"]["num_attention_heads"]
        dropout = config["model"]["dropout"]
        self.self_attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, config["model"]["ffn_dimension"]), nn.GELU(),
                                 nn.Linear(config["model"]["ffn_dimension"], d), nn.Dropout(dropout))
        self.norm_self = nn.LayerNorm(d)
        self.norm_cross_q = nn.LayerNorm(d)
        self.norm_cross_kv = nn.LayerNorm(d)
        self.norm_ffn = nn.LayerNorm(d)

    def forward(self, x, flat_mask, static_flat):
        # x [B,A,K,d]; flat_mask [B*H,A*K,A*K+1] True=blocked; static_flat [B,A*K+1,d]
        batch, agents, chunks, d = x.shape
        per_agent = x.reshape(batch * agents, chunks, d)
        normed = self.norm_self(per_agent)
        attended, _ = self.self_attention(normed, normed, normed,
                                          attn_mask=self.causal_mask(chunks, x.device),
                                          need_weights=False)
        x = x + attended.reshape(batch, agents, chunks, d)
        flat = x.reshape(batch, agents * chunks, d)
        normed_flat = self.norm_cross_q(flat)
        static = self.norm_cross_kv(static_flat)
        attended, _ = self.cross_attention(normed_flat, static, static, attn_mask=flat_mask,
                                           need_weights=False)
        x = x + attended.reshape(batch, agents, chunks, d)
        x = x + self.ffn(self.norm_ffn(x))
        return x

    @staticmethod
    def causal_mask(chunks: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(chunks, chunks, dtype=torch.bool, device=device),
                          diagonal=1)


class ARSceneV1(nn.Module):
    """Minimal AR-Scene-v1: joint 16-agent scene, 10 autoregressive time chunks."""

    def __init__(self, config: dict, codebook: np.ndarray):
        super().__init__()
        self.config = config
        model = config["model"]
        data = config["data"]
        d = model["d_model"]
        self.agents = data["max_agents"]
        self.history_steps = data["history_steps"]
        self.future_steps = data["future_steps"]
        self.chunk_steps = config["motion_tokens"]["chunk_steps"]
        self.chunks = self.future_steps // self.chunk_steps
        self.vocabulary = config["motion_tokens"]["vocabulary_size"]
        self.num_layers = model["num_layers"]
        self.heads = model["num_attention_heads"]

        codebook = torch.from_numpy(np.asarray(codebook, dtype=np.float32))
        self.register_buffer("codebook", codebook)  # [V,10] meters
        padded = torch.cat([codebook, torch.zeros(1, 10)])  # BOS row = zero centroid
        self.register_buffer("codebook_padded", padded)  # [V+1,10]
        self.register_buffer("time_table", _sinusoidal(self.history_steps, d))

        norm = data["normalization"]
        clip = norm["clip_absolute_value"]
        self.clip = float(clip)
        self.register_buffer("history_scale", torch.tensor(
            [norm["position_m"], norm["position_m"], norm["velocity_mps"], norm["velocity_mps"],
             1.0, 1.0, norm["length_m"], norm["width_m"]], dtype=torch.float32))
        self.register_buffer("map_scale", torch.tensor(
            [norm["position_m"], norm["position_m"], 1.0, 1.0,
             norm["curvature_inv_m"], norm["speed_limit_mps"]], dtype=torch.float32))

        self.centroid_projection = nn.Sequential(nn.Linear(10, d), nn.GELU(), nn.Linear(d, d))
        self.source_embedding = nn.Embedding(2, d)
        self.type_embedding = nn.Embedding(len(data["agent_type_vocabulary"]), d)
        self.role_embedding = nn.Embedding(len(data["agent_role_vocabulary"]), d)
        self.map_type_embedding = nn.Embedding(len(data["map_type_vocabulary"]), d)
        self.token_embedding = nn.Embedding(self.vocabulary + 1, d)  # +1 BOS
        self.chunk_embedding = nn.Embedding(self.chunks, d)
        self.null_agent_key = nn.Parameter(torch.zeros(d))
        self.null_map_key = nn.Parameter(torch.zeros(d))

        self.history_encoder = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.agent_pool = MaskedPool()
        self.agent_projector = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.map_encoder = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.map_pool = MaskedPool()
        self.encoder = nn.ModuleList(EncoderLayer(config) for _ in range(self.num_layers))
        self.encoder_norm = nn.LayerNorm(d)
        self.decoder = nn.ModuleList(DecoderLayer(config) for _ in range(self.num_layers))
        self.decoder_norm = nn.LayerNorm(d)
        self.token_head = nn.Linear(d, self.vocabulary)
        self.residual_head = nn.Linear(d, 10)

    # ----------------------------------------------------------------- inputs
    def _agent_static_tokens(self, batch: dict) -> torch.Tensor:
        """Own-history-only agent tokens [B,A,d]; never mixed across agents."""
        history = (batch["agent_history"] / self.history_scale).clamp(-self.clip, self.clip)
        valid = batch["state_valid_mask"].unsqueeze(-1).to(history.dtype)
        steps = (self.history_encoder(history) + self.time_table[None, None]) * valid
        mean = self.agent_pool(steps, batch["state_valid_mask"])
        pooled = torch.cat([mean, steps[:, :, -1]], dim=-1)
        token = self.agent_projector(pooled)
        return (token + self.type_embedding(batch["agent_type"].long())
                + self.role_embedding(batch["agent_role"].long())
                + self.source_embedding(batch["source_id"])[:, None])

    def _map_tokens(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Map tokens + key padding [B,map+1,d]/[B,map+1]; trailing NULL key."""
        points = (batch["map_polylines"] / self.map_scale).clamp(-self.clip, self.clip)
        pooled = self.map_pool(self.map_encoder(points), batch["map_point_mask"])
        tokens = pooled + self.map_type_embedding(batch["map_type"].long())
        batch_size = tokens.shape[0]
        null = self.null_map_key[None, None].expand(batch_size, 1, -1)
        tokens = torch.cat([tokens, null], dim=1)
        line_valid = batch["map_point_mask"].any(dim=2)
        pad = torch.cat([~line_valid, torch.zeros(batch_size, 1, dtype=torch.bool,
                                                  device=tokens.device)], dim=1)
        return tokens, pad

    def _encoder_mask(self, batch: dict) -> torch.Tensor:
        """[B,A,A] bool True=blocked: attend self + t=0-visible present sources."""
        visibility = batch["pairwise_visibility_mask"][:, :, -1, :]
        present = batch["agent_present_mask"]
        allowed = visibility & present[:, None, :] | torch.eye(
            self.agents, dtype=torch.bool, device=present.device)[None]
        return ~allowed

    def encode_context(self, batch: dict) -> dict[str, torch.Tensor]:
        static = self._agent_static_tokens(batch)
        map_tokens, map_pad = self._map_tokens(batch)
        attn_mask = self._encoder_mask(batch).repeat_interleave(self.heads, dim=0)
        x = static
        for layer in self.encoder:
            x = layer(x, static, attn_mask, map_tokens, map_pad)
        return {"agent_context": self.encoder_norm(x)}

    def _token_inputs(self, teacher: torch.Tensor) -> torch.Tensor:
        """Static per-(agent,chunk) identity embeddings [B,A,K,d] from teacher tokens.

        Chunk k encodes token k-1 (BOS at k=0); 255/ignore maps to BOS. Carries
        no agent history and no learned scene context, so it is safe to share
        as cross-attention keys under the strict hidden-information boundary.
        """
        batch_size = teacher.shape[0]
        bos = torch.full((batch_size, self.agents, 1), BOS_TOKEN, dtype=torch.long,
                         device=teacher.device)
        prefix = torch.cat([bos, teacher[:, :, :-1]], dim=2)
        prefix = torch.where(prefix == 255, bos.expand_as(prefix), prefix)
        centroid = self.codebook_padded[prefix] / CENTROID_SCALE_M
        return (self.token_embedding(prefix) + self.centroid_projection(centroid)
                + self.chunk_embedding.weight[None, None])

    def _decoder_mask(self, present: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        """[B,A*K,A*K+1] bool True=blocked, for cross-attn onto static token keys.

        Query (a,k) attends key (b,j) iff b present, b != a, visible(a,b) at
        t=0, j >= 1 (real token, not BOS) and j <= k (strictly past chunks).
        The trailing NULL key (index A*K) is always allowed.
        """
        device = present.device
        agents, chunks = self.agents, self.chunks
        arange = torch.arange(agents, device=device)
        same_agent = arange[:, None] == arange[None, :]  # [qA,kA]
        pair = visibility & present[:, None, :] & ~same_agent[None]  # [B,qA,kA]
        q_chunk = torch.arange(chunks, device=device)[:, None]  # [K,1]
        k_chunk = torch.arange(chunks, device=device)[None, :]  # [1,K]
        chunk_ok = (k_chunk >= 1) & (k_chunk <= q_chunk)  # [K(q),K(k)]
        # allowed[b, qA, Kq, kA, Kk] = pair[b,qA,kA] & chunk_ok[qK,kK]
        allowed = pair[:, :, None, :, None] & chunk_ok[:, None][None, None]
        allowed = allowed.reshape(present.shape[0], agents * chunks, agents * chunks)
        null = torch.ones(present.shape[0], agents * chunks, 1, dtype=torch.bool, device=device)
        return ~torch.cat([allowed, null], dim=2)

    def _decode_from_tokens(self, context_agents: torch.Tensor, present: torch.Tensor,
                            visibility: torch.Tensor,
                            teacher: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = self._token_inputs(teacher)
        x = inputs + context_agents[:, :, None]
        batch_size = x.shape[0]
        static_flat = inputs.reshape(batch_size, self.agents * self.chunks, -1)
        null = self.null_agent_key[None, None].expand(batch_size, 1, -1)
        static_flat = torch.cat([static_flat, null], dim=1)
        flat_mask = self._decoder_mask(present, visibility).repeat_interleave(self.heads, dim=0)
        for layer in self.decoder:
            x = layer(x, flat_mask, static_flat)
        x = self.decoder_norm(x)
        return {"motion_token_logits": self.token_head(x),
                "delta_xy_residual": self.residual_head(x)}

    def forward(self, batch: dict, teacher_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encode_context(batch)
        return self._decode_from_context(context["agent_context"], batch, teacher_tokens)

    def _decode_from_context(self, context_agents: torch.Tensor, batch: dict,
                             teacher: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._decode_from_tokens(context_agents, batch["agent_present_mask"],
                                        batch["pairwise_visibility_mask"][:, :, -1, :], teacher)

    def _decode_torch(self, tokens: torch.Tensor, residuals: torch.Tensor,
                      current: torch.Tensor) -> torch.Tensor:
        centroid = self.codebook_padded[tokens]
        deltas = (centroid + residuals).reshape(*tokens.shape[:-1], self.future_steps, 2)
        return current[:, :, None] + torch.cumsum(deltas, dim=-2)

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(self, batch: dict, num_samples: int = 6, temperature: float = 1.0,
                top_p: float = 0.95, seed: int = 0) -> dict[str, torch.Tensor]:
        """Sample num_samples trajectories per scene chunk-by-chunk (deterministic in seed)."""
        self.eval()
        device = batch["agent_present_mask"].device
        batch_size = batch["agent_present_mask"].shape[0]
        generator = torch.Generator().manual_seed(seed)
        context = self.encode_context(batch)["agent_context"]
        expanded_context = context.repeat_interleave(num_samples, dim=0)
        present = batch["agent_present_mask"].repeat_interleave(num_samples, dim=0)
        visibility = batch["pairwise_visibility_mask"][:, :, -1, :].repeat_interleave(
            num_samples, dim=0)
        tokens = torch.full((batch_size * num_samples, self.agents, self.chunks), 255,
                            dtype=torch.long, device=device)
        log_prob = torch.zeros(tokens.shape, device=device)
        residuals = None
        for chunk in range(self.chunks):
            outputs = self._decode_from_tokens(expanded_context, present, visibility, tokens)
            logits = outputs["motion_token_logits"][:, :, chunk] / max(temperature, 1e-6)
            log_probs = F.log_softmax(logits.double(), dim=-1)
            sorted_probs, sorted_index = log_probs.exp().sort(dim=-1, descending=True)
            keep = (sorted_probs.cumsum(dim=-1) - sorted_probs) < top_p  # shifted cumsum < p
            keep[..., 0] = True
            filtered = torch.zeros_like(sorted_probs).scatter_(-1, sorted_index,
                                                               torch.where(keep, sorted_probs,
                                                                          torch.zeros_like(sorted_probs)))
            # multinomial with a CPU generator must run on CPU tensors
            sampled = torch.multinomial(filtered.detach().cpu().reshape(-1, filtered.shape[-1]),
                                        1, generator=generator).reshape(tokens.shape[0],
                                                                        self.agents).to(device)
            tokens[:, :, chunk] = sampled
            log_prob[:, :, chunk] = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).float()
            residuals = outputs["delta_xy_residual"]
        current = batch["agent_history"][:, :, -1, :2].repeat_interleave(num_samples, dim=0)
        trajectories = self._decode_torch(tokens, residuals, current)
        return {"trajectories": trajectories.view(batch_size, num_samples, self.agents,
                                                  self.future_steps, 2),
                "tokens": tokens.view(batch_size, num_samples, self.agents, self.chunks),
                "token_log_prob": log_prob.view(batch_size, num_samples, self.agents, self.chunks),
                "residuals": residuals.view(batch_size, num_samples, self.agents, self.chunks, 10)}


def decode_tokens_to_trajectory(tokens: np.ndarray, residuals: np.ndarray,
                                codebook: np.ndarray, current_xy: np.ndarray) -> np.ndarray:
    """[...A,K] tokens + [...,A,K,10] residuals -> [...,A,50,2] metric positions."""
    token_array = np.asarray(tokens)
    safe = np.clip(token_array, 0, len(codebook) - 1)
    deltas = (codebook[safe].astype(np.float64)
              + np.asarray(residuals, dtype=np.float64)).reshape(*token_array.shape[:-1], -1, 2)
    return np.asarray(current_xy, dtype=np.float64)[..., None, :] + np.cumsum(deltas, axis=-2)


def chain_validity(state_valid: torch.Tensor, future_valid: torch.Tensor) -> torch.Tensor:
    """Cumulative anchor validity [B,A,50]: a decoded step counts only if every
    diff up to it (including current->first) is valid, matching token validity."""
    start = state_valid[:, :, -1].to(torch.float32)
    return (future_valid.to(torch.float32).cumprod(dim=-1) * start[:, :, None]) > 0


def ar_scene_loss(outputs: dict, batch: dict, codebook: torch.Tensor,
                  config: dict) -> dict[str, torch.Tensor]:
    """1.0*CE + 0.5*trajectory Huber + 0.2*endpoint Huber, each mean-normalized first."""
    weights = config["training"]["loss"]
    logits = outputs["motion_token_logits"]
    residuals = outputs["delta_xy_residual"]
    batch_size, agents, chunks, _ = logits.shape
    future_steps = chunks * config["motion_tokens"]["chunk_steps"]
    targets = batch["motion_token_target"]
    token_valid = batch["motion_token_valid_mask"] & (targets != 255)
    ignore = config["motion_tokens"]["ignore_index"]
    # ignore-by-validity must match the normalizer exactly, so validity is folded
    # into the targets instead of relying on 255 alone
    ce_targets = torch.where(token_valid, targets, torch.full_like(targets, ignore))
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ce_targets.reshape(-1),
                         ignore_index=ignore, reduction="sum") / token_valid.sum().clamp(min=1)
    with torch.no_grad():
        accuracy = ((logits.argmax(-1) == targets) & token_valid).sum() / token_valid.sum().clamp(min=1)

    safe_targets = targets.clamp(max=codebook.shape[0] - 1)
    deltas = (codebook[safe_targets] + residuals).reshape(batch_size, agents, future_steps, 2)
    current = batch["agent_history"][:, :, -1, :2]
    decoded = current[:, :, None] + torch.cumsum(deltas, dim=-2)
    chain = chain_validity(batch["state_valid_mask"], batch["future_valid_mask"])
    step_distance = torch.norm(decoded - batch["future_xy"], dim=-1)  # [B,A,50]
    step_huber = F.huber_loss(step_distance, torch.zeros_like(step_distance),
                              delta=HUBER_DELTA_M, reduction="none")
    traj = (step_huber * chain).sum() / chain.sum().clamp(min=1)

    main_roles = torch.isin(batch["agent_role"], torch.tensor([1, 2], device=targets.device))
    endpoint_ok = chain[:, :, -1] & batch["agent_present_mask"] & main_roles
    endpoint_huber = F.huber_loss(step_distance[:, :, -1], torch.zeros_like(step_distance[:, :, -1]),
                                  delta=HUBER_DELTA_M, reduction="none")
    endpoint = ((endpoint_huber * endpoint_ok).sum() / endpoint_ok.sum()
                if endpoint_ok.any() else step_distance.new_zeros(()))

    total = (weights["motion_token_cross_entropy"] * ce
             + weights["decoded_trajectory_huber"] * traj
             + weights["endpoint_huber"] * endpoint)
    return {"total": total, "token_cross_entropy": ce, "trajectory_huber": traj,
            "endpoint_huber": endpoint, "token_accuracy": accuracy.float()}
