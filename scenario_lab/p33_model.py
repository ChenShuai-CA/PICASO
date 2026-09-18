"""AR-Scene-v1: minimal autoregressive scene model per frozen spec p33.0-v1.1.

Architecture decisions the frozen spec leaves open, forced by the strict
hidden-information boundary (spec 4.5 per-query, per-time actor visibility:
query logits must be exactly invariant to changes of source-agent state at
frames where that source is invisible to the query):

* **Per-query view keys.**  Encoder agent-agent attention gives every query
  its own key set: the key for pair (query a, source s) pools s's per-frame
  encodings with weights ``pairwise_visibility[a, :, s] & state_valid[s, :]``,
  so a frame hidden from a contributes exactly zero weight (bit-exact
  invariance; masked mean divides by the visible-frame count).  Keys are
  functions of the source's raw history and the (a, s) visibility only and are
  never layer-updated, so agent ``b``'s evolving representation cannot relay
  hidden ``s`` to ``a``: the only path from ``s`` into ``a`` is the direct
  (a, s) key, which carries no hidden frame.  A pair with no visible frame is
  blocked at attention; the own key (diagonal) pools the agent's full valid
  history and is always attendable.
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

# Phase C (P3.3.4 SPEC §6.2) per-type construction bounds: agent_type id ->
# (jerk m/s^3, accel m/s^2).  They sit below the global kinematic limits
# (20/10) with the same engineering margin Phase B used (jerk 19 vs 20), so
# fp32 finite-difference noise (~0.03 m/s^3) can never cross the threshold.
KINEMATIC_C_LIMITS = {
    0: (19.0, 9.5),  # pad slots: vehicle envelope (masked out of every metric)
    1: (19.0, 9.5),  # vehicle
    2: (5.0, 3.0),   # pedestrian
    3: (10.0, 5.0),  # cyclist
    4: (10.0, 5.0),  # other
}
START_PAIR_CAP_MPS2 = 9.5  # ||a_1 + a_2|| cap -> implied start-jump accel < 10
SPEED_LIMIT_MPS = 35.0


def _norm_bounded(raw: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Smooth norm-bounded map ``limit*tanh(|u|) * u/|u|``; ``|out| <= limit``."""
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    return raw * (limit * torch.tanh(norm) / (norm + 1e-8))


def _project_disk(values: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Scale each 2-vector into the limit disk (no-op strictly inside it)."""
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values * torch.clamp(limit / (norm + 1e-9), max=1.0)


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
    """Pre-LN: agent-agent (per-query view keys) + agent-map + FFN."""

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

    def forward(self, x, pair_tokens, pair_blocked, map_tokens, map_pad):
        # x [B,A,d]; pair_tokens [B,q,s,d] per-query view keys; pair_blocked
        # [B,q,s] True=blocked (own key never blocked); map_tokens [B,map+1,d]
        # incl. NULL; map_pad [B,map+1] True=pad
        batch, agents, d = x.shape
        query = self.norm_query(x)
        keys = self.norm_static(pair_tokens)
        flat_query = query.reshape(batch * agents, 1, d)
        flat_keys = keys.reshape(batch * agents, agents, d)
        blocked = pair_blocked.reshape(batch * agents, agents)
        attended, _ = self.agent_attention(flat_query, flat_keys, flat_keys,
                                           key_padding_mask=blocked,
                                           need_weights=False)
        x = x + attended.reshape(batch, agents, d)
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
    def _pair_weights(self, batch: dict) -> torch.Tensor:
        """Frame weights W[b,q,t,s]: frame t of source s enters query q's view
        iff s is visible to q at t and the state is valid.  The diagonal (own
        history) uses the agent's full valid history regardless of visibility."""
        valid = batch["state_valid_mask"].transpose(1, 2)          # [B, t, s]
        visibility = batch["pairwise_visibility_mask"]              # [B, q, t, s]
        own = torch.eye(self.agents, dtype=torch.bool,
                        device=visibility.device)[None, :, None, :]
        return (visibility | own) & valid[:, None, :, :]            # [B, q, t, s]

    def _pair_view_tokens(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-query view keys [B,q,s,d] and attention blocks [B,q,s] (True=blocked).

        Spec 4.5: the history of source s is masked per (query, time) before any
        pooling or attention.  Hidden frames get exactly zero weight in both the
        mean and the last-visible-step gather, so perturbing them cannot change
        any downstream value; pairs with no visible frame are blocked entirely
        (the own diagonal key is never blocked).
        """
        history = (batch["agent_history"] / self.history_scale).clamp(-self.clip, self.clip)
        valid = batch["state_valid_mask"]
        steps = (self.history_encoder(history) + self.time_table[None, None]) * valid.unsqueeze(-1)
        # steps [B, s, t, d]: invalid frames are exactly zero
        weights = self._pair_weights(batch).to(steps.dtype)         # [B, q, t, s]
        mean = torch.einsum("bqts,bstd->bqsd", weights, steps)
        count = weights.sum(dim=2).clamp(min=1.0)                   # [B, q, s]
        mean = mean / count[..., None]
        positions = torch.arange(self.history_steps, device=steps.device)
        masked_pos = torch.where(weights > 0, positions[None, None, :, None], -1)
        last_visible = masked_pos.max(dim=2).values                 # [B, q, s]
        gather = last_visible.clamp(min=0)
        hot = (positions[None, None, :, None] == gather.unsqueeze(2)).to(steps.dtype)
        last = torch.einsum("bqts,bstd->bqsd", hot, steps)
        pooled = torch.cat([mean, last], dim=-1)
        token = self.agent_projector(pooled)
        token = (token + self.type_embedding(batch["agent_type"].long())[:, None, :]
                 + self.role_embedding(batch["agent_role"].long())[:, None, :]
                 + self.source_embedding(batch["source_id"])[:, None, None, :])
        present = batch["agent_present_mask"]
        has_visible = weights.sum(dim=2) > 0                        # [B, q, s]
        own_key = torch.eye(self.agents, dtype=torch.bool, device=present.device)
        blocked = ~(present[:, None, :] & has_visible) & ~own_key[None]
        return token, blocked

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

    def encode_context(self, batch: dict) -> dict[str, torch.Tensor]:
        pair_tokens, pair_blocked = self._pair_view_tokens(batch)
        map_tokens, map_pad = self._map_tokens(batch)
        index = torch.arange(self.agents, device=pair_tokens.device)
        x = pair_tokens[:, index, index]  # init from the own-history view token
        for layer in self.encoder:
            x = layer(x, pair_tokens, pair_blocked, map_tokens, map_pad)
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

    def _decoder_stack(self, context_agents: torch.Tensor, present: torch.Tensor,
                       visibility: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Shared decoder body: returns the normed per-(agent,chunk) hidden [B,A,K,d]."""
        inputs = self._token_inputs(teacher)
        x = inputs + context_agents[:, :, None]
        batch_size = x.shape[0]
        static_flat = inputs.reshape(batch_size, self.agents * self.chunks, -1)
        null = self.null_agent_key[None, None].expand(batch_size, 1, -1)
        static_flat = torch.cat([static_flat, null], dim=1)
        flat_mask = self._decoder_mask(present, visibility).repeat_interleave(self.heads, dim=0)
        for layer in self.decoder:
            x = layer(x, flat_mask, static_flat)
        return self.decoder_norm(x)

    def _decode_from_tokens(self, context_agents: torch.Tensor, present: torch.Tensor,
                            visibility: torch.Tensor,
                            teacher: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._decoder_stack(context_agents, present, visibility, teacher)
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


class ARSceneV1K(ARSceneV1):
    """Phase C: conditional kinematic decode head (P3.3.4 SPEC §6, frozen
    2026-09-15 before any Phase C training).

    Replaces the free per-chunk residual head -- the dominant kinematic
    violation source in the Phase A attribution -- with a state-carrying
    bounded-jerk integration conditioned on (selected-token hidden, history
    tail velocity, previous-chunk end state), per agent_type.  Semi-implicit
    recurrence, h = 1/sample_rate:

        j_t = J(type)*tanh(|u_t|)*u_t/|u_t|          (norm <= J by construction)
        a_{t+1} = disk(a_t + h*j_t, A(type))         (norm <= A by construction)
        v_{t+1} = v_t + h*a_{t+1};  P_{t+1} = P_t + h*v_{t+1}
        v_0 = history-tail velocity (exact start continuity),  P_0 = anchor
        start pair: v1 rescaled a_2 only (||a_1 + a_2|| NOT bounded -- the
        CODEX P1 defect); R1 ``start_pair_joint`` rescales the SUM jointly,
        which does yield ||a_1 + a_2|| <= min(2*A, START_PAIR_CAP) by
        construction (both terms scale by s <= 1, keeping disk membership).
        R1 computed a_2 from a_0, so the realized start jerk could reach 2J
        (R1_ERRATA 2.2); ``start_recurrence_version: 2`` (SPEC_R2 2.2)
        recurses a_2 from a_1_raw and caps the realized start jerk at J.

    Under the 50-output-step finite-difference metric convention (X_t = P_{t+1},
    anchor excluded) this yields implied accel = |a_{t+3}| <= A, implied jerk =
    |a_{t+4} - a_{t+3}|/h <= J (the disk projection is 1-Lipschitz from an
    in-disk point), and start-jump implied accel = |a_1 + a_2| <= 9.5 < 10.
    Speed is NOT bounded by construction (velocity integrates accel); the loss
    carries an exceedance penalty and evaluation reports it.  The head and the
    integration always run in fp32 outside any active autocast region: the
    construction margins (19/9.5 vs 20/10) must dominate fp32 third-difference
    noise, which bf16 would dwarf.
    """

    def __init__(self, config: dict, codebook: np.ndarray, c_config: dict | None = None):
        super().__init__(config, codebook)
        d = config["model"]["d_model"]
        del self.residual_head  # replaced by the kinematic head stack below
        self.construction = (c_config or {}).get("construction", "bounded")
        # R1 mechanisms (2026-09-16, SPEC_R1.md), all flag-gated: with the flags
        # absent the sealed v1 behavior is bit-exact, so v1 checkpoints and the
        # frozen R0 readings stay reproducible.
        self.token_conditioning = bool((c_config or {}).get("token_conditioning", False))
        self.start_pair_joint = bool((c_config or {}).get("start_pair_joint", False))
        # R2 fix (SPEC_R2.md 2.3, 2026-09-17): version 2 restores the continuous
        # start recurrence a_2 = disk(a_1_raw + h*j_2).  Absent/1 keeps the R1
        # joint path (a_2 from a_0, realized start jerk up to 2J = 38 m/s^3,
        # R1_ERRATA 2.2) so sealed v1/v2 checkpoints and R0/R1 readings replay
        # bit-exact.
        self.start_recurrence_version = int(
            (c_config or {}).get("start_recurrence_version", 1))
        self.v0_source = (c_config or {}).get("v0_source", "diff")
        self.v0_speed_limit = float((c_config or {}).get("v0_speed_limit_mps", 35.0))
        # R2-E output-precision revision arm ("C-v3-fp64out", user adjudication
        # 2026-09-19): re-integrate the SAME v_seq with a float64 anchor and
        # float64 cumsum.  The analytical recursion bounds hold exactly at the
        # type limits; only fp32 position storage/differencing amplified them
        # past limit + 1e-3 (26 accel / 2351 jerk candidates, FULL_DEV_REEVAL
        # type_checks -- fp64 reintegration of the same v_seq -> 0/0).  Default
        # "float32" keeps the sealed C-v3 output path bit-exact.
        self.output_position_precision = str(
            (c_config or {}).get("output_position_precision", "float32"))
        if self.output_position_precision not in ("float32", "float64"):
            raise ValueError("output_position_precision must be 'float32' or "
                             f"'float64', got {self.output_position_precision!r}")
        feature_dim = d + 6 + (2 * self.chunk_steps if self.token_conditioning else 0)
        self.state_mlp = nn.Sequential(nn.Linear(feature_dim, d), nn.GELU(), nn.Linear(d, d))
        self.jerk_head = nn.Linear(d, 2 * self.chunk_steps)
        self.init_head = nn.Linear(d, 2)
        self.h = 1.0 / float(config["data"]["sample_rate_hz"])
        limits = (c_config or {}).get("limits_by_type") or KINEMATIC_C_LIMITS
        vocabulary = config["data"]["agent_type_vocabulary"]
        size = max(vocabulary.values()) + 1
        jerk = torch.full((size,), KINEMATIC_C_LIMITS[0][0])
        accel = torch.full((size,), KINEMATIC_C_LIMITS[0][1])
        for type_id, (jerk_limit, accel_limit) in limits.items():
            jerk[int(type_id)] = float(jerk_limit)
            accel[int(type_id)] = float(accel_limit)
        self.register_buffer("jerk_limit", jerk)
        self.register_buffer("accel_limit", accel)

    # ----------------------------------------------------------------- history
    def _history_tail_velocity(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(v0 [B,A,2], usable [B,A]): finite difference of the last two history
        positions times the frame rate -- exactly the quantity the continuity
        metric compares against; zero where the tail is invalid or non-finite."""
        history = batch["agent_history"][:, :, -2:, :2].float()
        state_valid = batch["state_valid_mask"][:, :, -2:]
        velocity = (history[:, :, 1] - history[:, :, 0]) * (1.0 / self.h)
        usable = (state_valid[:, :, 0] & state_valid[:, :, 1]
                  & torch.isfinite(velocity).all(dim=-1))
        return torch.where(usable[..., None], velocity, torch.zeros_like(velocity)), usable

    def _history_tail_velocity_recorded(self, batch: dict) -> tuple[torch.Tensor,
                                                                    torch.Tensor,
                                                                    torch.Tensor]:
        """R1 v2 v0 policy (SPEC_R1 section 3): the RECORDED velocity channels
        of the last history frame -- SPEC section 6 as written -- disk-clamped
        to ``v0_speed_limit_mps`` with an explicit over-limit mask.  A corrupted
        position channel (R0's 613 m/s family) then cannot enter v0 through
        differencing, and a genuinely over-limit recording (the ~35 m/s real
        fast vehicles) keeps its direction with the excess counted, never
        silently integrated.  Returns (v0, usable, over_limit)."""
        recorded = batch["agent_history"][:, :, -1, 2:4].float()
        valid = batch["state_valid_mask"][:, :, -1]
        usable = valid & torch.isfinite(recorded).all(dim=-1)
        norm = torch.linalg.vector_norm(recorded, dim=-1, keepdim=True)
        over_limit = usable & (norm.squeeze(-1) > self.v0_speed_limit)
        limit = torch.full_like(norm, self.v0_speed_limit)
        clamped = _project_disk(recorded, limit)
        v0 = torch.where(usable[..., None], clamped, torch.zeros_like(clamped))
        return v0, usable, over_limit

    def _start_velocity(self, batch: dict) -> torch.Tensor:
        """v0 selector: recorded channels (R1 v2) or the sealed v1 position diff."""
        if self.v0_source == "recorded":
            return self._history_tail_velocity_recorded(batch)[0]
        return self._history_tail_velocity(batch)[0]

    # ------------------------------------------------------------------- head
    def _kinematic_pass(self, hidden: torch.Tensor, v0: torch.Tensor,
                        agent_types: torch.Tensor, carry: tuple | None = None,
                        upto: int | None = None,
                        tokens_current: torch.Tensor | None = None) -> dict:
        """Sequential head + integration -- the only place trajectories are born.

        ``hidden`` [B,A,K,d] decoder hidden (fp32 forced); ``carry`` =
        (a, v, next_chunk) resumes a rollout call; ``upto`` limits the chunk
        count.  Chunk k's head input is its own hidden (which encodes only
        tokens < k by decoder causality) plus the analytic state carried from
        earlier chunks, so the incremental rollout path computes exactly the
        same quantities as this parallel teacher-forced pass.

        R1 ``tokens_current`` [B,A,K] (flag ``token_conditioning``): the token
        selected FOR each chunk enters that chunk's head input as its codebook
        centroid, so flipping token k changes chunk k (and the last token has a
        live compute path).  The token is already decided when the head runs in
        both teacher forcing and rollout, and it is the agent's OWN token --
        cross-agent information still only enters through the decoder hidden,
        so the frozen visibility boundary is unchanged.  255/ignore maps to the
        zero BOS centroid.
        """
        bounded = self.construction == "bounded"
        jerk_limit = self.jerk_limit[agent_types][..., None]     # [B,A,1]
        accel_limit = self.accel_limit[agent_types][..., None]   # [B,A,1]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            x = hidden.float()
            batch, agents, chunks, _ = x.shape
            steps = self.chunk_steps
            if carry is None:
                a = (_norm_bounded(self.init_head(x[:, :, 0]), accel_limit) if bounded
                     else self.init_head(x[:, :, 0]))
                v, next_chunk, a0 = v0, 0, a
            else:
                a, v, next_chunk = carry
                a0 = None
            end = chunks if upto is None else next_chunk + upto
            v_list: list[torch.Tensor] = []
            a_list: list[torch.Tensor] = []
            jerks: list[torch.Tensor] = []
            a_first = None
            for position in range(next_chunk, end):
                local = position - next_chunk  # carry-resumed calls see a local slice
                # start pair = the trajectory's FIRST two steps (position 0).
                # v1 gated on `carry is None`, which in a parallel pass made the
                # rescale re-fire at every chunk boundary while rollout fired it
                # only at chunk 0 -- a latent teacher-forced/rollout
                # inconsistency (invisible while the rescale was a no-op).
                # Position gating is identical to v1 on the rollout path, which
                # every sealed v1 reading used.
                cap_pair = position == 0
                # R1 joint start-pair construction: rescale the SUM a_1 + a_2
                # (the exact implied start-jump accel), committing v_1 only
                # after the step-1 rescale.  Requires at least two steps in the
                # first chunk.  Applies to the penalty variant too (plan
                # section 2.4: branches agree except construction vs penalty).
                joint = cap_pair and self.start_pair_joint and steps >= 2
                features = [x[:, :, local], v / 10.0, a / 5.0, v0 / 10.0]
                if self.token_conditioning:
                    token = tokens_current[:, :, local]
                    token = torch.where(token == 255, BOS_TOKEN, token)
                    features.append(self.codebook_padded[token] / CENTROID_SCALE_M)
                feature = torch.cat(features, dim=-1)
                raw = self.jerk_head(self.state_mlp(feature)).reshape(batch, agents, steps, 2)
                jerk = _norm_bounded(raw, jerk_limit[..., None]) if bounded else raw
                jerks.append(jerk)
                for step in range(steps):
                    candidate = (_project_disk(a + self.h * jerk[:, :, step], accel_limit)
                                 if bounded else a + self.h * jerk[:, :, step])
                    if joint and step == 0:
                        a_first = candidate  # a_1 raw; v_1 commit deferred to step 1
                        if self.start_recurrence_version >= 2:
                            # R2 fix (SPEC_R2.md 2.2): advance the working state
                            # to a_1_raw so step 1 computes
                            # a_2 = disk(a_1_raw + h*j_2) -- the true recurrence.
                            # v_1 stays deferred; the step-1 joint rescale applies
                            # one scale s <= 1 to both terms, so the realized
                            # start jerk ||a_2 - a_1||/h = s*||a_2_raw - a_1_raw||/h
                            # <= ||j_2|| <= J (disk projection is non-expansive
                            # from the in-disk a_1_raw).  Version 1 left a = a_0
                            # here, capping the realized start jerk at 2J
                            # (38 m/s^3 for J = 19); kept for sealed replay.
                            a = candidate
                        continue
                    if joint and step == 1:
                        pair = torch.linalg.vector_norm(a_first + candidate,
                                                        dim=-1, keepdim=True)
                        ceiling = torch.minimum(2.0 * accel_limit,
                                                torch.full_like(accel_limit,
                                                                START_PAIR_CAP_MPS2))
                        scale = torch.clamp(ceiling / (pair + 1e-9), max=1.0)
                        a_first = a_first * scale
                        candidate = candidate * scale
                        v = v + self.h * a_first           # v_1 from the rescaled a_1
                        a = a_first
                        v_list.append(v)
                        a_list.append(a)
                        v = v + self.h * candidate          # v_2 from the rescaled a_2
                    else:
                        v = v + self.h * candidate
                        if cap_pair and step == 0:
                            a_first = candidate  # a_1
                        elif cap_pair and step == 1:
                            # sealed v1 behavior: rescale a_2 only (defective --
                            # ||a_1 + a_2|| is NOT bounded; kept for bit-exact
                            # v1 reproduction, counterexample-tested)
                            uncapped = candidate
                            pair = torch.linalg.vector_norm(a_first + uncapped,
                                                            dim=-1, keepdim=True)
                            ceiling = torch.minimum(2.0 * accel_limit,
                                                    torch.full_like(accel_limit,
                                                                    START_PAIR_CAP_MPS2))
                            scale = torch.clamp(ceiling / (pair + 1e-9), max=1.0)
                            candidate = uncapped * scale
                            v = v + self.h * (candidate - uncapped)
                    a = candidate
                    v_list.append(v)
                    a_list.append(a)
            return {"jerk": torch.stack(jerks, dim=2),
                    "v_seq": torch.stack(v_list, dim=2),   # v_1 .. v_{end*steps}
                    "a_seq": torch.stack(a_list, dim=2),   # a_1 .. a_{end*steps}
                    "a0": a0, "carry": (a, v, end),
                    "jerk_limit": jerk_limit, "accel_limit": accel_limit}

    def _positions_from_v(self, v_seq: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """P_{t+1} = P_t + h*v_{t+1}: positions are h * the cumsum of v_seq.

        ``output_position_precision`` == "float64" accumulates in float64
        (C-v3-fp64out arm): same v_seq, same anchor, no retraining -- removes
        the fp32 cumsum/storage quantization residual that pushed implied
        type-bound accel/jerk past limit + 1e-3 (R2-E numerical ledger)."""
        if self.output_position_precision == "float64":
            return (anchor[:, :, None].double()
                    + self.h * torch.cumsum(v_seq.double(), dim=-2))
        return anchor[:, :, None] + self.h * torch.cumsum(v_seq.float(), dim=-2)

    # ----------------------------------------------------------------- decode
    def _decode_from_tokens(self, context_agents: torch.Tensor, present: torch.Tensor,
                            visibility: torch.Tensor,
                            teacher: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._decoder_stack(context_agents, present, visibility, teacher)
        return {"motion_token_logits": self.token_head(x), "decoder_hidden": x}

    def forward(self, batch: dict, teacher_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encode_context(batch)
        outputs = self._decode_from_context(context["agent_context"], batch, teacher_tokens)
        v0 = self._start_velocity(batch)
        kin = self._kinematic_pass(outputs["decoder_hidden"], v0,
                                   batch["agent_type"].long(),
                                   tokens_current=(teacher_tokens
                                                   if self.token_conditioning else None))
        anchor = batch["agent_history"][:, :, -1, :2].float()
        kin["trajectories"] = self._positions_from_v(kin["v_seq"], anchor)
        outputs["kinematic"] = kin
        return outputs

    @torch.no_grad()
    def trajectory_from_tokens(self, tokens: torch.Tensor, batch: dict) -> torch.Tensor:
        """Decoder + kinematic head over given tokens (teacher/greedy guardrail)."""
        context = self.encode_context(batch)["agent_context"]
        x = self._decoder_stack(context, batch["agent_present_mask"],
                                batch["pairwise_visibility_mask"][:, :, -1, :], tokens)
        v0 = self._start_velocity(batch)
        kin = self._kinematic_pass(x, v0, batch["agent_type"].long(),
                                   tokens_current=(tokens
                                                   if self.token_conditioning else None))
        anchor = batch["agent_history"][:, :, -1, :2].float()
        return self._positions_from_v(kin["v_seq"], anchor)

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(self, batch: dict, num_samples: int = 6, temperature: float = 1.0,
                top_p: float = 0.95, seed: int = 0) -> dict[str, torch.Tensor]:
        """Sample num_samples trajectories chunk-by-chunk (deterministic in seed).

        Identical sampling skeleton to ARSceneV1.rollout; the kinematic state
        (v, a) carries across the chunk loop through ``carry``, so integration
        is exact across sampled chunk boundaries."""
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
        v0 = self._start_velocity(batch).repeat_interleave(num_samples, dim=0)
        types = batch["agent_type"].long().repeat_interleave(num_samples, dim=0)
        carry = None
        v_pieces: list[torch.Tensor] = []
        jerk_pieces: list[torch.Tensor] = []
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
            kin = self._kinematic_pass(outputs["decoder_hidden"][:, :, chunk:chunk + 1],
                                       v0, types, carry=carry, upto=1,
                                       tokens_current=(tokens[:, :, chunk:chunk + 1]
                                                       if self.token_conditioning else None))
            carry = kin["carry"]
            v_pieces.append(kin["v_seq"])
            jerk_pieces.append(kin["jerk"])
        v_seq = torch.cat(v_pieces, dim=2)
        jerk = torch.cat(jerk_pieces, dim=2)
        anchor = batch["agent_history"][:, :, -1, :2].repeat_interleave(num_samples, dim=0)
        trajectories = self._positions_from_v(v_seq, anchor).view(
            batch_size, num_samples, self.agents, self.future_steps, 2)
        return {"trajectories": trajectories,
                "tokens": tokens.view(batch_size, num_samples, self.agents, self.chunks),
                "token_log_prob": log_prob.view(batch_size, num_samples, self.agents,
                                                self.chunks),
                "jerk": jerk.view(batch_size, num_samples, self.agents, self.chunks,
                                  self.chunk_steps, 2),
                "v_seq": v_seq.view(batch_size, num_samples, self.agents,
                                    self.future_steps, 2)}


def ar_scene_loss_c(outputs: dict, batch: dict, codebook: torch.Tensor, config: dict,
                    c_config: dict | None = None) -> dict[str, torch.Tensor]:
    """Phase C loss (SPEC §6.3): the three frozen terms verbatim computed on the
    integrated trajectories, plus token-intent alignment (keeps the token ->
    displacement semantics alive so the head cannot bypass the tokens) and a
    speed-exceedance penalty (speed is the one quantity NOT bounded by the
    construction).  The penalty control variant additionally trains explicit
    jerk/accel exceedance terms instead of relying on the construction."""
    weights = config["training"]["loss"]
    extras = (c_config or {}).get("loss", {})
    intent_weight = float(extras.get("token_intent_huber", 0.1))
    speed_weight = float(extras.get("speed_exceedance", 0.05))
    logits = outputs["motion_token_logits"]
    kin = outputs["kinematic"]
    decoded = kin["trajectories"]
    batch_size, agents, chunks, _ = logits.shape
    steps = config["motion_tokens"]["chunk_steps"]
    h = 1.0 / float(config["data"]["sample_rate_hz"])

    targets = batch["motion_token_target"]
    token_valid = batch["motion_token_valid_mask"] & (targets != 255)
    ignore = config["motion_tokens"]["ignore_index"]
    ce_targets = torch.where(token_valid, targets, torch.full_like(targets, ignore))
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ce_targets.reshape(-1),
                         ignore_index=ignore, reduction="sum") / token_valid.sum().clamp(min=1)
    with torch.no_grad():
        accuracy = ((logits.argmax(-1) == targets) & token_valid).sum() / token_valid.sum().clamp(min=1)

    chain = chain_validity(batch["state_valid_mask"], batch["future_valid_mask"])
    step_distance = torch.norm(decoded - batch["future_xy"], dim=-1)  # [B,A,50]
    step_huber = F.huber_loss(step_distance, torch.zeros_like(step_distance),
                              delta=HUBER_DELTA_M, reduction="none")
    traj = (step_huber * chain).sum() / chain.sum().clamp(min=1)

    main_roles = torch.isin(batch["agent_role"], torch.tensor([1, 2], device=targets.device))
    endpoint_ok = chain[:, :, -1] & batch["agent_present_mask"] & main_roles
    endpoint_huber = F.huber_loss(step_distance[:, :, -1],
                                  torch.zeros_like(step_distance[:, :, -1]),
                                  delta=HUBER_DELTA_M, reduction="none")
    endpoint = ((endpoint_huber * endpoint_ok).sum() / endpoint_ok.sum()
                if endpoint_ok.any() else step_distance.new_zeros(()))

    # token intent: integrated per-chunk displacement vs the selected centroid's
    safe_targets = targets.clamp(max=codebook.shape[0] - 1)
    centroid = codebook[safe_targets].reshape(batch_size, agents, chunks, steps, 2).sum(dim=3)
    model_disp = h * kin["v_seq"].reshape(batch_size, agents, chunks, steps, 2).sum(dim=3)
    chunk_chain = chain[:, :, steps - 1::steps]                 # chunk-end validity
    intent_mask = token_valid & chunk_chain
    intent_distance = torch.norm(model_disp - centroid, dim=-1)  # [B,A,K]
    intent_huber = F.huber_loss(intent_distance, torch.zeros_like(intent_distance),
                                delta=HUBER_DELTA_M, reduction="none")
    intent = (intent_huber * intent_mask).sum() / intent_mask.sum().clamp(min=1)

    # speed exceedance on the implied velocities (v_2..v_50 = v_seq[:, :, 1:])
    speed = torch.norm(kin["v_seq"][:, :, 1:], dim=-1)           # [B,A,49]
    speed_valid = chain[:, :, :-1] & chain[:, :, 1:]
    over = torch.relu(speed - SPEED_LIMIT_MPS)
    speed_penalty = ((over ** 2) * speed_valid).sum() / speed_valid.sum().clamp(min=1)

    total = (weights["motion_token_cross_entropy"] * ce
             + weights["decoded_trajectory_huber"] * traj
             + weights["endpoint_huber"] * endpoint
             + intent_weight * intent + speed_weight * speed_penalty)
    result = {"total": total, "token_cross_entropy": ce, "trajectory_huber": traj,
              "endpoint_huber": endpoint, "token_accuracy": accuracy.float(),
              "intent_huber": intent, "speed_exceedance": speed_penalty}
    if (c_config or {}).get("construction", "bounded") != "bounded":
        # plan section 2.4: exceedance means weighted by valid agents and the
        # chain-validity mask (no pad pollution), matching the bounded branch's
        # convention of only counting real steps
        step_mask = (chain.reshape(batch_size, agents, chunks, steps)
                     & batch["agent_present_mask"][:, :, None, None])
        jerk_over = torch.relu(torch.norm(kin["jerk"], dim=-1) - kin["jerk_limit"][..., None])
        jerk_penalty = ((jerk_over ** 2) * step_mask).sum() / step_mask.sum().clamp(min=1)
        a_seq = kin["a_seq"].reshape(batch_size, agents, chunks, steps, 2)
        accel_over = torch.relu(torch.norm(a_seq, dim=-1) - kin["accel_limit"][..., None])
        accel_penalty = ((accel_over ** 2) * step_mask).sum() / step_mask.sum().clamp(min=1)
        total = total + (float(extras.get("jerk_exceedance", 0.5)) * jerk_penalty
                         + float(extras.get("accel_exceedance", 0.5)) * accel_penalty)
        result["total"] = total
        result["jerk_exceedance"] = jerk_penalty
        result["accel_exceedance"] = accel_penalty
    return result
