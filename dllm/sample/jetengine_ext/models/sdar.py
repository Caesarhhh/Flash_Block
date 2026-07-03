import torch
from torch import nn
import torch.distributed as dist
import time
import os
from jetengine_ext.utils.context import set_context, get_context, reset_context
from jetengine_ext.layers.activation import SiluAndMul
from jetengine_ext.layers.attention import BlockAttention
from jetengine_ext.layers.layernorm import RMSNorm
from jetengine_ext.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from jetengine_ext.layers.rotary_embedding import get_rope
from jetengine_ext.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


def _detail_profile_enabled() -> bool:
    return os.environ.get("TRADO_DETAIL_PROFILE", "0") == "1"


def _profile_now(x: torch.Tensor | None = None) -> float:
    if x is not None and x.is_cuda:
        torch.cuda.synchronize(x.device)
    return time.perf_counter()


def _add_profile_s(name: str, value: float) -> None:
    context = get_context()
    context.profile_detail_s[name] = context.profile_detail_s.get(name, 0.0) + value


class SDARAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = BlockAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        sparsity_params=None
    ) -> torch.Tensor:
        detail_profile = _detail_profile_enabled() and hidden_states.is_cuda
        if detail_profile:
            st = _profile_now(hidden_states)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        if detail_profile:
            ed = _profile_now(hidden_states)
            _add_profile_s("detail_qkv_norm_rope_s", ed - st)
            st = ed
        o = self.attn(q, k, v, sparsity_params=sparsity_params)
        if detail_profile:
            ed = _profile_now(o)
            _add_profile_s("detail_block_attn_call_s", ed - st)
            st = ed
        output = self.o_proj(o)
        if detail_profile:
            ed = _profile_now(output)
            _add_profile_s("detail_o_proj_s", ed - st)
        return output


class SDARMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class SDARDecoderLayer(nn.Module):

    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        self.self_attn = SDARAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = SDARMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        sparsity_params=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detail_profile = _detail_profile_enabled() and hidden_states.is_cuda
        if detail_profile:
            layer_start = _profile_now(hidden_states)
            st = layer_start
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if detail_profile:
            ed = _profile_now(hidden_states)
            _add_profile_s("detail_input_norm_s", ed - st)
            st = ed
        
        hidden_states = self.self_attn(positions, hidden_states, sparsity_params=sparsity_params)
        if detail_profile:
            ed = _profile_now(hidden_states)
            _add_profile_s("detail_self_attn_total_s", ed - st)
            st = ed
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if detail_profile:
            ed = _profile_now(hidden_states)
            _add_profile_s("detail_post_attn_norm_s", ed - st)
            st = ed
        hidden_states = self.mlp(hidden_states)
        if detail_profile:
            ed = _profile_now(hidden_states)
            _add_profile_s("detail_mlp_s", ed - st)
            _add_profile_s("detail_decoder_layer_total_s", ed - layer_start)
        return hidden_states, residual


class SDARModel(nn.Module):

    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SDARDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        sparsity_params=None
    ) -> torch.Tensor:
        detail_profile = _detail_profile_enabled() and input_ids.is_cuda
        if detail_profile:
            st = _profile_now(input_ids)
        hidden_states = self.embed_tokens(input_ids)
        if detail_profile:
            ed = _profile_now(hidden_states)
            _add_profile_s("detail_embed_s", ed - st)
            st = ed
        residual = None
        for layer in self.layers:
            #st=time.time()
            hidden_states, residual = layer(positions, hidden_states, residual, sparsity_params)
            #print(f"layer {time.time()-st}s")
        # print(f"whole {time.time()-st}s")
        if detail_profile:
            st = _profile_now(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        if detail_profile:
            ed = _profile_now(hidden_states)
            _add_profile_s("detail_final_norm_s", ed - st)
        return hidden_states


class SDARForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        self.model = SDARModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        sparsity_params=None
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions,sparsity_params=sparsity_params)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        detail_profile = _detail_profile_enabled() and hidden_states.is_cuda
        if detail_profile:
            st = _profile_now(hidden_states)
        logits = self.lm_head(hidden_states)
        if detail_profile:
            ed = _profile_now(logits)
            _add_profile_s("detail_lm_head_s", ed - st)
        return logits
