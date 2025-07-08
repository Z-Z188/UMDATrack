from collections import defaultdict
import logging


import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.utils._pytree import tree_map

logger = logging.getLogger(__name__)


class TTTCache:
    def __init__(self, model, batch_size, mini_batch_size, num_ttt_layers):
        self.seqlen_offset = 0
        self.mini_batch_size = mini_batch_size
        self.ttt_params_dict = defaultdict(dict)
        self.ttt_param_names = ["W1", "b1"]

        self.conv_states_dic = defaultdict(dict)
        logger.info(f"Creating cache of size: {batch_size}")

        for layer_idx in range(num_ttt_layers):

            # 获取当前层的 TTTLinear 对象
            layer = model[layer_idx]

            for name in self.ttt_param_names:

                weight = getattr(layer, name)
                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim())
                # tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(device)
                #batch_size,) + (1,) * weight.dim() 会生成一个形状元组，例如 (batch_size, 1, 1, 1)，表示第一个维度重复 batch_size 次，其他维度不重复。

                self.ttt_params_dict[f"{name}_states"][layer_idx] = tiled_weight
                # for decoding, we need to store the gradients as well
                self.ttt_params_dict[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)

    def update(self, py_tree, layer_idx, seq_len):
        if seq_len % self.mini_batch_size == 0:
            # copy last mini-batch states, clear gradients
            for name in self.ttt_param_names:
                # 用 .detach().clone() 避免就地操作
                self.ttt_params_dict[f"{name}_states"][layer_idx] = py_tree[f"{name}_states"].detach().clone()
                # 不使用 zero_ 来清空梯度
                self.ttt_params_dict[f"{name}_grad"][layer_idx] = torch.zeros_like(
                    self.ttt_params_dict[f"{name}_grad"][layer_idx])

        elif seq_len < self.mini_batch_size:
            if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.mini_batch_size != 0:
                raise ValueError("fractional update not supported yet.")
            if (seq_len + self.seqlen_offset) % self.mini_batch_size == 0:
                # copy last mini-batch states, clear gradients
                for name in self.ttt_param_names:
                    # 同样使用 .detach().clone() 避免就地操作
                    self.ttt_params_dict[f"{name}_states"][layer_idx] = py_tree[f"{name}_states"].detach().clone()
                    # 不使用 zero_ 来清空梯度
                    self.ttt_params_dict[f"{name}_grad"][layer_idx] = torch.zeros_like(
                        self.ttt_params_dict[f"{name}_grad"][layer_idx])
            else:
                # copy gradients for the next update
                for name in self.ttt_param_names:
                    # 同样避免就地操作
                    self.ttt_params_dict[f"{name}_grad"][layer_idx] = py_tree[f"{name}_grad"].detach().clone()
        else:
            raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")


    def ttt_params_to_dict(self, layer_idx):
        return {name: self.ttt_params_dict[name][layer_idx] for name in self.ttt_params_dict}


#TTT层
class TTTBase(nn.Module):
    def __init__(self, num_heads, hidden_size, mini_batch_size):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.mini_batch_size = mini_batch_size

        # token_idx is a scale factor that scale the summation in Eqn. 4
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)

        self.register_buffer("token_idx", token_idx, persistent=False)
        # make the scale factor learnable
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))

        self._init_qkvo_proj()
        self._init_ttt_lr_gate()
        self._init_ttt_ln()

        self.post_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ttt_base_lr = 1.0

    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)


    #注意线性层最后的输出shape为1
    def _init_ttt_lr_gate(self):
        linear_weight_data = nn.Linear(self.hidden_size, 1, bias=True).weight.data
        # [num_heads, width, 1]
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        linear_bias_data = nn.Linear(self.hidden_size, 1, bias=True).bias.data
        # [num_heads, 1]
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data

        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.q_proj(hidden_states), #调用了 self.q_proj 的前向传播，执行了线性变化操作
            self.k_proj(hidden_states), #同理
            self.v_proj(hidden_states),
        )
        return XQ, XK, XV

    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)

        # [B, num_heads, num_mini_batch, 1, mini_batch_size] 32 8 2 1 64
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.ttt_base_lr * ttt_lr / self.head_dim

        # [B, L]
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[mini_batch_step_offset: mini_batch_step_offset + mini_batch_size]

        # token idx should be greast than 0
        token_idx = torch.clamp_min(token_idx, 0.0)

        # NOTE: token_eta is a scale factor that applies to each token in the mini-batch
        # [B, num_heads, num_mini_batch, mini_batch_size, 1] [32, 8, 2, 64, 1]
        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )

        return token_eta, ttt_lr_eta

    def get_ttt_inputs(self, inputs, mini_batch_size, cache_params):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X = inputs["X"]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        # [B ,num_mini_batch, mini_batch_size, C] 32 2 64 512
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.hidden_size)
        # [B, num_heads, L, head_dim] -> [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)

        if cache_params is not None:
            mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
        else:
            mini_batch_step_offset = 0
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)

        # [B, num_heads, num_mini_batch, mini_batch_size, mini_batch_size]
        eta = token_eta * ttt_lr_eta
        # decouple token_coeff and ilr_coeff for decoding
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }
        return inputs

    def ttt(
            self,
            inputs,
            mini_batch_size,
            last_mini_batch_params_dict,
            cache_params,
            lay_idx,
    ):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")


    def forward(self, hidden_states, cache_params, lay_idx):
        B, L = hidden_states.shape[:2]
        reminder_len = L % self.mini_batch_size
        num_mini_batch = L // self.mini_batch_size
        last_mini_batch_params_dict = None

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim] 32, 8, 128, 64
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        output_hidden_states = []
        # when input sequence length is not a multiple of mini_batch_size
        # we need to compute them seperately, when computing the reminder,
        # we will need the last_mini_batch_params_dict to continue TTT learning
        # 说人话就是凑整了用对偶的方法求解，没凑整只能一个一个求
        if num_mini_batch > 0:
            inputs = {
                "XQ": XQ[:, :, : num_mini_batch * self.mini_batch_size],
                "XK": XK[:, :, : num_mini_batch * self.mini_batch_size],
                "XV": XV[:, :, : num_mini_batch * self.mini_batch_size],
                "X": hidden_states[:, : num_mini_batch * self.mini_batch_size],
            }
            # return XQW_batch, batch_params_dict
            output_mod, last_mini_batch_params_dict = self.ttt(
                self.get_ttt_inputs(inputs, self.mini_batch_size, cache_params),
                mini_batch_size=self.mini_batch_size,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
                cache_params=cache_params,
                lay_idx=lay_idx,
            )
            output_hidden_states.append(output_mod)
        if reminder_len > 0:
            inputs = {
                "XQ": XQ[:, :, -reminder_len:],
                "XK": XK[:, :, -reminder_len:],
                "XV": XV[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            output_reminder, _ = self.ttt(
                self.get_ttt_inputs(inputs, reminder_len, cache_params),
                mini_batch_size=reminder_len,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
                cache_params=cache_params,
                lay_idx=lay_idx,
            )
            output_hidden_states.append(output_reminder)

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)

        return output_hidden_states


def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out


def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y


#grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)
def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma

    z = (
            (1.0 / D)
            * (
                    D * grad_x_hat
                    - grad_x_hat.sum(dim=-1, keepdim=True)
                    - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
            )
            / std
    )

    return z


class TTTLinear(TTTBase):
    def __init__(self, num_heads, hidden_size, mini_batch_size):
        super().__init__(num_heads, hidden_size, mini_batch_size)
        # TTT model initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(
            self,
            inputs,
            mini_batch_size,
            last_mini_batch_params_dict,
            cache_params,
            lay_idx
    ):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        # in this case, we are decoding
        if last_mini_batch_params_dict is None and cache_params is not None:
            #def ttt_params_to_dict(self, layer_idx):
            last_mini_batch_params_dict = cache_params.ttt_params_to_dict(lay_idx)

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        # NOTE:
        # for prefilling, we will always use dual form for faster computation
        # we need to use primal form if mini_batch_size is not a multiple of self.mini_batch_size
        # since we need store the gradient for the next mini-batch computation
        use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, f], nh=num_heads, f=head_dim 64
            W1_init = params_dict["W1_states"]
            # [B, nh, 1, f]
            b1_init = params_dict["b1_states"]

            # [B, nh, K,f], K=mini_batch_size  2, 32, 8, 64, 64
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B, nh, K, 1]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)

            # [B,nh,K,f]
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

            if use_dual_form:
                # [B,nh,K,K]
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]

                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                # [B,nh,1,f]
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                # [B, nh, K, f, f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)

                # [B, nh, K, f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch

                # [B, nh, K, 1, f] @ [B, nh, K, f, f]
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]
            #end else


            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)
            # f(x) = x + LN(linear(x))
            # x: XQ_mini_batch
            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
            }
            return last_param_dict, XQW_mini_batch
        #end def

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim] ->
        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] 2, 32, 8, 64, 64
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            0
        )
        # [B, num_heads, L, C]
        if cache_params is not None:
            #    def update(self, py_tree, layer_idx, seq_len):
            cache_params.update(batch_params_dict, lay_idx, L)   #将cache_params_dict更新为batch_params_dict

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] ->
        # [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.hidden_size)

        return XQW_batch, batch_params_dict