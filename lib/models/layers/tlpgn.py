import functools

import torch
import torchvision

import torch.nn as nn

from lib.models.layers.resnet import ResNet


def _get_act_fn(act_fn):
    if act_fn == 'softmax':
        return functools.partial(torch.softmax, dim=-1)
    elif act_fn == 'sigmoid':
        return torch.sigmoid
    else:
        raise ValueError("Invalid IDP activation function")


class TLPGN(nn.Module):
    def __init__(
            self,
            nr_output_vectors,
            mixture_size,
            vector_dim,
            pgn_act_fn,  #激活函数
            pgn_resolution,  #图像处理的分辨率

    ) -> None:
        super().__init__()
        # print("No topk!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.nr_output_vectors = nr_output_vectors
        self.mixture_size = mixture_size
        self.vector_dim = vector_dim
        self.resolution = pgn_resolution
        self.model = ResNet(
            num_classes=4096,
            proj_type='linear',
            num_blocks=[1, 1, 1, 1],
            c_hidden=[16, 32, 64, 128],
            init_max_pool=True,

        )

        tl_vectors = torch.empty(
            mixture_size,
            vector_dim,
            # dtype=self.dtype,
            # device=self.device,
        )
        torch.nn.init.normal_(tl_vectors, std=0.02)  #提示向量在模型中通过随机初始化（标准差为 0.02 的正态分布）生成
        self.tl_vectors = torch.nn.Parameter(tl_vectors)

        self.pgn_act_fn = pgn_act_fn
        self.act_fn = self._get_act_fn(pgn_act_fn)

        self.w_k = nn.Linear(768, 256)
        self.w_v = nn.Linear(768, 768)


    def forward(self, images):
        images = torchvision.transforms.functional.resize(images, self.resolution)

        logits = self.model(images)

        q = logits.reshape(
            len(logits),
            self.nr_output_vectors,
            self.mixture_size
        )

        k = v = self.tl_vectors
        k = self.w_k(k)
        v = self.w_v(v)

        k = k.permute(1, 0)

        attn_weights = torch.matmul(q, k) / (256 ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        pgn_prompts = torch.matmul(attn_weights, v)


        return pgn_prompts

    @staticmethod
    def _get_act_fn(act_fn):
        if act_fn == 'softmax':
            return functools.partial(torch.softmax, dim=-1)
        elif act_fn == 'sigmoid':
            return torch.sigmoid
        else:
            raise ValueError("Invalid PGN activation function")
