import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange, repeat

from ...util import append_dims, default
import open_clip

logpy = logging.getLogger(__name__)


class Guider(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        pass


class VanillaCFG(Guider):
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        print(sigma)
        x_u, x_c = x.chunk(2)
        x_pred = x_u + self.scale * (x_c - x_u)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out
    

class SparseVanillaCFG(Guider):
    def __init__(self, scale: float, sigma_lower: float, sigma_upper: float):
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        print(sigma)
        x_u, x_c = x.chunk(2)
        if sigma > self.sigma_lower and sigma < self.sigma_upper:
            x_pred = x_u + self.scale * (x_c - x_u)
            return x_pred
        return x_u

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class ClipGuider(Guider):
    def __init__(self, model = 'Latent-ViT-B-8-512',
                       pretrained = '/dlabdata1/wendler/models/latent-clip-b-8.pt',
                       tokenizer = None, 
                       scale=1.0,
                       scale_clip=1.0,
                       prompt="painting"):
        #self.device = 'cpu'
        self.prompt = prompt
        self.model, _, _ = open_clip.create_model_and_transforms(model, \
                pretrained=pretrained)
        if tokenizer is None:
            tokenizer = model
        self.tokenizer = open_clip.get_tokenizer(tokenizer)
        self.model.cuda()
        self.device = 'cuda'
        self.scale = scale
        self.scale_clip = scale_clip
        self.txt_feat = self.model.encode_text(self.tokenizer(self.prompt).to(self.device))


    def clip_score(self, x):
        txt_feat = self.txt_feat
        img_feat = self.model.encode_image(x)
        txt_feat_normalized = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        img_feat_normalized = img_feat / img_feat.norm(dim=-1, keepdim=True)
        return (txt_feat_normalized @ img_feat_normalized.T).squeeze()

    def __call__(self, x, sigma):
        x_u, x_c = x.chunk(2)
        # perform the normal guidance
        x_pred = x_u + self.scale * (x_c - x_u)
        # clip guidance
        x_ = x_pred.clone().detach().requires_grad_(True)
        # print whether x_ requires grad
        print(x_.requires_grad)
        score = self.clip_score(x_)
        print(score)
        #score.backward()
        grad = torch.autograd.grad(score, x_)[0]
        normalizer = x_c.norm()/x_u.norm()
        x_pred = x_pred + self.scale_clip * sigma * normalizer * (grad - x_pred)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]

        return torch.cat([x] * 2), torch.cat([s] * 2), c_out

class IdentityGuider(Guider):
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out


class LinearPredictionGuider(Guider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_frames = num_frames
        self.scale = torch.linspace(min_scale, max_scale, num_frames).unsqueeze(0)

        additional_cond_keys = default(additional_cond_keys, [])
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)

        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=self.num_frames)
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=self.num_frames)
        scale = repeat(self.scale, "1 t -> b t", b=x_u.shape[0])
        scale = append_dims(scale, x_u.ndim).to(x_u.device)

        return rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")

    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"] + self.additional_cond_keys:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out
