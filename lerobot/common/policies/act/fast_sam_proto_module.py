# ruff: noqa: N812

import types
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from ultralytics import FastSAM
from ultralytics.nn.modules.head import Segment

ImageInput = Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, Image.Image]]]
NUM_PROTOS = 32


class FastSAMProto(nn.Module):
    def __init__(
        self,
        imgsz: int = 512,
        weights: Union[str, Path] = "weights/FastSAM-x.pt",
        device: Optional[Union[str, torch.device]] = None,
        proto_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.imgsz = imgsz
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.proto_indices = proto_indices if proto_indices is not None else list(range(NUM_PROTOS))
        self.filter_protos = proto_indices is not None
        self._sam = FastSAM(str(weights)).eval().to(self.device).half()
        self._patched = False
        self.fc = nn.Identity()
        in_channels = len(self.proto_indices)
        out_channels = in_channels * 64  # mult by 16 because of space to depth pooling
        self.fc.in_features = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=8, stride=8, padding=0)

    def _init_predictor(self):
        dummy = torch.zeros(1, 3, self.imgsz, self.imgsz)
        self._sam(dummy, imgsz=self.imgsz, stream=False, verbose=False)
        seg: Segment = next(m for m in self._sam.model.model if isinstance(m, Segment))
        seg.forward = types.MethodType(lambda self, x: self.proto(x[0]), seg)
        self._sam.predictor.postprocess = lambda preds, *_, **__: preds
        self._patched = True

    def forward(self, images: ImageInput) -> torch.Tensor:
        # FastSAM expects images in 0-1 range
        # make sure that input to this is 0-255 range (aka no normalisation)
        images /= 255.0

        if not self._patched or self._sam.predictor is None:
            self._init_predictor()

        with torch.no_grad():
            x = torch.stack(self._sam(images.half(), imgsz=self.imgsz, stream=False, verbose=False), dim=0)

        if self.filter_protos:
            x = x[:, self.proto_indices]

        # avg pool to half w,h dim but keep channel size same
        # x = F.avg_pool2d(x, kernel_size=4, stride=4)

        # Learnt 8x8 convolution
        b, c, h, w = x.shape
        x = F.pad(x, (0, w % 8, 0, h % 8))  # pad so height and width are multiples of 8
        x = self.conv(x)

        return {"feature_map": x}

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._sam = self._sam.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        return self
