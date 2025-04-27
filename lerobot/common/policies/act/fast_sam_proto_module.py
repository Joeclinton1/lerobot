import types
from pathlib import Path
from typing import Optional, Sequence, Union, List

import numpy as np
import torch
from torch import nn
from ultralytics import FastSAM
from ultralytics.nn.modules.head import Segment


ImageInput = Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, "PIL.Image.Image"]]]
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
        self.filter_protos =  proto_indices is not None
        self.proto_indices = proto_indices if self.filter_protos else list(range(NUM_PROTOS))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        with torch.inference_mode():
            sam = FastSAM(str(weights)).eval().to(self.device).half()
            sam.fuse()
            sam.eval()
            sam.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz)

            seg: Segment = next(m for m in sam.model.model if isinstance(m, Segment))
            seg.forward = types.MethodType(lambda self, x: self.proto(x[0]), seg)
            sam.predictor.postprocess = lambda preds, *_, **__: preds

        self._sam = sam

    @torch.inference_mode()
    def forward(self, images: ImageInput) -> List[torch.Tensor]:
        preds = self._sam(images, imgsz=self.imgsz, stream=False, verbose=False)
        if self.filter_protos : preds = [p[:, self.proto_indices, :, :] for p in preds]
        return preds

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)  # Move FastSAMProto's own parameters
        self._sam = self._sam.to(*args, **kwargs)  # Move the wrapped FastSAM manually
        return self
