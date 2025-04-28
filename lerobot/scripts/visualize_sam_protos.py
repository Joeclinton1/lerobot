#!/usr/bin/env python3
"""
FastSAM – Prototype Channel Visualiser (Refined ProtoExtractor)
=================================================================

Left  : live webcam frame
Right : colour–mapped prototype channel (alpha blended)

Controls
--------
q                – quit
Camera Index     – switch camera
Prototype Index  – browse prototype channels
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

from lerobot.common.policies.act.fast_sam_proto_module import FastSAMProto


# ───────────────────────── helpers ─────────────────────────
def default_backend() -> int:
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("darwin"):
        return cv2.CAP_AVFOUNDATION
    return 0


def find_cams(max_test: int = 5) -> list[int]:
    be = default_backend()
    out: list[int] = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, be)
        if cap.isOpened():
            out.append(i)
            cap.release()
    return out


# ───────────────────────── main ────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    # camera detection & setup
    cams = find_cams()
    if not cams:
        print("No cameras detected.")
        return
    cam_pos = cams.index(args.camera) if args.camera in cams else 0
    cur_cam = cams[cam_pos]
    cap = cv2.VideoCapture(cur_cam, default_backend())
    if not cap.isOpened():
        print(f"Cannot open camera {cur_cam}")
        return

    # window + UI
    win = "FastSAM Prototypes"
    cv2.namedWindow(win, cv2.WINDOW_GUI_EXPANDED)
    cv2.createTrackbar("Camera Index", win, cam_pos, len(cams) - 1, lambda _: None)
    cv2.createTrackbar("Prototype Index", win, 0, 0, lambda _: None)

    # Setup model
    model = FastSAMProto(imgsz=384, device="cuda")

    prev, fps = time.time(), 0.0
    proto_channels = None

    while True:
        # camera switch
        new_cam_idx = cv2.getTrackbarPos("Camera Index", win)
        new_cam = cams[new_cam_idx]
        if new_cam != cur_cam:
            cap.release()
            cap = cv2.VideoCapture(new_cam, default_backend())
            if cap.isOpened():
                cur_cam = new_cam
                print(f"Switched to camera {cur_cam}")

        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        # inference
        with torch.no_grad(), autocast(model.device != torch.device("cpu")):
            img_t = (
                torch.from_numpy(frame)  # H,W,3 uint8
                .permute(2, 0, 1)  # 3,H,W
                .to(model.device)  # move to GPU
                .half()  # fastsam wants fp16
                .unsqueeze(0)  # 1,3,H,W
            )
            protos = model(img_t)[0]

        c, _, _ = protos.shape  # C,H,W

        # adjust trackbar max
        if proto_channels != c:
            proto_channels = c
            cv2.setTrackbarMax("Prototype Index", win, c - 1)

        # select channel
        k = min(cv2.getTrackbarPos("Prototype Index", win), c - 1)
        proto_k = protos[k].cpu().numpy()

        # normalize & colour-map
        mn, mx = proto_k.min(), proto_k.max()
        if mx - mn > 1e-6:
            proto_uint8 = ((proto_k - mn) * (255.0 / (mx - mn))).astype(np.uint8)
        else:
            proto_uint8 = np.zeros_like(proto_k, dtype=np.uint8)
        proto_resized = cv2.resize(proto_uint8, (frame.shape[1], frame.shape[0]))
        colour = cv2.applyColorMap(proto_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.5, colour, 0.5, 0.0)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - prev))
        prev = now
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay, f"Proto {k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # display
        display = np.hstack((frame, overlay))
        cv2.imshow(win, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
