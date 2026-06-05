#!/usr/bin/env python

import argparse

import cv2
import torch

from lerobot.datasets import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay a live camera feed on a LeRobot dataset frame.")
    parser.add_argument("--repo-id", default="jclinton1/gem_stack_blocks_three_20260502_194746")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index for the live head camera.")
    parser.add_argument("--view", default="observation.images.head", help="Dataset image key to use as reference.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0, help="Frame index within the selected episode.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Reference opacity in the overlay.")
    return parser.parse_args()


def tensor_chw_to_bgr(image: torch.Tensor) -> torch.Tensor:
    if image.dtype != torch.uint8:
        image = (image.clamp(0, 1) * 255).to(torch.uint8)
    image = image.permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def center_crop_or_resize(frame, height: int, width: int):
    frame_height, frame_width = frame.shape[:2]
    if frame_height >= height and frame_width >= width:
        y = (frame_height - height) // 2
        x = (frame_width - width) // 2
        return frame[y : y + height, x : x + width]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    dataset = LeRobotDataset(
        args.repo_id,
        episodes=[args.episode],
        video_backend="pyav",
        return_uint8=True,
    )
    if args.view not in dataset.features:
        raise KeyError(f"{args.view!r} is not in dataset features: {list(dataset.features)}")
    if not 0 <= args.frame < len(dataset):
        raise IndexError(f"--frame must be in [0, {len(dataset) - 1}] for episode {args.episode}")

    reference = tensor_chw_to_bgr(dataset[args.frame][args.view])
    height, width = reference.shape[:2]

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    alpha = max(0.0, min(1.0, args.alpha))
    cv2.namedWindow("align", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("align", width * 3, height)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            live = center_crop_or_resize(frame, height, width)
            overlay = cv2.addWeighted(reference, alpha, live, 1.0 - alpha, 0)
            cv2.imshow("align", cv2.hconcat([reference, live, overlay]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
