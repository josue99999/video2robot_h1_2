#!/usr/bin/env python3
"""Record pose extraction visualization as MP4.

Overlays 2D skeleton (ViTPose COCO-17) and bounding boxes on top of the
original video to show the pose extraction result.

Must run inside the 'phmr' conda environment.

Usage:
    conda run -n phmr python scripts/record_pose_video.py \
        --project data/Danza_peruana_huayno

    # Custom output path:
    conda run -n phmr python scripts/record_pose_video.py \
        --project data/Danza_peruana_huayno \
        --output data/Danza_peruana_huayno/video_pose.mp4

    # Or via visualize.py wrapper:
    python scripts/visualize.py --project data/Danza_peruana_huayno --pose --record
"""
import argparse
import cv2
import joblib
import numpy as np
from pathlib import Path


# COCO-17 skeleton connections (joint index pairs)
COCO17_PAIRS = [
    (0, 1), (0, 2),       # nose → eyes
    (1, 3), (2, 4),       # eyes → ears
    (5, 6),               # shoulder bar
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (5, 11), (6, 12),     # torso sides
    (11, 12),             # hip bar
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
]

# Per-limb BGR colors
LIMB_COLORS_BGR = [
    (255, 0, 255),   (255, 0, 255),   # head
    (255, 0, 255),   (255, 0, 255),   # ears
    (0, 165, 255),                    # shoulders
    (0, 255, 255),   (0, 255, 255),   # left arm
    (255, 165, 0),   (255, 165, 0),   # right arm
    (0, 255, 0),     (0, 255, 0),     # torso
    (0, 255, 0),                      # hip bar
    (0, 200, 255),   (0, 200, 255),   # left leg
    (255, 100, 100), (255, 100, 100), # right leg
]

# Per-track bbox / skeleton tint colors (BGR)
TRACK_COLORS_BGR = [
    (0, 165, 255),   # orange
    (0, 255, 0),     # green
    (255, 165, 0),   # blue-ish
    (255, 0, 255),   # magenta
]


def draw_pose(frame: np.ndarray, kpts: np.ndarray, conf_thresh: float = 0.25) -> None:
    """Draw COCO-17 skeleton on frame in-place.

    Args:
        frame: HxWx3 BGR uint8 image (modified in-place)
        kpts:  (17, 3) array — columns: x, y, confidence
        conf_thresh: minimum confidence to draw a joint/limb
    """
    # Limbs
    for i, (a, b) in enumerate(COCO17_PAIRS):
        if kpts[a, 2] > conf_thresh and kpts[b, 2] > conf_thresh:
            p1 = (int(kpts[a, 0]), int(kpts[a, 1]))
            p2 = (int(kpts[b, 0]), int(kpts[b, 1]))
            color = LIMB_COLORS_BGR[i] if i < len(LIMB_COLORS_BGR) else (255, 255, 255)
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    # Joints
    for j in range(min(17, len(kpts))):
        if kpts[j, 2] > conf_thresh:
            cx, cy = int(kpts[j, 0]), int(kpts[j, 1])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)


def main():
    parser = argparse.ArgumentParser(description="Record pose overlay video")
    parser.add_argument("--project", "-p", required=True, help="Project folder")
    parser.add_argument("--output", default=None, help="Output MP4 path (default: <project>/video_pose.mp4)")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Keypoint confidence threshold")
    args = parser.parse_args()

    project_dir = Path(args.project)
    video_path = project_dir / "original.mp4"
    results_path = project_dir / "results.pkl"
    output_path = Path(args.output) if args.output else project_dir / "video_pose.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"results.pkl not found: {results_path}")

    print(f"[PoseRecord] Loading results.pkl ...")
    results = joblib.load(results_path)
    people = results.get("people", {})
    if not people:
        raise RuntimeError("No people found in results.pkl")

    # Collect per-person data
    all_vitpose: dict[str, np.ndarray] = {}
    all_bboxes: dict[str, np.ndarray] = {}
    all_frame_ids: dict[str, np.ndarray] = {}
    for pid, person in people.items():
        vp = person.get("vitpose")
        bboxes = person.get("bboxes")
        frames = person.get("frames")
        if vp is not None:
            all_vitpose[pid] = np.asarray(vp)
        if bboxes is not None:
            all_bboxes[pid] = np.asarray(bboxes)
        if frames is not None:
            all_frame_ids[pid] = np.asarray(frames, dtype=int)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    print(f"[PoseRecord] Video     : {W}x{H} @ {fps:.1f} fps  ({total_frames} frames)")
    print(f"[PoseRecord] People    : {len(people)}")
    print(f"[PoseRecord] Output    : {output_path}")

    # Build fast per-frame lookup: frame_id → {pid: local_index}
    frame_to_person_idx: dict[int, dict[str, int]] = {}
    for pid, fids in all_frame_ids.items():
        for local_idx, fid in enumerate(fids):
            frame_to_person_idx.setdefault(int(fid), {})[pid] = int(local_idx)

    person_ids = list(all_vitpose.keys())

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        person_map = frame_to_person_idx.get(frame_idx, {})
        for ti, pid in enumerate(person_ids):
            if pid not in person_map:
                continue
            li = person_map[pid]
            bbox_color = TRACK_COLORS_BGR[ti % len(TRACK_COLORS_BGR)]

            # Bounding box
            if pid in all_bboxes:
                b = all_bboxes[pid][li]
                if b is not None and len(b) == 4:
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                    label = f"Person {ti+1}"
                    cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bbox_color, 2, cv2.LINE_AA)

            # Skeleton
            if pid in all_vitpose:
                draw_pose(frame, all_vitpose[pid][li], conf_thresh=args.conf_thresh)

        # HUD
        cv2.putText(frame, f"Pose Extraction  |  Frame {frame_idx:04d}",
                    (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"\r[PoseRecord] Frame {frame_idx}/{total_frames}", end="", flush=True)

    print()
    cap.release()
    out.release()
    print(f"[PoseRecord] Done!  {frame_idx} frames  →  {output_path}")


if __name__ == "__main__":
    main()
