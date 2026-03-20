#!/usr/bin/env python3
"""Record robot motion as MP4 video using MuJoCo.

Must run inside the 'gmr' conda environment.

Usage:
    conda run -n gmr python scripts/record_robot_video.py \
        --motion data/Danza_peruana_huayno/robot_motion_h1_2.pkl \
        --robot unitree_h1_2 \
        --output data/Danza_peruana_huayno/video_robot_h1_2.mp4

    # Or via visualize.py wrapper (auto-selects env):
    python scripts/visualize.py --project data/Danza_peruana_huayno \
        --robot --robot-type unitree_h1_2 --record
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "GMR"))

from general_motion_retargeting import load_robot_motion, RobotMotionViewer


def main():
    parser = argparse.ArgumentParser(description="Record robot motion as MP4")
    parser.add_argument("--motion", required=True, help="Path to robot_motion*.pkl")
    parser.add_argument("--robot", required=True, help="Robot type (e.g. unitree_g1, unitree_h1_2)")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--width", type=int, default=1280, help="Video width (default 1280)")
    parser.add_argument("--height", type=int, default=720, help="Video height (default 720)")
    args = parser.parse_args()

    motion_path = Path(args.motion)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    _, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(str(motion_path))
    num_frames = len(root_pos)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[RobotRecord] Robot    : {args.robot}")
    print(f"[RobotRecord] Frames   : {num_frames}  FPS: {fps:.1f}")
    print(f"[RobotRecord] Output   : {output_path}")
    print(f"[RobotRecord] Size     : {args.width}x{args.height}")
    print("[RobotRecord] Starting MuJoCo viewer (rate_limit=False for fast recording)...")

    env = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=fps,
        camera_follow=False,
        record_video=True,
        video_path=str(output_path),
        video_width=args.width,
        video_height=args.height,
    )

    for i in range(num_frames):
        env.step(root_pos[i], root_rot[i], dof_pos[i], rate_limit=False)
        if (i + 1) % 30 == 0 or i == num_frames - 1:
            print(f"\r[RobotRecord] Frame {i+1}/{num_frames}", end="", flush=True)

    print()
    env.close()
    print(f"[RobotRecord] Done! Video saved to: {output_path}")


if __name__ == "__main__":
    main()
