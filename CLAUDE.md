# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Pipeline Overview

```
[Prompt/Video] → (phmr env) Veo/Sora → original.mp4
                           ↓
              PromptHMR → smplx.npz
                           ↓
              (gmr env) GMR → robot_motion.pkl
```

**Two conda environments are required and used automatically:**
- `phmr` — video generation (Veo/Sora) + PromptHMR pose extraction
- `gmr` — GMR robot motion retargeting

`scripts/run_pipeline.py` orchestrates both envs via `conda run` using `video2robot.utils.run_in_conda()`.

## Common Commands

All commands are run from the repo root (`/home/josu/video2robot`).

```bash
# Full pipeline from video file
python scripts/run_pipeline.py --video /path/to/video.mp4

# Full pipeline from video with specific robot
python scripts/run_pipeline.py --video /path/to/video.mp4 --robot unitree_g1

# Full pipeline from text prompt (requires GOOGLE_API_KEY)
python scripts/run_pipeline.py --action "Action sequence: The subject walks forward."

# Resume from existing project (skips already-completed steps)
python scripts/run_pipeline.py --project data/Danza_peruana_huayno

# Run individual steps
python scripts/extract_pose.py --project data/video_001 [--static-camera]
python scripts/convert_to_robot.py --project data/video_001 --robot unitree_g1

# Visualization
python scripts/visualize.py --project data/video_001 --robot

# Web UI
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

### GPU Memory Constraint (RTX 3050 6GB)

Always use `--static-camera` to skip DROID-SLAM + Metric3D (requires ~4GB extra VRAM):

```bash
python scripts/run_pipeline.py --video /path/to/video.mp4 --static-camera
```

## Architecture

### `video2robot/` Package (Orchestration Layer)

- **`config.py`** — All path constants (`PROJECT_ROOT`, `PROMPTHMR_DIR`, `GMR_DIR`, `GMR_BODY_MODELS_DIR`, etc.)
- **`utils.py`** — `run_in_conda()` (subprocess env switching), `emit_progress()` (structured stdout for web UI)
- **`pose/extractor.py`** — Wraps PromptHMR subprocess call; applies coordinate transform (`_PROMPTHMR_TO_GMR_COORD_TRANSFORM`) and saves per-track `smplx_track_N.npz` files
- **`robot/retargeter.py`** — Lazy-loads GMR; calls `load_smplx_file()` then runs IK retargeting frame-by-frame
- **`video/`** — Veo (`veo_client.py`) and Sora (`sora_client.py`) API clients
- **`visualization/robot_viser.py`** — MuJoCo + viser-based 3D viewer

### `scripts/` Entry Points

Each script corresponds to one pipeline stage and handles argparse + env coordination:
- `run_pipeline.py` — Orchestrates all 3 steps, calls `run_in_conda()` for env switching
- `extract_pose.py` — Runs inside `phmr` env; calls PromptHMR then `convert_all_prompthmr_tracks_to_smplx()`
- `convert_to_robot.py` — Runs inside `gmr` env; calls `RobotRetargeter.retarget()`

### `third_party/` Dependencies

- **`PromptHMR/`** — Pose extraction. Entry: `pipeline/pipeline.py`, model: `prompt_hmr/models/phmr.py`
- **`GMR/`** — Motion retargeting. Entry: `general_motion_retargeting/motion_retarget.py`, SMPLX loading: `general_motion_retargeting/utils/smpl.py`

### Project Data Layout

Each video project lives in `data/<name>/`:
```
data/my_project/
  original.mp4            ← input video
  smplx.npz               ← default track alias (→ smplx_track_1.npz)
  smplx_track_1.npz       ← SMPL-X pose per tracked person
  smplx_tracks.json       ← track metadata
  robot_motion.pkl         ← default robot motion alias
  robot_motion_track_1.pkl ← 29 DOF robot motion
  robot_motion_twist.pkl   ← 23 DOF TWIST format
  results.pkl              ← raw PromptHMR output
  world4d.glb / .mcs      ← 3D mesh for Blender/viewer
```

## Known Bugs Fixed in This Repo

These patches are already applied but relevant if pulling upstream:

1. **`pipeline/pipeline.py`** — Added `gc.collect(); torch.cuda.empty_cache()` after detect/keypoint stages to free GPU memory before loading PHMR model
2. **`pipeline/detector/sam2_video_predictor.py`** — Removed unused `load_video_frames_from_np` import (not in sam2 1.1.0)
3. **`pipeline/droidcalib/src/altcorr_kernel.cu` + `correlation_kernels.cu`** — Patched `.type()` → `.scalar_type()` for PyTorch 2.x API
4. **`third_party/GMR/general_motion_retargeting/utils/smpl.py`** — Added `num_betas=num_betas_in_file` to `smplx.create()` in `load_smplx_file()` to match 10-beta files against 16-beta model default
5. **`/home/josu/.local/lib/python3.10/site-packages/chumpy/__init__.py`** — Patched deprecated `from numpy import bool, int...` for SMPL v1.0.0 loading

## Body Model Setup

```
third_party/PromptHMR/data/body_models/
  smplx/           ← symlink to /home/josu/GMR/assets/body_models/smplx
  smpl/            ← contains SMPL_NEUTRAL.pkl (created from SMPL_python_v.1.0.0.zip male+female average)
  smplx2smpl.pkl
  smplx2smpl_joints.npy
  J_regressor_h36m.npy

third_party/GMR/assets/body_models/  ← symlink to /home/josu/GMR/assets/body_models
```

## Environment Variables

```bash
GOOGLE_API_KEY   # For Veo video generation
OPENAI_API_KEY   # For Sora video generation
```

## Output Format

```python
# robot_motion.pkl
{
    "fps": 30.0,
    "robot_type": "unitree_g1",
    "num_frames": 240,
    "root_pos": np.ndarray,  # (N, 3)
    "root_rot": np.ndarray,  # (N, 4) quaternion xyzw
    "dof_pos":  np.ndarray,  # (N, DOF)
}
```

## Supported Robots

`unitree_g1` (29 DOF), `unitree_h1` (19 DOF), `booster_t1` (23 DOF), and others — see `video2robot/robot/retargeter.py:SUPPORTED_ROBOTS`.
