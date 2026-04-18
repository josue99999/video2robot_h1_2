# video2robot — Human Video to Robot Motion

> Convert any video of a person moving into robot motion data.  
> **Synthetic demonstration generation pipeline for humanoid robots — entirely from video.**

```
[Video / Prompt] → PromptHMR → SMPL-X pose → GMR IK → robot_motion.pkl
```

---

## Demos

### Huayno Dance (Peruvian Folk)

| Original | Unitree H1-2 | Unitree G1 |
|:---:|:---:|:---:|
| ![huayno](huayno.gif) | ![H1-2](video_robot_h1_2.gif) | ![G1](video_robot_g1.gif) |

### Caporal Dance (Peru)

| Original | Unitree H1-2 | Unitree G1 |
|:---:|:---:|:---:|
| ![caporal-original](original.gif) | ![caporal-H1-2](video_robot_h1_2_caporal.gif) | ![caporal-G1](video_robot_g1_caporal.gif) |

---

## What This Project Does

This project builds a **fully automated pipeline** that takes a video of a human moving and converts it into robot joint trajectories — without any manual annotation or teleoperation.

The output is structured robot motion data ready to be used as **demonstration data for Imitation Learning** (Behavior Cloning, motion tracking RL, or diffusion policy training).

The pipeline is **robot-agnostic**: once the human pose is extracted, it can be retargeted to any supported humanoid without reprocessing the video.

**This directly addresses the "Synthetic Data Generation" requirement** — producing large-scale, human-like demonstration data entirely from video, usable as training input for robotic policies.

---

## Pipeline Architecture

```
Video (.mp4)  or  Text Prompt
        │
        ▼
  PromptHMR  ──────────────────────  monocular 3D human pose estimation
  (phmr env)                         outputs SMPL-X parameters per frame
        │
        ▼  smplx.npz
        │
        ▼
   GMR IK  ────────────────────────  motion retargeting to robot joint space
  (gmr env)                          maps SMPL-X → robot DOF angles
        │
        ▼
  robot_motion.pkl  ──────────────  structured trajectory ready for training
```

Two isolated conda environments handle the two stages automatically. The orchestrator script (`run_pipeline.py`) switches between them transparently using `conda run`.

---

## Components

### 1. PromptHMR — Human Pose Estimation

- Monocular 3D pose estimation from a single RGB video
- Outputs **SMPL-X body model parameters** (shape, pose, translation) per frame
- Supports both **video files** and **text prompts** as input (text → video via generative model API, then pose extraction)
- Handles multi-person tracking; each detected person is saved as a separate track (`smplx_track_N.npz`)

### 2. GMR — Motion Retargeting (Inverse Kinematics)

- Takes SMPL-X pose sequences and solves **robot-specific IK** to produce joint angle trajectories
- Retargeting is robot-agnostic: same `smplx.npz` can be converted to any supported robot without re-running pose extraction
- Output format includes root position, root quaternion, and per-DOF joint angles

### 3. Output Format

```python
# robot_motion.pkl
{
    "fps":        30.0,
    "robot_type": "unitree_g1",
    "num_frames": 300,
    "root_pos":   np.ndarray,  # (N, 3)   root XYZ position
    "root_rot":   np.ndarray,  # (N, 4)   root quaternion (xyzw)
    "dof_pos":    np.ndarray,  # (N, DOF) joint angles [rad]
}
```

This format is directly consumable by **Behavior Cloning**, **motion tracking RL environments**, or **Diffusion Policy** training pipelines.

---

## Supported Robots

| Robot | ID | DOF |
|---|---|---|
| Unitree G1 | `unitree_g1` | 29 |
| Unitree G1 + hands | `unitree_g1_with_hands` | — |
| Unitree H1 | `unitree_h1` | 19 |
| Unitree H1-2 | `unitree_h1_2` | 27 |
| Booster T1 | `booster_t1` | 23 |
| Booster T1 29DOF | `booster_t1_29dof` | 29 |
| Booster K1 | `booster_k1` | — |
| Fourier N1 | `fourier_n1` | — |
| Stanford Toddy | `stanford_toddy` | — |
| EngineAI PM01 | `engineai_pm01` | — |
| Kuavo S45 | `kuavo_s45` | — |
| Galaxea R1 Pro | `galaxea_r1pro` | — |

---

## Why This Matters for Robotics AI

Traditional demonstration collection requires physical teleoperation — expensive, slow, and hard to scale. This pipeline enables:

- **Scalable synthetic data generation** from any internet video or text description
- **Human-like motion quality** because the source is real human movement
- **Cross-robot reuse** — one video generates demonstrations for multiple robot morphologies
- **Zero hardware required** during data collection

This is directly applicable as a data source for **Behavior Cloning**, **Diffusion Policy training**, and **motion tracking RL** pipelines.

---

## Top 3 Hardest Problems Solved

### Problem 1 — CUDA Extension Compilation Across Conflicting Dependencies

PromptHMR requires multiple CUDA-compiled extensions (lietorch, droidcalib, detectron2, sam2) that conflict with each other and with PyTorch's bundled CUDA runtime. Build failures were common because `nvcc` was not aligned with the installed CUDA version and NVIDIA header paths were not exposed to the compiler. I solved this by isolating the phmr environment completely, explicitly setting `PATH` to CUDA 12.8 binaries, and exporting `CPATH` from pip-installed NVIDIA packages so the build system could find all required headers. Each extension was compiled individually in the correct order with architecture flags matching the target GPU (`compute_86,code=sm_86`). The result was a fully reproducible build that works on RTX 3050 6GB with minimal VRAM usage.

### Problem 2 — NumPy Compatibility Break in SMPL Body Model Dependencies

The `chumpy` library — a core dependency of the SMPL body model — uses deprecated NumPy type aliases (`numpy.bool`, `numpy.int`, `numpy.float`) that were removed in NumPy 1.24, causing a hard crash at import time. This was a silent blocker because the error only appeared when the body model was loaded mid-pipeline, not at install time. I patched `chumpy/__init__.py` programmatically at install time to replace all deprecated aliases with their modern equivalents (`numpy.bool_`, `numpy.int_`, `numpy.float64`). The patch is applied automatically during setup so users never encounter the issue. This unblocked the full SMPL-X loading pipeline without requiring a downgrade of NumPy.

### Problem 3 — Robot-Agnostic Retargeting with Correct Joint Space Mapping

Different humanoid robots have different kinematic structures, joint limits, and DOF counts, making it non-trivial to retarget a single SMPL-X sequence to multiple targets correctly. A naive mapping produces physically invalid poses — joints exceeding limits, inverted limb orientations, or root drift. I structured the retargeting layer so each robot has its own configuration (joint mapping, axis conventions, limit clamping) while sharing the same IK solver core from GMR. The pipeline separates pose extraction (done once) from retargeting (done per robot) so the same `smplx.npz` can be converted to any target without re-running the expensive PromptHMR stage. The result is consistent, physically plausible motion across all supported robots from a single source video.

---

## Project Structure

```
video2robot_h1_2/
├── scripts/
│   ├── run_pipeline.py        # Main orchestrator (handles env switching)
│   ├── extract_pose.py        # Step 1: PromptHMR pose extraction
│   ├── convert_to_robot.py    # Step 2: GMR retargeting
│   └── visualize.py           # MuJoCo + viser visualization
├── video2robot/
│   ├── config/                # Per-robot retargeting configs
│   ├── pose/                  # SMPL-X processing utilities
│   ├── retargeter/            # GMR IK wrapper
│   ├── api/                   # Generative video API (text → video)
│   └── visualization/         # 3D viewer utilities
├── third_party/
│   ├── PromptHMR/             # Monocular pose estimation model
│   └── GMR/                   # Motion retargeting library
└── data/<project>/            # Per-run artifacts (video, poses, robot motion)
```

---

## Credits

- [PromptHMR](https://github.com/yufu-wang/PromptHMR) — monocular 3D human pose estimation
- [GMR](https://github.com/YanjieZe/GMR) — motion retargeting to humanoid robots

## License

Core code: MIT · PromptHMR: non-commercial research only.
