# Topic B: PointPillars baseline

## Experiment

Use the official MMDetection3D KITTI three-class PointPillars checkpoint on the 20-frame `data/kitti_mini` subset. The primary comparison holds the model, checkpoint, input frames, voxelization, NMS, and hardware fixed while filtering predictions at score thresholds 0.10, 0.30, and 0.50. Benchmark batch size is one, with model load excluded, three warm-up runs, and at least 20 timed runs. CUDA is synchronized on both sides of each timed run.

Selected frames: `000011,000015,000001,000007,000008,000010,000004,000009,000012,000016,000049,000019`.

## Model source

- [Official model index](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/metafile.yml): `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class`.
- Checkpoint: `external/checkpoints/pointpillars_kitti_3class.pth` (ignored by Git).
- Config and inherited base files: `external/configs/` (ignored by Git).
- KITTI mini is the lab's original data; no additional dataset is needed.

## Local environment

Use Python 3.10 in `.detector-venv/`. The official Windows MMCV wheel for PyTorch 2.1/CUDA 12.1 is linked from [OpenMMLab's wheel index](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html). Dependencies are isolated from the lab's `.venv`.

## Commands

From the repository root in PowerShell, after installing the dependencies and checkpoint:

```powershell
.detector-venv\Scripts\python.exe src/detector_runner.py --frame 000011 --score-thr 0.10
.detector-venv\Scripts\python.exe src/benchmark_detector.py
.detector-venv\Scripts\python.exe src/visualize_bev.py
.detector-venv\Scripts\python.exe src/analyze_results.py
```

The commands above will be validated after environment setup and then copied into `report/REPORT.md` with the exact package versions and measured hardware details.
