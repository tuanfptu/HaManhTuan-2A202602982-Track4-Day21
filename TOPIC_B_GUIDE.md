# Topic B — PointPillars KITTI baseline

## Environment and model

Run on Windows with an NVIDIA GPU and driver supporting the PyTorch CUDA 12.1 runtime. Measured hardware: NVIDIA GeForce GTX 1660 Ti, 6 GB VRAM (driver 610.62). The experiment uses Python 3.10.20 in the isolated `.detector-venv/`; the lab `.venv/` is untouched. Installed versions: PyTorch 2.1.2+cu121, torchvision 0.16.2+cu121, NumPy 1.26.4, MMEngine 0.10.7, MMCV 2.1.0, MMDetection 3.3.0, MMDetection3D 1.4.0, matplotlib 3.10.7. The [MMDetection3D v1.4.0 compatibility table](https://mmdetection3d.readthedocs.io/en/v1.4.0/notes/faq.html) supports this OpenMMLab version combination. The prebuilt [Windows MMCV CUDA wheel](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html) avoids compiling CUDA extensions locally.

Model: official [PointPillars KITTI three-class checkpoint](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/configs/pointpillars/metafile.yml) (`pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class`). Checkpoint SHA256 `37dc242098f0d3dd7d870e0fe7cd4514f76fd85a04fb1377681279f963b3cbe5`. `src/setup_detector.py` downloads it, its config/base files from immutable v1.4.0 tag, and the MMCV wheel into ignored `external/`; it verifies hashes. No model/dataset download is committed.

## Recreate the environment

PowerShell commands from the repository root (requires [`uv`](https://docs.astral.sh/uv/) and internet for the setup stage):

```powershell
$env:UV_CACHE_DIR = (Join-Path (Get-Location) '.uv-cache')
$env:UV_PYTHON_INSTALL_DIR = (Join-Path (Get-Location) '.uv-python')
uv python install 3.10.20
uv venv .detector-venv --python .uv-python\cpython-3.10.20-windows-x86_64-none\python.exe
uv pip install --python .detector-venv\Scripts\python.exe torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
.uv-python\cpython-3.10.20-windows-x86_64-none\python.exe src/setup_detector.py
uv pip install --python .detector-venv\Scripts\python.exe numpy==1.26.4 mmengine==0.10.7 mmdet==3.3.0 mmdet3d==1.4.0 matplotlib==3.10.7 .\external\mmcv-2.1.0-cp310-cp310-win_amd64.whl
```

## Run and regenerate evidence

```powershell
.detector-venv\Scripts\python.exe tools/verify_data.py --data-root data/kitti_mini
.detector-venv\Scripts\python.exe src/detector_runner.py --frame 000011 --score-thr 0.10 --output results/demo_000011.json
.detector-venv\Scripts\python.exe src/benchmark_detector.py --runs 20 --warmup 3
.detector-venv\Scripts\python.exe src/visualize_bev.py --frames 000011,000001,000004,000008 --score-thr 0.30
.detector-venv\Scripts\python.exe src/analyze_results.py
.detector-venv\Scripts\python.exe tools/check_submission.py
```

All CLIs support `--help`. `benchmark_detector.py` defaults to the same 12 frames and thresholds 0.10/0.30/0.50 used in the report. It loads the model once, runs one inference per frame, then applies only a score postfilter. Inference latency is measured on frame 000011 after three warm-up runs, with 20 timed runs and CUDA synchronization around each run. Model load is excluded; data read, preprocessing and postprocessing are included. Batch size is one.

## Config and interpretation

Config values extracted from the loaded model are in `results/benchmark_summary.json` and `results/demo_000011.json`: `point_cloud_range=[0,-39.68,-3,69.12,39.68,1]`, `voxel_size=[0.16,0.16,4]`, class order `Pedestrian,Cyclist,Car`, internal `score_thr=0.1`, rotated NMS threshold `0.01`, pre-NMS top 100, maximum 50 predictions. Test pipeline loads 4D KITTI `.bin` LiDAR points, applies deterministic test transform/range filter, then packs points. This is **single-frame KITTI PointPillars, no temporal multi-sweep aggregation**.

The 12-frame mini subset is selected for presentation and has no valid full KITTI AP estimate. The threshold sweep measures exported box counts, not false positive rate or recall. The failure matcher is a documented same-class BEV center distance ≤2 m proxy. The reported point count is inside the converted KITTI GT box. Latency can change with GPU load and other active processes. Production evaluation requires a larger validation set, rigorous 3D/BEV IoU matching, and workload-level latency tests.
