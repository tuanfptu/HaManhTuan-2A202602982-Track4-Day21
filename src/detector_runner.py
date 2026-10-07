"""Run the official MMDetection3D KITTI PointPillars checkpoint on one LiDAR frame.

Model/config source: https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'external/configs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
DEFAULT_CHECKPOINT = ROOT / 'external/checkpoints/pointpillars_kitti_3class.pth'
DEFAULT_DATA = ROOT / 'data/kitti_mini'


def load_model(config: Path = DEFAULT_CONFIG, checkpoint: Path = DEFAULT_CHECKPOINT,
               device: str = 'cuda:0'):
    # MMEngine expands inherited configs via NamedTemporaryFile. The default
    # Windows TEMP may be inaccessible in managed lab environments.
    tmp = ROOT / 'external/tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = str(tmp)
    os.environ['TMP'] = str(tmp)
    tempfile.tempdir = str(tmp)
    from mmdet3d.apis import init_model
    if not config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f'Missing model file: {config} or {checkpoint}. See TOPIC_B_GUIDE.md')
    return init_model(str(config), str(checkpoint), device=device)


def infer_frame(model, frame_id: str, data_root: Path = DEFAULT_DATA,
                score_thr: float = 0.1) -> list[dict]:
    from mmdet3d.apis import inference_detector
    path = data_root / 'training/velodyne' / f'{frame_id}.bin'
    if not path.is_file():
        raise FileNotFoundError(path)
    sample, _ = inference_detector(model, str(path))
    pred = sample.pred_instances_3d
    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()
    classes = tuple(model.dataset_meta['classes'])
    return [dict(frame_id=frame_id, box=b.tolist(), score=float(s),
                 label=int(label), class_name=classes[int(label)],
                 center_distance_m=float(np.linalg.norm(b[:2])))
            for b, s, label in zip(boxes, scores, labels) if s >= score_thr]


def config_metadata(model) -> dict:
    """Extract from the actual loaded config; do not assume default values."""
    cfg = model.cfg
    test_cfg = cfg.model.test_cfg
    pipeline = cfg.test_pipeline
    return dict(point_cloud_range=list(cfg.point_cloud_range), voxel_size=list(cfg.voxel_size),
                class_names=list(cfg.class_names), score_thr=float(test_cfg.score_thr),
                nms={k: test_cfg[k] for k in ('use_rotate_nms', 'nms_across_levels', 'nms_thr',
                                              'nms_pre', 'max_num')},
                test_pipeline=[dict(x) for x in pipeline],
                sweeps='single-frame KITTI PointPillars, no temporal multi-sweep aggregation')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame', default='000011')
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--score-thr', type=float, default=0.1)
    p.add_argument('--output', type=Path, help='Optional JSON output path')
    args = p.parse_args()
    model = load_model(args.config, args.checkpoint, args.device)
    result = dict(metadata=config_metadata(model), detections=infer_frame(model, args.frame, args.data_root, args.score_thr))
    body = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding='utf-8')
    print(body)


if __name__ == '__main__':
    main()
