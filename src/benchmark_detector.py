"""Benchmark PointPillars on fixed KITTI frames with a score threshold sweep."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from detector_runner import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATA, ROOT, config_metadata, infer_frame, load_model

DEFAULT_FRAMES = '000011,000015,000001,000007,000008,000010,000004,000009,000012,000016,000049,000019'


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(frame: str, threshold: float, detections: list[dict]) -> dict:
    scores = [d['score'] for d in detections]
    ranges = [d['center_distance_m'] for d in detections]
    return dict(frame_id=frame, score_thr=threshold, num_detections=len(detections),
                num_car=sum(d['class_name'] == 'Car' for d in detections),
                num_pedestrian=sum(d['class_name'] == 'Pedestrian' for d in detections),
                num_cyclist=sum(d['class_name'] == 'Cyclist' for d in detections),
                mean_score=statistics.mean(scores) if scores else '',
                median_score=statistics.median(scores) if scores else '',
                min_score=min(scores) if scores else '', max_score=max(scores) if scores else '',
                mean_range_m=statistics.mean(ranges) if ranges else '',
                max_range_m=max(ranges) if ranges else '')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', default=DEFAULT_FRAMES, help='Comma-separated KITTI frame IDs')
    p.add_argument('--thresholds', default='0.10,0.30,0.50')
    p.add_argument('--runs', type=int, default=20)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--latency-frame', default='000011')
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'results')
    args = p.parse_args()
    if args.runs < 20:
        p.error('--runs must be at least 20')
    frames = args.frames.split(',')
    thresholds = [float(x) for x in args.thresholds.split(',')]
    model = load_model(args.config, args.checkpoint, args.device)
    # Run inference once per frame, then apply only the requested score threshold.
    # This keeps checkpoint, voxelization, NMS and hardware constant.
    all_predictions = {frame: infer_frame(model, frame, args.data_root, 0.0) for frame in frames}
    rows = [summarize(frame, threshold, [d for d in all_predictions[frame] if d['score'] >= threshold])
            for threshold in thresholds for frame in frames]
    write_csv(args.out_dir / 'threshold_sweep.csv', rows)
    write_csv(args.out_dir / 'detection_results.csv',
              [dict(frame_id=frame, class_name=d['class_name'], score=d['score'],
                    center_distance_m=d['center_distance_m'], box=json.dumps(d['box']))
               for frame in frames for d in all_predictions[frame]])

    import torch
    for _ in range(args.warmup):
        infer_frame(model, args.latency_frame, args.data_root, 0.0)
    times = []
    for run in range(args.runs):
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        infer_frame(model, args.latency_frame, args.data_root, 0.0)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    gpu = torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'
    write_csv(args.out_dir / 'latency.csv',
              [dict(frame_id=args.latency_frame, run=i + 1, latency_ms=ms, device=gpu,
                    pytorch=torch.__version__, cuda=torch.version.cuda) for i, ms in enumerate(times)])
    summary = dict(device=gpu, pytorch=torch.__version__, cuda=torch.version.cuda,
                   frame_id=args.latency_frame, warmup=args.warmup, timed_runs=args.runs,
                   mean_ms=float(np.mean(times)), p50_ms=float(np.percentile(times, 50)),
                   p95_ms=float(np.percentile(times, 95)), min_ms=min(times), max_ms=max(times),
                   config=config_metadata(model))
    (args.out_dir / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'config'}, indent=2))


if __name__ == '__main__':
    main()
