"""Plot observed score, threshold, and latency distributions from benchmark CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from detector_runner import ROOT
from visualize_bev import render


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(path)


def ground_truth(frame: str, data_root: Path) -> list[dict]:
    calib = {}
    for line in (data_root / 'training/calib' / f'{frame}.txt').read_text().splitlines():
        if ':' not in line:
            continue
        key, values = line.split(':', 1)
        calib[key] = np.fromstring(values, sep=' ')
    rect = calib['R0_rect'].reshape(3, 3)
    velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    rotation = rect @ velo_to_cam[:, :3]
    translation = rect @ velo_to_cam[:, 3]
    out = []
    for line in (data_root / 'training/label_2' / f'{frame}.txt').read_text().splitlines():
        v = line.split()
        if v[0] not in ('Car', 'Pedestrian', 'Cyclist'):
            continue
        h, w, length = map(float, v[8:11])
        loc = np.asarray(list(map(float, v[11:14])))
        loc[1] -= h / 2  # KITTI annotation location is the box bottom center.
        center = np.linalg.solve(rotation, loc - translation)
        ry = float(v[14])
        heading = np.linalg.solve(rotation, np.array([np.cos(ry), 0, -np.sin(ry)]))[:2]
        heading /= np.linalg.norm(heading)
        out.append(dict(class_name=v[0], center=center, heading=heading,
                        h=h, w=w, length=length, occluded=int(v[2])))
    return out


def points_in_gt(frame: str, data_root: Path, gt: dict) -> int:
    pts = np.fromfile(data_root / 'training/velodyne' / f'{frame}.bin', dtype=np.float32).reshape(-1, 4)
    delta = pts[:, :3] - gt['center']
    longitudinal = delta[:, :2] @ gt['heading']
    lateral = delta[:, :2] @ np.array([-gt['heading'][1], gt['heading'][0]])
    return int(np.sum((abs(longitudinal) <= gt['length']/2) &
                      (abs(lateral) <= gt['w']/2) & (abs(delta[:, 2]) <= gt['h']/2)))


def find_failure(det: list[dict], data_root: Path, low: float, high: float) -> dict | None:
    by_frame = defaultdict(list)
    for row in det:
        row = dict(row)
        row['score'] = float(row['score'])
        row['box'] = json.loads(row['box'])
        row['center_distance_m'] = float(row['center_distance_m'])
        by_frame[row['frame_id']].append(row)
    candidates = []
    missed = []
    for frame, predictions in by_frame.items():
        for gt in ground_truth(frame, data_root):
            # A higher-score detection at the same GT center means the GT survives.
            survives = any(p['class_name'] == gt['class_name'] and p['score'] >= high
                           and np.linalg.norm(np.asarray(p['box'][:2]) - gt['center'][:2]) <= 2.0
                           for p in predictions)
            if not survives:
                missed.append(dict(frame_id=frame, class_name=gt['class_name'], score=None,
                                   range_m=float(np.linalg.norm(gt['center'][:2])),
                                   occluded=gt['occluded'], points_in_gt=points_in_gt(frame, data_root, gt),
                                   gt_center_xy=gt['center'][:2].tolist(), predictions=predictions,
                                   failure_type='GT unmatched at high threshold'))
            if survives:
                continue
            matches = [p for p in predictions if p['class_name'] == gt['class_name']
                       and low <= p['score'] < high
                       and np.linalg.norm(np.asarray(p['box'][:2]) - gt['center'][:2]) <= 2.0]
            if not matches:
                continue
            best = max(matches, key=lambda p: p['score'])
            candidates.append(dict(frame_id=frame, class_name=gt['class_name'],
                                   score=best['score'], range_m=float(np.linalg.norm(gt['center'][:2])),
                                   occluded=gt['occluded'], points_in_gt=points_in_gt(frame, data_root, gt),
                                   gt_center_xy=gt['center'][:2].tolist(), predictions=predictions,
                                   failure_type='GT-associated detection dropped by threshold'))
    if not candidates:
        candidates = [x for x in missed if x['points_in_gt'] > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c['class_name'] in ('Pedestrian', 'Cyclist'), c['range_m']), reverse=True)
    return candidates[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data/kitti_mini')
    args = p.parse_args()
    out = args.results_dir / 'figures'
    sweep = read_csv(args.results_dir / 'threshold_sweep.csv')
    det = read_csv(args.results_dir / 'detection_results.csv')
    latency = read_csv(args.results_dir / 'latency.csv')

    scores = defaultdict(list)
    for row in det:
        scores[row['class_name']].append(float(row['score']))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, values in scores.items():
        ax.hist(values, bins=20, range=(0, 1), alpha=.55, label=f'{cls} (n={len(values)})')
    ax.set(xlabel='PointPillars score', ylabel='Detections', title='Observed detection scores')
    ax.legend()
    save(fig, out / 'score_histogram.png')

    counts = defaultdict(int)
    for row in sweep:
        counts[float(row['score_thr'])] += int(row['num_detections'])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sorted(counts), [counts[t] for t in sorted(counts)], marker='o')
    ax.set(xlabel='Score threshold', ylabel='Total detections across fixed frames',
           title='Threshold sweep')
    ax.grid(alpha=.3)
    save(fig, out / 'threshold_vs_detections.png')

    times = [float(row['latency_ms']) for row in latency]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(times, bins=min(15, max(5, len(times)//2)))
    ax.set(xlabel='Inference latency (ms)', ylabel='Timed runs', title='Latency distribution')
    save(fig, out / 'latency_distribution.png')

    first, last = min(counts), max(counts)
    reduction = (counts[first] - counts[last]) / counts[first] * 100 if counts[first] else None
    summary = dict(total_detections={str(k): v for k, v in sorted(counts.items())},
                   reduction_percent_low_to_high=reduction)
    (args.results_dir / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

    failure = find_failure(det, args.data_root, first, last)
    if failure:
        frame = failure['frame_id']
        path = out / f'fail_01_threshold_drop_{frame}.png'
        text = (f"GT {failure['class_name']} {failure['range_m']:.1f} m\n"
                + (f"score {failure['score']:.2f} < {last:.2f}" if failure['score'] is not None
                   else f'unmatched at {last:.2f}'))
        render(frame, failure['predictions'], args.data_root, path,
               title=f"KITTI {frame}: {failure['failure_type']}",
               highlight=dict(gt_center_xy=failure['gt_center_xy'], text=text))
        failure.pop('predictions')
        failure.update(low_threshold=first, high_threshold=last,
                       matching_rule='same class, BEV center distance <= 2 m; proxy, not IoU',
                       figure=str(path.relative_to(ROOT)))
        (args.results_dir / 'failure_case.json').write_text(json.dumps(failure, indent=2), encoding='utf-8')
        print(json.dumps(failure, indent=2))
    else:
        print('No GT-associated threshold-drop failure found under the stated matching rule.')


if __name__ == '__main__':
    main()
