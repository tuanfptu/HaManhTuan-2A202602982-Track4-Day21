"""Create a top-down LiDAR plot with scored PointPillars boxes."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from detector_runner import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATA, ROOT, infer_frame, load_model

COLORS = {'Car': '#ffca3a', 'Pedestrian': '#ff595e', 'Cyclist': '#8ac926'}


def corners(box: list[float]) -> np.ndarray:
    x, y, _, dx, dy, _, yaw = box[:7]
    local = np.array([[dx/2, dy/2], [dx/2, -dy/2], [-dx/2, -dy/2], [-dx/2, dy/2], [dx/2, dy/2]])
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def render(frame: str, detections: list[dict], data_root: Path, output: Path,
           title: str = '', highlight: dict | None = None) -> None:
    points = np.fromfile(data_root / 'training/velodyne' / f'{frame}.bin', dtype=np.float32).reshape(-1, 4)
    points = points[np.isfinite(points).all(axis=1)]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.scatter(points[:, 1], points[:, 0], c=points[:, 3], cmap='Greys', s=.1, alpha=.5,
               vmin=0, vmax=1, rasterized=True)
    for det in detections:
        poly = corners(det['box'])
        color = COLORS.get(det['class_name'], 'cyan')
        ax.plot(poly[:, 1], poly[:, 0], color=color, lw=1.5)
        front = (poly[0] + poly[1]) / 2
        center = np.asarray(det['box'][:2])
        ax.plot([center[1], front[1]], [center[0], front[0]], color=color, lw=2)
        ax.text(det['box'][1], det['box'][0], f"{det['class_name']} {det['score']:.2f}",
                fontsize=6, color=color, bbox=dict(facecolor='black', alpha=.7, pad=1, edgecolor='none'))
    ax.scatter([0], [0], marker='^', s=150, color='deepskyblue', label='Ego')
    if highlight:
        x, y = highlight['gt_center_xy']
        ax.scatter([y], [x], marker='x', s=180, linewidths=3, color='magenta', label='KITTI GT center')
        ax.annotate(highlight['text'], (y, x), xytext=(y + 3, x + 3), color='magenta',
                    arrowprops=dict(arrowstyle='->', color='magenta'), fontsize=9,
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='magenta'))
    ax.set(xlim=(-40, 40), ylim=(-10, 75), xlabel='LiDAR y (m, left)', ylabel='LiDAR x (m, forward)',
           title=title or f'KITTI {frame}: PointPillars detections')
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
    ax.legend(loc='upper right')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', default='000011,000001,000004')
    p.add_argument('--score-thr', type=float, default=.3)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--out-dir', type=Path, default=ROOT / 'results/figures')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    model = load_model(args.config, args.checkpoint, args.device)
    for frame in args.frames.split(','):
        det = infer_frame(model, frame, args.data_root, args.score_thr)
        path = args.out_dir / f'bev_{frame}.png'
        render(frame, det, args.data_root, path)
        print(f'{path}: {len(det)} detections')


if __name__ == '__main__':
    main()
