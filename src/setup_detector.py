"""Download the official MMDetection3D PointPillars KITTI config and checkpoint.

Sources: https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/metafile.yml
The files are stored below Git-ignored external/ and are never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE = 'https://raw.githubusercontent.com/open-mmlab/mmdetection3d/v1.4.0/configs/'
CHECKPOINT_URL = ('https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/'
                  'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/'
                  'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth')
MMCV_URL = 'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-win_amd64.whl'
FILES = {
    ROOT / 'external/configs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py':
        BASE + 'pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py',
    ROOT / 'external/_base_/models/pointpillars_hv_secfpn_kitti.py':
        BASE + '_base_/models/pointpillars_hv_secfpn_kitti.py',
    ROOT / 'external/_base_/datasets/kitti-3d-3class.py':
        BASE + '_base_/datasets/kitti-3d-3class.py',
    ROOT / 'external/_base_/schedules/cyclic-40e.py':
        BASE + '_base_/schedules/cyclic-40e.py',
    ROOT / 'external/_base_/default_runtime.py': BASE + '_base_/default_runtime.py',
    ROOT / 'external/checkpoints/pointpillars_kitti_3class.pth': CHECKPOINT_URL,
    ROOT / 'external/mmcv-2.1.0-cp310-cp310-win_amd64.whl': MMCV_URL,
}
CHECKPOINT_SHA256 = '37dc242098f0d3dd7d870e0fe7cd4514f76fd85a04fb1377681279f963b3cbe5'
MMCV_SHA256 = '78b161e1498d21f39210452e908286cc7115ec1e223fa1fa8931daabedf08fd7'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--force', action='store_true', help='Download existing files again')
    args = p.parse_args()
    for path, url in FILES.items():
        if args.force or not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + '.part')
            print(f'Downloading {url} -> {path}', flush=True)
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        print(f'{path}: {path.stat().st_size:,} bytes')
    checkpoint = ROOT / 'external/checkpoints/pointpillars_kitti_3class.pth'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint SHA256 mismatch: {digest}')
    print('Official checkpoint SHA256 verified.')
    wheel = ROOT / 'external/mmcv-2.1.0-cp310-cp310-win_amd64.whl'
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != MMCV_SHA256:
        raise ValueError(f'MMCV wheel SHA256 mismatch: {digest}')
    print('Official MMCV wheel SHA256 verified.')


if __name__ == '__main__':
    main()
