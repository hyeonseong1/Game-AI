"""
Export Space Invaders RL weights to JSON for browser inference.

Supports:
    - rainbow_feature_c51 checkpoints from train.py
    - legacy/simba_v2_discrete checkpoints from older trainers

Usage:
    python export_weights.py --save-dir models --out-dir ../public/models
"""
import argparse, json, os, sys
from pathlib import Path
from space_invaders_env import OBS_DIM, ACT_DIM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from rl_common import checkpoint_to_json
from rainbow_agent import checkpoint_to_rainbow_json


def model_to_json(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("architecture") == "rainbow_feature_c51":
        return checkpoint_to_rainbow_json(path)
    return checkpoint_to_json(path, state_dim=OBS_DIM, action_dim=ACT_DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save-dir', default='models')
    ap.add_argument('--out-dir',  default='../public/models')
    ap.add_argument('--file-prefix', default='si',
                    help='Output file prefix, e.g. si -> si_mid.json/si_best.json')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for src, suffix in [('mid_model.pt', 'mid'),
                        ('best_model.pt', 'best')]:
        dst = f'{args.file_prefix}_{suffix}.json'
        path = os.path.join(args.save_dir, src)
        if not os.path.exists(path):
            print(f'  [SKIP] {path}', file=sys.stderr); continue
        data = model_to_json(path)
        out  = os.path.join(args.out_dir, dst)
        with open(out, 'w') as f: json.dump(data, f)
        print(f'  [OK] {dst}  (steps={data["meta"]["total_steps"]:,}, '
              f'obs={OBS_DIM}, act={ACT_DIM})')

if __name__ == '__main__':
    main()
