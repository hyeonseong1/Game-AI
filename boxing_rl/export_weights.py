"""
Export Boxing PPO model weights to JSON for browser inference.
Usage:
    python export_weights.py --save-dir models --out-dir ../public/models
"""
import argparse, json, os, sys
from pathlib import Path
from boxing_env import OBS_DIM, ACT_DIM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_common import checkpoint_to_json


def model_to_json(path: str) -> dict:
    return checkpoint_to_json(path, state_dim=OBS_DIM, action_dim=ACT_DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save-dir', default='models')
    ap.add_argument('--out-dir',  default='../public/models')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for src_name, dst_name in [('mid_model.pt', 'boxing_mid.json'),
                                ('best_model.pt', 'boxing_best.json')]:
        src = os.path.join(args.save_dir, src_name)
        if not os.path.exists(src):
            print(f'  [SKIP] {src} not found', file=sys.stderr)
            continue
        data = model_to_json(src)
        dst  = os.path.join(args.out_dir, dst_name)
        with open(dst, 'w') as f:
            json.dump(data, f)
        steps = data['meta']['total_steps']
        print(f'  [OK] {dst_name}  (steps={steps:,}, '
              f'obs={OBS_DIM}, act={ACT_DIM})')


if __name__ == '__main__':
    main()
