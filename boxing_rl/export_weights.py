"""
Export Boxing PPO model weights to JSON for browser inference.
Usage:
    python export_weights.py --save-dir models --out-dir ../public/models
"""
import argparse, json, os, sys
import torch
from boxing_env import OBS_DIM, ACT_DIM
from ppo_agent import PPOAgent


def model_to_json(path: str) -> dict:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd   = ckpt['policy']

    def t(key): return sd[key].numpy().tolist()

    return {
        'shared_0_weight': t('shared.0.weight'),
        'shared_0_bias':   t('shared.0.bias'),
        'shared_2_weight': t('shared.2.weight'),
        'shared_2_bias':   t('shared.2.bias'),
        'actor_weight':    t('actor_head.weight'),
        'actor_bias':      t('actor_head.bias'),
        'meta': {
            'total_steps': int(ckpt.get('total_steps', 0)),
            'updates':     int(ckpt.get('updates', 0)),
            'obs_dim':     OBS_DIM,
            'act_dim':     ACT_DIM,
        },
    }


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
