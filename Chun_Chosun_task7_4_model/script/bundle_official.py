"""
bundle_official.py — 공식 제출용 단일 dictionary pth 생성

dcase.community/challenge2026/submission 요구:
  [label]_D2_dictionary.pth / [label]_D3_dictionary.pth (각 1파일) + [label]_model.py

flat dictionary(개별 pth 30개)에서:
  - 공유 frozen backbone(전 모델 비트 동일, D1 pretrained)을 1회만 저장
  - 모델별 domain_loras 만 개별 저장
  - D2 = 5-fold 전체(step2 시스템, count=1)
  - D3 = val_macro auto-N 선정 모델(step3 최종 시스템, count=2)

사용: python bundle_official.py
"""

import os
import sys
import glob

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_MODEL)
sys.path.insert(0, os.path.join(_MODEL, 'common'))
os.environ.setdefault('CONFIG', os.path.join(_MODEL, 'config.yaml'))
LABEL = 'Chun_Chosun_task7_4'
D2_DICT = os.path.join(_ROOT, f'{LABEL}_D2_dictionary')
D3_DICT = os.path.join(_ROOT, f'{LABEL}_D3_dictionary')


def split_state(sd):
    bb = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
    lr = {k[len('domain_loras.'):]: v for k, v in sd.items() if k.startswith('domain_loras.')}
    assert len(bb) + len(lr) == len(sd), '예상 밖 키 존재'
    return bb, lr


def main():
    # ── D3: auto-N 선정 (val_macro ≥ 평균, inference.py 와 동일 기준) ──────────
    d3_paths = sorted(glob.glob(os.path.join(D3_DICT, 'd3_d2f*_d3f*.pth')))
    assert len(d3_paths) == 25, f'D3 25개 기대, {len(d3_paths)}개'
    metas = []
    for p in d3_paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        metas.append({'path': p, 'name': os.path.splitext(os.path.basename(p))[0],
                      'val_macro': float(ck['val_macro'])})
        del ck
    import config as c
    mean_vm = sum(m['val_macro'] for m in metas) / len(metas)
    ranked = sorted(metas, key=lambda m: m['val_macro'], reverse=True)
    sub_n = (c.CFG.get('kfold', {}) or {}).get('submission_n', None)
    if sub_n is not None:
        n = min(int(sub_n), len(ranked))
        sel = ranked[:n]
        sel_desc = f'val_macro Top-{n} 고정(config submission_n; dev-test 미참조)'
        print(f'D3 Top-N 고정: {n}/{len(metas)} (config kfold.submission_n; val_macro 상위 N, test 미참조)')
    else:
        sel = [m for m in ranked if m['val_macro'] >= mean_vm]
        sel_desc = f'val_macro >= mean({mean_vm:.4f}) auto-N, dev-test 미참조'
        print(f'D3 auto-N: {len(sel)}/{len(metas)} (val_macro ≥ 평균 {mean_vm:.2f})')

    backbone_state = None
    d3_models = []
    for m in sel:
        ck = torch.load(m['path'], map_location='cpu', weights_only=False)
        bb, lr = split_state(ck['model'])
        if backbone_state is None:
            backbone_state = bb
        else:
            assert all(torch.equal(backbone_state[k], bb[k]) for k in backbone_state), \
                f'backbone 불일치: {m["name"]}'
        d3_models.append({'name': m['name'], 'val_macro': m['val_macro'], 'loras': lr})
        del ck
        print(f'  + {m["name"]} (val_macro={m["val_macro"]:.2f})')

    out3 = os.path.join(_MODEL, 'result', f'{LABEL}_D3_dictionary.pth')
    torch.save({'backbone': backbone_state, 'models': d3_models,
                'active_domain_count': 2, 'task': 3,
                'selection': sel_desc},
               out3)
    print(f'→ {out3} ({os.path.getsize(out3)/1e6:.0f} MB)')

    # ── D2: 5-fold 전체 (step2 시스템) ────────────────────────────────────────
    d2_paths = sorted(glob.glob(os.path.join(D2_DICT, 'd2_fold*.pth')))
    assert len(d2_paths) == 5, f'D2 5개 기대, {len(d2_paths)}개'
    d2_models = []
    for p in d2_paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        bb, lr = split_state(ck['model'])
        assert all(torch.equal(backbone_state[k], bb[k]) for k in backbone_state), \
            f'backbone 불일치: {p}'
        name = os.path.splitext(os.path.basename(p))[0]
        d2_models.append({'name': name,
                          'val_macro': float(ck.get('val_macro', -1)), 'loras': lr})
        del ck
        print(f'  + {name}')

    out2 = os.path.join(_MODEL, 'result', f'{LABEL}_D2_dictionary.pth')
    torch.save({'backbone': backbone_state, 'models': d2_models,
                'active_domain_count': 1, 'task': 2,
                'selection': '5-fold 전체 soft-voting'},
               out2)
    print(f'→ {out2} ({os.path.getsize(out2)/1e6:.0f} MB)')


if __name__ == '__main__':
    main()
