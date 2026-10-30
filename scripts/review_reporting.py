"""Rebuild uncertainty reports, primary-resolution detail measurements and DiT charts.

CPU only. Reads saved evaluation results and images; does not run any models.
Usage: python review_reporting.py [--root PROJECT] [--output-root PROJECT]
Requires numpy and Pillow. Paths and identifiers are not embedded in outputs.
"""
import argparse
import csv
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def paired(a, b):
    assert set(a) == set(b), 'Prompt IDs must match exactly.'
    d = np.array([a[k] - b[k] for k in sorted(a, key=int)], dtype=np.float64)
    assert len(d) == 24 and np.isfinite(d).all(), 'Expected 24 finite paired scores.'
    mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    return dict(n=len(d), mean=mean, se=se, ci95=[mean-2.068657610*se, mean+2.068657610*se])


def font(size, bold=False):
    name = 'arialbd.ttf' if bold else 'arial.ttf'
    try:
        return ImageFont.truetype(str(Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/name), size)
    except OSError:
        return ImageFont.load_default(size=size)


def color(row):
    f = row['family']
    if 'failed' in f: return '#777777'
    if 'folded' in f: return '#8054b8'
    if 'distilled bf16' in f: return '#dc8500'
    if 'bf16 teacher' in f: return '#222222'
    if 'non-distilled' in f: return '#1578b5'
    return '#c44949'


def charts(rows, chart_dir, fig_dir):
    chart_dir.mkdir(parents=True, exist_ok=True)
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
    for kind in ['likeness', 'quality']:
        im = Image.new('RGB', (1800, 1200), 'white')
        d = ImageDraw.Draw(im)
        title = 'DiT file size and likeness to the 25-step reference' if kind=='likeness' else 'Local HPSv3-based score difference to the 25-step reference'
        d.text((70,30),title,font=font(37,True),fill='#182b3a')
        d.text((70,86),'24 fixed prompt-seed pairs; file sizes in decimal GB',font=font(25),fill='#455565')
        if kind=='likeness':
            top,bottom=190,1070
            xp=lambda x:610+(math.log(x)-math.log(1.2))/(math.log(11)-math.log(1.2))*400
            lp=lambda y:1160+y/100*470
            d.text((620,140),'Model files (GB, log scale)',font=font(26,True),fill='#182b3a')
            d.text((1150,140),'DINOv2 cosine x 100',font=font(26,True),fill='#182b3a')
            for x in [1.3,2,3,5,10]:
                xx=xp(x);d.line((xx,top,xx,bottom),fill='#e2e7eb',width=1);d.text((xx-15,bottom+12),str(x),font=font(22),fill='#455565')
            for x in [0,20,40,60,80,100]:
                xx=lp(x);d.line((xx,top,xx,bottom),fill='#e2e7eb',width=1);d.text((xx-15,bottom+12),str(x),font=font(22),fill='#455565')
            for i,r in enumerate(rows,1):
                yy=top+(i-.5)*(bottom-top)/len(rows);c=color(r)
                d.text((65,yy-12),f"{i}. {r['label']}",font=font(23),fill='#182b3a')
                for x in [xp(float(r['size_gb'])),lp(float(r['dino_likeness_x100']))]:
                    d.ellipse((x-7,yy-7,x+7,yy+7),fill=c)
                d.text((1030,yy-12),f"{float(r['size_gb']):.2f}",font=font(22),fill=c)
                d.text((1660,yy-12),f"{float(r['dino_likeness_x100']):.1f}",font=font(22),fill=c)
            d.text((70,1145),'Aligned rows share a model. Likeness is similarity, not percentage image quality; values: Table I4.',font=font(24),fill='#455565')
        else:
            valid=[r for r in rows if r['hpsv3_delta_to_teacher']!='']
            left,right,top,bottom=600,1640,170,1090
            low=min(float(r['hpsv3_ci95_low']) for r in valid)-.6
            high=max(float(r['hpsv3_ci95_high']) for r in valid)+.4
            xp=lambda x:left+(x-low)/(high-low)*(right-left)
            for x in [-22,-20,-15,-10,-5,0,1]:
                if low<=x<=high:
                    xx=xp(x); d.line((xx,top,xx,bottom),fill='#8496a5' if x==0 else '#e2e7eb',width=3 if x==0 else 1)
                    d.text((xx-18,bottom+12),str(x),font=font(22),fill='#455565')
            for i,r in enumerate(valid):
                yy=top+(i+.5)*(bottom-top)/len(valid)
                label=f"{r['label']}  ({float(r['size_gb']):.2f} GB)"
                d.text((65,yy-13),label,font=font(23),fill='#182b3a')
                lo,hi,mean=[float(r[k]) for k in ['hpsv3_ci95_low','hpsv3_ci95_high','hpsv3_delta_to_teacher']]
                c=color(r);d.line((xp(lo),yy,xp(hi),yy),fill=c,width=4)
                d.line((xp(lo),yy-6,xp(lo),yy+6),fill=c,width=2);d.line((xp(hi),yy-6,xp(hi),yy+6),fill=c,width=2)
                x=xp(mean);d.ellipse((x-6,yy-6,x+6,yy+6),fill=c)
                d.text((1660,yy-12),f'{mean:+.2f}',font=font(23),fill=c)
            d.text((600,1140),'Mean difference; bars: exploratory paired t 95% intervals',font=font(25),fill='#455565')
        name='dit_likeness_vs_size.png' if kind=='likeness' else 'dit_quality_vs_size.png'
        im.save(chart_dir/name)
        if fig_dir is not None:
            im.save(fig_dir/('v4_fig32_dit_likeness.png' if kind=='likeness' else 'v4_fig33_dit_quality.png'))


def detail_one(item):
    tag, pid, path = item
    with Image.open(path) as im:
        im=im.convert('RGB').resize((768,768),Image.Resampling.LANCZOS)
        rgb=np.asarray(im,dtype=np.float64)/255.0
    y=rgb@np.array([.2126,.7152,.0722])
    lap=y[:-2,1:-1]+y[2:,1:-1]+y[1:-1,:-2]+y[1:-1,2:]-4*y[1:-1,1:-1]
    return tag,str(pid),dict(luminance_mean=float(y.mean()),contrast=float(y.std()),laplacian_rms=float(np.sqrt(np.mean(lap*lap))))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(os.environ.get('LOWBITMOSS_HOME',Path(__file__).resolve().parents[1])))
    ap.add_argument('--output-root',type=Path)
    ap.add_argument('--skip-images',action='store_true')
    args=ap.parse_args();root=args.root;out=args.output_root or root
    build=out/'paper/_build';build.mkdir(parents=True,exist_ok=True)
    sources=['hps3_summary.json','hps3_turbo.json','hps3_turbo_fold.json','hps3_tmixG.json','hps3_tmixG_tuned.json','hps3_tmixG_traj.json','hps3_tmixH.json','hps3_nd_gptq.json','hps3_vae_full_ref.json','hps3_vae_final.json']
    scores={}
    for name in sources:
        for tag,v in json.loads((root/'eval'/name).read_text(encoding='utf-8-sig')).items():
            if isinstance(v,dict) and 'mu_by_prompt' in v:
                scores.setdefault(tag,v['mu_by_prompt'])
    comparisons=[('mixH2_x_f8','mixH2_x'),('mixH2_xm_f8','mixH2_x_f8'),('mixH2_x','mixH_x'),('tmixG_x_1024','turbo4_1024'),('turboF_mixG_tuned','turbo4_1024'),('nd_mixG_tuned_gptq','mixG_tuned'),('nd_mixH2_x_f8_gptq','mixH2_x_f8')]
    report={'method':'mean paired difference, sample SD/sqrt(24), Student t critical 2.068657610 (df=23); exploratory, no multiplicity adjustment; not equivalence tests','comparisons':[dict(candidate=a,reference=b,**paired(scores[a],scores[b])) for a,b in comparisons]}
    (build/'review_pairs.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    rows=list(csv.DictReader((root/'RELEASE_CANDIDATES/charts/dit_likeness.csv').open(encoding='utf-8-sig')))
    for r in rows:
        if r['tag']=='mixI_x':
            r['label']='mixI_x (combined 1-bit)'
        r.pop('hpsv3_pct_of_teacher',None)
        keys=['hpsv3_local_mean','hpsv3_delta_to_teacher','hpsv3_paired_se','hpsv3_ci95_low','hpsv3_ci95_high']
        r.update({k:'' for k in keys})
        if r['tag'] in scores:
            x=scores[r['tag']];p=paired(x,scores['teacher_bf16'])
            values=[float(np.mean(list(x.values()))),p['mean'],p['se'],*p['ci95']]
            r.update({k:f'{v:.6f}' for k,v in zip(keys,values)})
    chart_dir=out/'RELEASE_CANDIDATES/charts';chart_dir.mkdir(parents=True,exist_ok=True)
    with (chart_dir/'dit_likeness.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    charts(rows,chart_dir,out/'paper/figures')
    shutil.copyfile(chart_dir/'dit_quality_vs_size.png',out/'paper/figures/v2_fig19_hpsv3.png')
    if not args.skip_images:
        prompts=json.loads((root/'eval/eval_prompts.json').read_text(encoding='utf-8-sig'))
        tags=['teacher_bf16','turbo4_1024','tmixG_x_1024','tmixG_tuned_1024','tmixH_x_f8_1024','turboF_mixG_tuned']
        tasks=[(tag,p['id'],root/'eval'/tag/f"{p['id']:02d}_seed{p['seed']}.png") for tag in tags for p in prompts]
        per={t:{} for t in tags}
        with ThreadPoolExecutor(max_workers=3) as pool:
            for tag,pid,v in pool.map(detail_one,tasks):per[tag][pid]=v
        detail={'method':'RGB converted to [0,1]; PIL LANCZOS resize to 768x768; Rec.709 luma; population luma SD (contrast); RMS of 4-neighbour discrete Laplacian on interior pixels (edge energy). Primary images are native 1024 except the 25-step teacher, native 768. Same preprocessing for all tags. Edge energy is not retained meaningful detail.','models':{}}
        for tag in tags:
            means={k:float(np.mean([v[k] for v in per[tag].values()])) for k in ['luminance_mean','contrast','laplacian_rms']}
            ref='teacher_bf16' if tag=='turbo4_1024' else 'turbo4_1024'
            if tag=='teacher_bf16':ref=tag
            reference={k:float(np.mean([v[k] for v in per[ref].values()])) for k in means}
            detail['models'][tag]=dict(n=len(per[tag]),means=means,reference=ref,edge_energy_change_pct=100*(means['laplacian_rms']/reference['laplacian_rms']-1),contrast_change_pct=100*(means['contrast']/reference['contrast']-1),per_prompt=per[tag])
        (build/'review_detail_stats.json').write_text(json.dumps(detail,indent=2)+'\n',encoding='utf-8')
        print('Measured common-preprocessing statistics for',len(tasks),'saved images.')
    print('Wrote paired intervals, raw-score CSV, and two corrected charts.')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('Reporting failed; identifying locations omitted. Error type:',type(exc).__name__)
        raise SystemExit(1)
