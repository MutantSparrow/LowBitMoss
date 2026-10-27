"""Labeled contact sheet: rows = prompts (matched by file name), columns = tags. Splits into several PNGs if wider/taller than the limits.

  python -I sheet.py --dirs eval/teacher_bf16 eval/mixF_ptq eval/mixG_x --out eval/sheet.png [--thumb 256] [--rows 8]
Column labels are the directory names. Row label = file stem (NN_seedSSSS). One output PNG per --rows prompts (sheet_1.png, ...).
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LABEL_H, ROW_LABEL_W = 22, 110


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thumb", type=int, default=256)
    ap.add_argument("--rows", type=int, default=8, help="prompts per sheet image")
    a = ap.parse_args()

    dirs = [Path(d) for d in a.dirs]
    names = sorted(set.intersection(*[{p.stem for p in d.glob("*.png")} for d in dirs]) or set.union(*[{p.stem for p in d.glob("*.png")} for d in dirs]))
    width = ROW_LABEL_W + a.thumb * len(dirs)
    if width > 3000:
        raise SystemExit(f"sheet would be {width}px wide; use a smaller --thumb or fewer dirs")
    font = ImageFont.load_default()
    out = Path(a.out)
    chunks = [names[i:i + a.rows] for i in range(0, len(names), a.rows)]
    for ci, chunk in enumerate(chunks):
        sheet = Image.new("RGB", (width, LABEL_H + a.thumb * len(chunk)), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for c, p in enumerate(dirs):
            d.text((ROW_LABEL_W + c * a.thumb + 4, 6), p.name, fill=(255, 255, 255), font=font)
        for r, n in enumerate(chunk):
            y = LABEL_H + r * a.thumb
            d.text((4, y + a.thumb // 2), n, fill=(255, 255, 255), font=font)
            for c, p in enumerate(dirs):
                f = p / f"{n}.png"
                if f.exists():
                    im = Image.open(f).convert("RGB")
                    im.thumbnail((a.thumb, a.thumb), Image.LANCZOS)
                    sheet.paste(im, (ROW_LABEL_W + c * a.thumb, y))
        target = out if len(chunks) == 1 else out.with_name(f"{out.stem}_{ci + 1}{out.suffix}")
        sheet.save(target)
        print(target, sheet.size)


if __name__ == "__main__":
    main()
