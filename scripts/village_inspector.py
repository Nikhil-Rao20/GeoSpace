"""
village_inspector.py  —  Per-village class & subclass distribution analyzer.

Usage:
    python3 scripts/village_inspector.py --class utility
    python3 scripts/village_inspector.py --class building
    python3 scripts/village_inspector.py --class road
    python3 scripts/village_inspector.py --class waterbody
    python3 scripts/village_inspector.py --class bridge
    python3 scripts/village_inspector.py --class all        # every class at once

Options:
    --split   train | val | all   (default: all)
    --top N                        show top-N patches by fg pixel count
"""

import os
import sys
import json
import argparse
import rasterio
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ── ANSI colours ─────────────────────────────────────────────────────────────
R  = "\033[0m"          # reset
B  = "\033[1m"          # bold
DIM= "\033[2m"          # dim

FG = {
    "red":    "\033[91m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "blue":   "\033[94m",
    "purple": "\033[95m",
    "cyan":   "\033[96m",
    "white":  "\033[97m",
    "orange": "\033[38;5;208m",
    "teal":   "\033[38;5;43m",
    "gray":   "\033[90m",
}

BG = {
    "blue":   "\033[44m",
    "green":  "\033[42m",
    "yellow": "\033[43m",
    "purple": "\033[45m",
    "teal":   "\033[48;5;23m",
    "orange": "\033[48;5;130m",
    "gray":   "\033[100m",
}

def col(text, fg=None, bg=None, bold=False):
    s = ""
    if bold:  s += B
    if bg:    s += BG.get(bg, "")
    if fg:    s += FG.get(fg, "")
    return s + str(text) + R

def hbar(value, max_val, width=30, color="blue"):
    filled = int(round(width * value / max_val)) if max_val > 0 else 0
    bar    = "█" * filled + "░" * (width - filled)
    return col(bar, fg=color)

# ── Class configs ─────────────────────────────────────────────────────────────
CLASS_CONFIGS = {
    "building": {
        "channel":     0,
        "label":       "Building",
        "color":       "blue",
        "subcodes": {
            1: ("RCC",    "blue"),
            2: ("Tiled",  "cyan"),
            3: ("Tin",    "teal"),
            4: ("Others", "purple"),
        },
    },
    "road": {
        "channel":     1,
        "label":       "Road",
        "color":       "orange",
        "drop_codes":  {1},        # noise — 4 patches only
        "subcodes": {
            2: ("Type-2", "blue"),
            3: ("Type-3", "cyan"),
            4: ("Type-4", "orange"),
            5: ("Type-5", "yellow"),
        },
    },
    "waterbody": {
        "channel":     3,
        "label":       "Waterbody Polygon",
        "color":       "teal",
        "subcodes": {
            1: ("Type-1", "blue"),
            2: ("Type-2", "cyan"),
            3: ("Type-3 (→Other)", "gray"),
            4: ("Type-4", "teal"),
            5: ("Type-5 (→Other)", "gray"),
            6: ("Type-6 (→Other)", "purple"),
        },
    },
    "waterbody_line": {
        "channel":     4,
        "label":       "Waterbody Line",
        "color":       "cyan",
        "subcodes": {
            1: ("Line-Type-1", "blue"),
            2: ("Line-Type-2", "cyan"),
            3: ("Line-Type-3", "teal"),
        },
    },
    "utility": {
        "channel":     7,
        "label":       "Utility Polygon",
        "color":       "green",
        "subcodes": {
            1: ("Utility (code 1)", "green"),
        },
    },
    "bridge": {
        "channel":     8,
        "label":       "Bridge",
        "color":       "red",
        "subcodes": {
            1: ("Bridge (code 1)", "red"),
        },
    },
    "railway": {
        "channel":     2,
        "label":       "Railway",
        "color":       "gray",
        "subcodes": {},
    },
}

# ── Village name extractor ────────────────────────────────────────────────────
def get_village(fname):
    return fname.rsplit("_", 2)[0]

# ── Scanner ───────────────────────────────────────────────────────────────────
def scan(base_dir, patches, class_name):
    cfg       = CLASS_CONFIGS[class_name]
    ch        = cfg["channel"]
    drop      = cfg.get("drop_codes", set())
    mask_dir  = os.path.join(base_dir, "masks")

    # village → { "total_patches", "fg_patches", "fg_pixels",
    #              "subcodes": { code: {"patches","pixels"} } }
    villages = defaultdict(lambda: {
        "total_patches": 0,
        "fg_patches":    0,
        "fg_pixels":     0,
        "subcodes":      defaultdict(lambda: {"patches": 0, "pixels": 0}),
    })

    errors = 0
    for fname in tqdm(patches, desc=f"  Scanning {class_name}", ncols=80,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
        v = get_village(fname)
        villages[v]["total_patches"] += 1

        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            errors += 1
            continue

        try:
            with rasterio.open(mask_path) as src:
                ch_data = src.read(ch + 1)   # rasterio is 1-indexed
        except Exception:
            errors += 1
            continue

        unique = np.unique(ch_data)
        has_fg = any(v_ not in {0} | drop for v_ in unique)

        if has_fg:
            villages[v]["fg_patches"] += 1
            for code in unique:
                if code == 0 or code in drop:
                    continue
                px = int((ch_data == code).sum())
                villages[v]["fg_pixels"] += px
                villages[v]["subcodes"][code]["patches"] += 1
                villages[v]["subcodes"][code]["pixels"]  += px

    return dict(villages), errors


# ── Pretty printer ────────────────────────────────────────────────────────────
def print_header(title, color="blue"):
    w = 74
    print()
    print(col("═" * w, fg=color, bold=True))
    pad = (w - len(title) - 2) // 2
    print(col("║" + " " * pad + title + " " * (w - pad - len(title) - 2) + "║",
              fg=color, bold=True))
    print(col("═" * w, fg=color, bold=True))

def print_section(title, color="yellow"):
    print()
    print(col(f"  ── {title} ", fg=color, bold=True) +
          col("─" * max(0, 68 - len(title)), fg="gray"))


def print_village_table(villages, cfg, total_patches, split_name):
    color    = cfg["color"]
    subcodes = cfg.get("subcodes", {})

    # Sort by fg_patches descending
    sorted_v = sorted(villages.items(), key=lambda x: x[1]["fg_patches"], reverse=True)
    max_fg   = max((v["fg_patches"] for v in villages.values()), default=1)
    max_fp   = max((v["fg_pixels"]  for v in villages.values()), default=1)

    # Column widths
    v_w = min(48, max(len(k) for k in villages) + 2)

    print_section(f"Village Summary  [{split_name}]", color=color)

    hdr = (
        f"  {col('Village', fg='white', bold=True):<{v_w+10}}"
        f"  {col('Total Patches', fg='white', bold=True):>14}"
        f"  {col('FG Patches', fg='white', bold=True):>11}"
        f"  {col('FG %', fg='white', bold=True):>7}"
        f"  {col('FG Pixels', fg='white', bold=True):>12}"
        f"  {col('Bar', fg='white', bold=True)}"
    )
    print(hdr)
    print(col("  " + "─" * 72, fg="gray"))

    grand_total = 0
    grand_fg    = 0
    grand_px    = 0

    for vname, d in sorted_v:
        t  = d["total_patches"]
        fg = d["fg_patches"]
        px = d["fg_pixels"]
        pct = 100.0 * fg / t if t > 0 else 0.0

        grand_total += t
        grand_fg    += fg
        grand_px    += px

        bar_color = color if fg > 0 else "gray"
        pct_color = "green" if pct > 20 else ("yellow" if pct > 5 else ("orange" if pct > 0 else "gray"))

        print(
            f"  {col(vname[:v_w], fg=color):<{v_w+10}}"
            f"  {col(t, fg='white'):>14}"
            f"  {col(fg, fg='green' if fg>0 else 'gray'):>11}"
            f"  {col(f'{pct:.1f}%', fg=pct_color):>15}"
            f"  {col(f'{px:,}', fg='cyan'):>20}"
            f"  {hbar(fg, max_fg, width=22, color=bar_color)}"
        )

    print(col("  " + "─" * 72, fg="gray"))
    grand_pct = 100.0 * grand_fg / grand_total if grand_total > 0 else 0.0
    print(
        f"  {col('TOTAL', fg='yellow', bold=True):<{v_w+10}}"
        f"  {col(grand_total, fg='yellow', bold=True):>14}"
        f"  {col(grand_fg, fg='yellow', bold=True):>11}"
        f"  {col(f'{grand_pct:.1f}%', fg='yellow', bold=True):>15}"
        f"  {col(f'{grand_px:,}', fg='yellow', bold=True):>20}"
    )

    # ── Subcode breakdown per village ─────────────────────────────────────
    if not subcodes:
        print(col("\n  No subcodes defined for this class (binary only).", fg="gray"))
        return

    print_section("Per-Village Subclass Breakdown", color=color)

    # Gather all codes that appear
    all_codes = set()
    for d in villages.values():
        all_codes.update(d["subcodes"].keys())
    all_codes = sorted(all_codes)

    if not all_codes:
        print(col("  No subclass foreground found across any village.", fg="red"))
        return

    # Header row
    code_labels = []
    for code in all_codes:
        if code in subcodes:
            lbl, _ = subcodes[code]
            code_labels.append(f"Code {code}\n{lbl}")
        else:
            code_labels.append(f"Code {code}\n(unknown)")

    # Print table
    col_w = 14
    vname_w = min(46, max(len(k) for k in villages) + 2)

    header_top = f"  {'':<{vname_w}}"
    for code in all_codes:
        lbl = subcodes[code][0] if code in subcodes else f"code{code}"
        # truncate label
        header_top += f"  {col(f'Code {code}', fg='white', bold=True):>{col_w+10}}"
    print(header_top)

    header_sub = f"  {col('Village', fg='white', bold=True):<{vname_w+10}}"
    for code in all_codes:
        lbl, lc = subcodes[code] if code in subcodes else (f"code{code}", "gray")
        header_sub += f"  {col(lbl[:10], fg=lc):>{col_w+10}}"
    print(header_sub)
    print(col("  " + "─" * (vname_w + len(all_codes) * (col_w + 2) + 4), fg="gray"))

    code_totals = defaultdict(lambda: {"patches": 0, "pixels": 0})

    for vname, d in sorted_v:
        row = f"  {col(vname[:vname_w], fg=color):<{vname_w+10}}"
        for code in all_codes:
            sc  = d["subcodes"].get(code, {"patches": 0, "pixels": 0})
            pt  = sc["patches"]
            px  = sc["pixels"]
            code_totals[code]["patches"] += pt
            code_totals[code]["pixels"]  += px
            lc = subcodes[code][1] if code in subcodes else "gray"
            if pt > 0:
                row += f"  {col(f'{pt}p/{px//1000}Kpx', fg=lc):>{col_w+10}}"
            else:
                row += f"  {col('—', fg='gray'):>{col_w+10}}"
        print(row)

    print(col("  " + "─" * (vname_w + len(all_codes) * (col_w + 2) + 4), fg="gray"))
    total_row = f"  {col('TOTAL', fg='yellow', bold=True):<{vname_w+10}}"
    for code in all_codes:
        lc = subcodes[code][1] if code in subcodes else "gray"
        pt = code_totals[code]["patches"]
        px = code_totals[code]["pixels"]
        if pt > 0:
            total_row += f"  {col(f'{pt}p/{px//1000}Kpx', fg='yellow', bold=True):>{col_w+10}}"
        else:
            total_row += f"  {col('0', fg='gray'):>{col_w+10}}"
    print(total_row)

    # ── Per-code pixel share ──────────────────────────────────────────────
    print_section("Subclass Pixel Share (across all villages)", color=color)
    total_fg_px = sum(code_totals[c]["pixels"] for c in all_codes)
    max_code_px = max((code_totals[c]["pixels"] for c in all_codes), default=1)

    print(f"  {col('Subclass', fg='white', bold=True):<28}"
          f"  {col('Patches', fg='white', bold=True):>10}"
          f"  {col('Pixels', fg='white', bold=True):>14}"
          f"  {col('% of FG', fg='white', bold=True):>10}"
          f"  Bar")
    print(col("  " + "─" * 72, fg="gray"))

    for code in all_codes:
        lbl, lc = subcodes[code] if code in subcodes else (f"code {code}", "gray")
        pt  = code_totals[code]["patches"]
        px  = code_totals[code]["pixels"]
        pct = 100.0 * px / total_fg_px if total_fg_px > 0 else 0.0
        print(
            f"  {col(f'Code {code}: {lbl}', fg=lc):<38}"
            f"  {col(pt, fg='white'):>10}"
            f"  {col(f'{px:,}', fg='cyan'):>14}"
            f"  {col(f'{pct:.1f}%', fg='yellow'):>10}"
            f"  {hbar(px, max_code_px, width=24, color=lc)}"
        )


def print_class_summary(class_name, villages, cfg, errors):
    color = cfg["color"]
    label = cfg["label"]

    all_fg    = sum(d["fg_patches"]    for d in villages.values())
    all_total = sum(d["total_patches"] for d in villages.values())
    all_px    = sum(d["fg_pixels"]     for d in villages.values())
    pct       = 100.0 * all_fg / all_total if all_total > 0 else 0

    print_section("Quick Stats", color=color)
    print(f"  {col('Class:', fg='white', bold=True):<22} {col(label, fg=color, bold=True)}")
    print(f"  {col('Channel:', fg='white', bold=True):<22} {col(cfg['channel'], fg='cyan')}")
    print(f"  {col('Total patches scanned:', fg='white', bold=True):<22} {col(all_total, fg='white')}")
    print(f"  {col('Patches with FG:', fg='white', bold=True):<22} "
          f"{col(all_fg, fg='green' if all_fg>0 else 'red')}  "
          f"{col(f'({pct:.3f}%)', fg='yellow')}")
    print(f"  {col('Total FG pixels:', fg='white', bold=True):<22} {col(f'{all_px:,}', fg='cyan')}")
    print(f"  {col('Villages with FG:', fg='white', bold=True):<22} "
          f"{col(sum(1 for d in villages.values() if d['fg_patches']>0), fg='green')} "
          f"/ {col(len(villages), fg='white')}")
    if errors:
        print(f"  {col('Read errors:', fg='red', bold=True):<22} {col(errors, fg='red')}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-village class & subclass distribution inspector"
    )
    parser.add_argument(
        "--class", dest="cls",
        choices=list(CLASS_CONFIGS.keys()) + ["all"],
        required=True,
        help="Which class to inspect"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "all"], default="all",
        help="Which split to scan (default: all)"
    )
    parser.add_argument(
        "--base-dir", default=None,
        help="Override base_dir from config"
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from config import Config
        cfg_obj  = Config()
        base_dir = args.base_dir or cfg_obj.base_dir
    except ImportError:
        base_dir = args.base_dir or "."

    train_file = os.path.join(base_dir, "train.txt")
    val_file   = os.path.join(base_dir, "val.txt")

    def load_split(path):
        with open(path) as f:
            return [ln.strip() for ln in f if ln.strip()]

    if args.split == "train":
        patches = load_split(train_file)
        split_label = "TRAIN"
    elif args.split == "val":
        patches = load_split(val_file)
        split_label = "VAL"
    else:
        patches = load_split(train_file) + load_split(val_file)
        split_label = "TRAIN + VAL"

    # ── Which classes to run ──────────────────────────────────────────────
    classes_to_run = list(CLASS_CONFIGS.keys()) if args.cls == "all" else [args.cls]

    for class_name in classes_to_run:
        cfg   = CLASS_CONFIGS[class_name]
        color = cfg["color"]

        title = f"  SVAMITVA Village Inspector  —  {cfg['label'].upper()}  [{split_label}]  "
        print_header(title, color=color)

        villages, errors = scan(base_dir, patches, class_name)
        print_class_summary(class_name, villages, cfg, errors)
        print_village_table(villages, cfg, len(patches), split_label)

    print()
    print(col("  Done.", fg="green", bold=True))
    print()


if __name__ == "__main__":
    main()