#!/bin/bash
########### Training & Validation Dataset Download Script (CNNDet) ###########
# Download and unzip the synthetic training dataset from CNNDet
# Reference: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf
# Data source: https://github.com/peterwang512/CNNDetection

# Preflight: check required tools. 7z is needed to extract .7z.NNN split
# archives, unzip for the inner .zip file.
_missing=0
for tool in wget 7z unzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not found in PATH."
        case "$tool" in
            7z)
                echo "Install inside the active conda env:"
                echo "    conda install -c conda-forge p7zip"
                echo "(Alternatively: apt install p7zip-full  /  dnf install p7zip)"
                ;;
            *)
                echo "Install inside the active conda env:"
                echo "    conda install -c conda-forge $tool"
                echo "(Alternatively: apt install $tool  /  dnf install $tool)"
                ;;
        esac
        _missing=1
    fi
done
[ "$_missing" = 1 ] && exit 1

set -e

ROOT="${ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}"
BASE="https://huggingface.co/datasets/sywang/CNNDetection/resolve/main"

mkdir -p "$ROOT/train" "$ROOT/val"
cd "$ROOT"

has_files() {
    [ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]
}

ensure_downloaded() {
    file="$1"

    if [ -s "$file" ]; then
        echo "Checking existing file and resuming only if incomplete: $file"
    else
        echo "Downloading: $file"
    fi
    wget -c "$BASE/$file"
}

# Keep existing extracted data; if an archive is present, allow extraction to resume.
train_archives_present=0
for archive in "$ROOT"/progan_train.7z.00?; do
    if [ -e "$archive" ]; then
        train_archives_present=1
        break
    fi
done

if has_files "$ROOT/train" && [ ! -e "$ROOT/progan_train.zip" ] \
    && [ "$train_archives_present" = 0 ]; then
    echo "Training data already exists; skipping."
else
    for part in 001 002 003 004 005 006 007; do
        ensure_downloaded "progan_train.7z.$part"
    done

    if [ ! -s "$ROOT/progan_train.zip" ]; then
        7z x "$ROOT/progan_train.7z.001" "-o$ROOT"
    fi

    unzip -o "$ROOT/progan_train.zip" -d "$ROOT/train"
    rm -f "$ROOT"/progan_train.7z.00? "$ROOT/progan_train.zip"
fi

if has_files "$ROOT/val" && [ ! -e "$ROOT/progan_val.zip" ]; then
    echo "Validation data already exists; skipping."
else
    ensure_downloaded "progan_val.zip"
    unzip -o "$ROOT/progan_val.zip" -d "$ROOT/val"
    rm -f "$ROOT/progan_val.zip"
fi
