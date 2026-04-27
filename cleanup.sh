#!/usr/bin/env bash
# Удаляет файлы legacy-итерации проекта (mPlug-DocOwl, MonkeyOCR, разрозненные ноутбуки).
# Запустить один раз перед коммитом в репозиторий.
set -e
cd "$(dirname "$0")"

rm -f configs/mplug_docowl.yaml \
      configs/monkeyocr.yaml \
      notebooks/01_setup_and_dataset.ipynb \
      notebooks/02_deepseek_ocr.ipynb \
      notebooks/03_mplug_docowl.ipynb \
      notebooks/04_olmocr.ipynb \
      notebooks/05_monkeyocr.ipynb \
      scripts/build_notebooks.py

echo "cleanup done."
echo "теперь актуальны только:"
echo "  configs/{deepseek_ocr,olmocr,lightonocr}.yaml"
echo "  notebooks/00_kaggle_full_pipeline.ipynb"
