"""
Генератор .ipynb-ноутбуков для практической части НИР.

Запускать так:
    python scripts/build_notebooks.py

После запуска в notebooks/ появятся 5 файлов:
    01_setup_and_dataset.ipynb
    02_deepseek_ocr.ipynb
    03_mplug_docowl.ipynb
    04_olmocr.ipynb
    05_monkeyocr.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) or [""],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True) or [""],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save(name: str, cells: list[dict]):
    path = NOTEBOOKS_DIR / name
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"wrote {path}")


# -------------------- общий пролог Colab --------------------

COLAB_PROLOG = '''\
# === Colab / Kaggle bootstrap =================================================
# В Colab клонируем репозиторий проекта (предполагается, что код выложен
# в GitHub) и переходим в его корень. Для локального запуска просто
# проверьте, что текущая рабочая директория — корень ocr_eval/.
import os, sys, subprocess, pathlib

REPO_NAME = "ocr_eval"
if not pathlib.Path(REPO_NAME).exists():
    # Замените URL на ваш форк, если работаете в Colab
    # !git clone https://github.com/<your-username>/ocr_eval.git
    pass

if pathlib.Path(REPO_NAME).exists():
    os.chdir(REPO_NAME)
sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
print("CWD =", os.getcwd())
'''


# ============================================================================
# 01_setup_and_dataset.ipynb
# ============================================================================
nb01 = [
    md("# 01 · Подготовка среды и загрузка OmniDocBench\n\n"
       "Этот ноутбук готовит Colab-среду к экспериментам:\n"
       "1. Устанавливает зависимости из `requirements.txt`.\n"
       "2. Скачивает датасет **OmniDocBench v1.6** "
       "(`opendatalab/OmniDocBench` на HuggingFace).\n"
       "3. Загружает разметку, фильтрует страницы типа "
       "`academic_literature` (научные статьи arXiv-стиля).\n"
       "4. Показывает превью первой страницы и её ground-truth."),
    md("## 1. Установка зависимостей"),
    code("# В Colab можно использовать GPU runtime → T4/L4/A100\n"
         "!nvidia-smi || echo 'GPU не подключена'"),
    code("!pip install -q -r requirements.txt"),
    md("## 2. Bootstrap путей"),
    code(COLAB_PROLOG),
    md("## 3. Скачивание OmniDocBench"),
    code("from src.dataset_loader import download_omnidocbench, load_omnidocbench, iter_pages\n"
         "\n"
         "DATA_ROOT = 'data/OmniDocBench'\n"
         "download_omnidocbench(DATA_ROOT, source='huggingface')\n"
         "print('OK, датасет в', DATA_ROOT)"),
    md("## 4. Фильтр научных страниц и быстрый превью"),
    code("# Берём 100 случайных страниц-научных публикаций для экспериментов\n"
         "items = load_omnidocbench(\n"
         "    root=DATA_ROOT,\n"
         "    page_types=['academic_literature'],\n"
         "    languages=['english'],\n"
         "    subset_size=100,\n"
         "    seed=42,\n"
         ")\n"
         "print(f'Отобрано страниц: {len(items)}')\n"
         "items[0].to_dict() if items else 'empty'"),
    code("import matplotlib.pyplot as plt\n"
         "from PIL import Image\n"
         "from pathlib import Path\n"
         "\n"
         "first = items[0]\n"
         "img = Image.open(Path(DATA_ROOT) / first.image_path).convert('RGB')\n"
         "fig, ax = plt.subplots(figsize=(8, 11))\n"
         "ax.imshow(img); ax.axis('off')\n"
         "ax.set_title(f'{first.page_id}  ·  {first.page_type}')\n"
         "plt.show()"),
    md("## 5. Сохраняем список выбранных страниц\n\n"
       "Все 4 ноутбука инференса используют один и тот же subset — "
       "так результаты сравнимы."),
    code("import json, pathlib\n"
         "subset_path = pathlib.Path('data/subset.json')\n"
         "subset_path.parent.mkdir(parents=True, exist_ok=True)\n"
         "subset_path.write_text(\n"
         "    json.dumps([x.to_dict() for x in items], ensure_ascii=False),\n"
         "    encoding='utf-8',\n"
         ")\n"
         "print('сохранено:', subset_path, subset_path.stat().st_size, 'байт')"),
    md("---\n*Готово.* Дальше — `02_deepseek_ocr.ipynb`."),
]


# ============================================================================
# Шаблон ноутбука инференса для модели (с минимальными отличиями)
# ============================================================================
def inference_intro(title: str, descr: str) -> list[dict]:
    return [
        md(f"# {title}\n\n{descr}\n\n"
           "Ноутбук берёт subset, сохранённый в `data/subset.json` "
           "ноутбуком `01_setup_and_dataset.ipynb`, и прогоняет на нём модель. "
           "Результаты записываются в `results/<model>/predictions.jsonl`."),
        md("## 1. Установка"),
        code("!pip install -q -r requirements.txt"),
        code(COLAB_PROLOG),
    ]


def common_load_subset() -> dict:
    return code(
        "import json, pathlib\n"
        "from src.dataset_loader import GroundTruth\n"
        "\n"
        "DATA_ROOT = pathlib.Path('data/OmniDocBench')\n"
        "subset = [GroundTruth(**rec) for rec in json.loads(\n"
        "    pathlib.Path('data/subset.json').read_text(encoding='utf-8'))]\n"
        "print(f'subset: {len(subset)} страниц')"
    )


# ============================================================================
# 02_deepseek_ocr.ipynb
# ============================================================================
nb02 = [
    *inference_intro(
        "02 · DeepSeek-OCR",
        "**DeepSeek-OCR** — открытая мульти-модальная модель (≈3B параметров) "
        "от DeepSeek-AI; поддерживает grounding-промпт и markdown-вывод. "
        "Чекпоинт: [`deepseek-ai/DeepSeek-OCR`](https://huggingface.co/deepseek-ai/DeepSeek-OCR)."
    ),
    md("## 2. Конфиг и загрузка модели"),
    code("from src.utils import load_config, JsonlWriter, Timer, cuda_free, gpu_info, already_processed_ids\n"
         "from src.io_records import PredictionRecord\n"
         "\n"
         "cfg = load_config('configs/deepseek_ocr.yaml')\n"
         "print(gpu_info())\n"
         "cfg"),
    code("import torch\n"
         "from transformers import AutoModel, AutoTokenizer\n"
         "\n"
         "MODEL_REPO = cfg['model']['hf_repo']\n"
         "tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)\n"
         "model = AutoModel.from_pretrained(\n"
         "    MODEL_REPO,\n"
         "    trust_remote_code=True,\n"
         "    torch_dtype=getattr(torch, cfg['model']['torch_dtype']),\n"
         "    device_map=cfg['model']['device_map'],\n"
         ").eval()\n"
         "print(gpu_info())"),
    md("## 3. Загружаем общий subset"),
    common_load_subset(),
    md("## 4. Инференс\n\nDeepSeek-OCR использует свой удобный метод `model.infer(...)`."),
    code("import os, time, traceback\n"
         "from pathlib import Path\n"
         "from PIL import Image\n"
         "\n"
         "out_dir = Path(cfg['output']['results_dir'])\n"
         "out_dir.mkdir(parents=True, exist_ok=True)\n"
         "out_path = out_dir / 'predictions.jsonl'\n"
         "done = already_processed_ids(out_path)\n"
         "print(f'уже обработано: {len(done)} / {len(subset)}')\n"
         "\n"
         "with JsonlWriter(out_path) as w:\n"
         "    for gt in subset:\n"
         "        if gt.page_id in done:\n"
         "            continue\n"
         "        img_path = DATA_ROOT / gt.image_path\n"
         "        if not img_path.exists():\n"
         "            continue\n"
         "        rec = PredictionRecord(page_id=gt.page_id, model='deepseek_ocr')\n"
         "        try:\n"
         "            with Timer('infer') as t:\n"
         "                # API DeepSeek-OCR (см. README модели):\n"
         "                # model.infer(tokenizer, prompt, image_file, output_path,\n"
         "                #             base_size, image_size, crop_mode, save_results, test_compress)\n"
         "                tmp_out = out_dir / f'{gt.page_id}'\n"
         "                tmp_out.mkdir(parents=True, exist_ok=True)\n"
         "                result_md = model.infer(\n"
         "                    tokenizer,\n"
         "                    prompt=cfg['inference']['prompt'],\n"
         "                    image_file=str(img_path),\n"
         "                    output_path=str(tmp_out),\n"
         "                    base_size=cfg['inference']['base_size'],\n"
         "                    image_size=cfg['inference']['image_size'],\n"
         "                    crop_mode=cfg['inference']['crop_mode'],\n"
         "                    save_results=False,\n"
         "                    test_compress=cfg['inference']['test_compress'],\n"
         "                )\n"
         "            rec.full_text = result_md if isinstance(result_md, str) else str(result_md)\n"
         "            rec.raw_output = rec.full_text\n"
         "            rec.inference_time_s = t.elapsed\n"
         "        except Exception as e:\n"
         "            rec.error = f'{type(e).__name__}: {e}'\n"
         "            traceback.print_exc()\n"
         "        w.write(rec.to_dict())\n"
         "print('готово →', out_path)"),
    md("## 5. Превью одного результата"),
    code("from src.utils import read_jsonl\n"
         "preds = read_jsonl(out_path)\n"
         "print(len(preds), 'записей')\n"
         "print(preds[0]['full_text'][:1000])"),
    md("## 6. Освободить GPU\nПосле выгрузки можно запускать следующую модель в этом же runtime."),
    code("del model; cuda_free(); print(gpu_info())"),
]


# ============================================================================
# 03_mplug_docowl.ipynb
# ============================================================================
nb03 = [
    *inference_intro(
        "03 · mPLUG-DocOwl 2",
        "**mPLUG-DocOwl 2** — модель Alibaba для понимания документов с "
        "shape-adaptive cropping. Чекпоинт: "
        "[`mPLUG/DocOwl2`](https://huggingface.co/mPLUG/DocOwl2)."
    ),
    md("## 2. Конфиг и загрузка модели"),
    code("from src.utils import load_config, JsonlWriter, Timer, cuda_free, gpu_info, already_processed_ids\n"
         "from src.io_records import PredictionRecord\n"
         "\n"
         "cfg = load_config('configs/mplug_docowl.yaml')\n"
         "cfg"),
    code("import torch\n"
         "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
         "\n"
         "MODEL_REPO = cfg['model']['hf_repo']\n"
         "tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)\n"
         "model = AutoModelForCausalLM.from_pretrained(\n"
         "    MODEL_REPO,\n"
         "    trust_remote_code=True,\n"
         "    torch_dtype=getattr(torch, cfg['model']['torch_dtype']),\n"
         "    device_map=cfg['model']['device_map'],\n"
         ").eval()\n"
         "print(gpu_info())"),
    md("## 3. Subset и helper"),
    common_load_subset(),
    code("# DocOwl ожидает PIL.Image + текстовый промпт; в зависимости от версии\n"
         "# либо есть метод model.chat(images=[img], query=prompt), либо нужно\n"
         "# собрать messages вручную. Универсальный путь — через preprocessor:\n"
         "from transformers import AutoProcessor\n"
         "try:\n"
         "    processor = AutoProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)\n"
         "except Exception:\n"
         "    processor = None\n"
         "    print('processor не нужен — используем model.chat() напрямую')\n"),
    md("## 4. Инференс"),
    code("import traceback\n"
         "from pathlib import Path\n"
         "from PIL import Image\n"
         "\n"
         "out_path = Path(cfg['output']['results_dir']) / 'predictions.jsonl'\n"
         "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
         "done = already_processed_ids(out_path)\n"
         "\n"
         "PROMPT = cfg['inference']['prompt']\n"
         "MAX_NEW = cfg['inference']['max_new_tokens']\n"
         "\n"
         "with JsonlWriter(out_path) as w:\n"
         "    for gt in subset:\n"
         "        if gt.page_id in done: continue\n"
         "        img_path = DATA_ROOT / gt.image_path\n"
         "        if not img_path.exists(): continue\n"
         "        rec = PredictionRecord(page_id=gt.page_id, model='mplug_docowl')\n"
         "        try:\n"
         "            img = Image.open(img_path).convert('RGB')\n"
         "            with Timer('infer') as t:\n"
         "                if hasattr(model, 'chat'):\n"
         "                    out = model.chat(image=img, msgs=[{'role':'user','content': PROMPT}],\n"
         "                                     tokenizer=tokenizer, sampling=False, max_new_tokens=MAX_NEW)\n"
         "                else:\n"
         "                    inputs = processor(images=img, text=PROMPT, return_tensors='pt').to(model.device)\n"
         "                    with torch.no_grad():\n"
         "                        gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)\n"
         "                    out = processor.batch_decode(gen, skip_special_tokens=True)[0]\n"
         "            rec.full_text = out if isinstance(out, str) else str(out)\n"
         "            rec.raw_output = rec.full_text\n"
         "            rec.inference_time_s = t.elapsed\n"
         "        except Exception as e:\n"
         "            rec.error = f'{type(e).__name__}: {e}'\n"
         "            traceback.print_exc()\n"
         "        w.write(rec.to_dict())\n"
         "print('готово →', out_path)"),
    md("## 5. Освободить GPU"),
    code("del model; cuda_free(); print(gpu_info())"),
]


# ============================================================================
# 04_olmocr.ipynb
# ============================================================================
nb04 = [
    *inference_intro(
        "04 · olmOCR (AllenAI)",
        "**olmOCR** — пайплайн от AllenAI поверх Qwen2-VL-7B-Instruct, "
        "дообученный на ~250K страницах. Чекпоинт: "
        "[`allenai/olmOCR-7B-0225-preview`](https://huggingface.co/allenai/olmOCR-7B-0225-preview)."
    ),
    md("## 2. Системные зависимости (poppler нужен для anchor-текста)"),
    code("!apt-get -qq install -y poppler-utils ttf-mscorefonts-installer 2>&1 | tail -1\n"
         "!pip install -q olmocr"),
    code("from src.utils import load_config, JsonlWriter, Timer, cuda_free, gpu_info, already_processed_ids\n"
         "from src.io_records import PredictionRecord\n"
         "cfg = load_config('configs/olmocr.yaml')\n"
         "cfg"),
    md("## 3. Загрузка модели Qwen2-VL"),
    code("import torch\n"
         "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor\n"
         "\n"
         "MODEL_REPO = cfg['model']['hf_repo']\n"
         "processor = AutoProcessor.from_pretrained(MODEL_REPO)\n"
         "model = Qwen2VLForConditionalGeneration.from_pretrained(\n"
         "    MODEL_REPO,\n"
         "    torch_dtype=getattr(torch, cfg['model']['torch_dtype']),\n"
         "    device_map=cfg['model']['device_map'],\n"
         ").eval()\n"
         "print(gpu_info())"),
    md("## 4. Inference\n\n"
       "olmOCR работает по схеме: PDF → рендер страницы + anchor-текст из poppler "
       "→ многостраничный prompt → Qwen2-VL. Для OmniDocBench у нас уже есть "
       "PNG-страницы, поэтому используем упрощённый путь без anchor-текста "
       "(в отчёте отметить, что это даёт небольшой проигрыш по сравнению с "
       "полным пайплайном)."),
    common_load_subset(),
    code("import traceback, base64, io\n"
         "from pathlib import Path\n"
         "from PIL import Image\n"
         "\n"
         "out_path = Path(cfg['output']['results_dir']) / 'predictions.jsonl'\n"
         "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
         "done = already_processed_ids(out_path)\n"
         "\n"
         "MAX_NEW = cfg['inference']['max_new_tokens']\n"
         "TEMP    = cfg['inference']['temperature']\n"
         "\n"
         "OLMOCR_PROMPT = (\n"
         "    'Below is the image of one page of a document. Just return the plain text '\n"
         "    'representation of this document as if you were reading it naturally. '\n"
         "    'Convert equations to LaTeX and tables to HTML. Do not hallucinate.'\n"
         ")\n"
         "\n"
         "with JsonlWriter(out_path) as w:\n"
         "    for gt in subset:\n"
         "        if gt.page_id in done: continue\n"
         "        img_path = DATA_ROOT / gt.image_path\n"
         "        if not img_path.exists(): continue\n"
         "        rec = PredictionRecord(page_id=gt.page_id, model='olmocr')\n"
         "        try:\n"
         "            img = Image.open(img_path).convert('RGB')\n"
         "            messages = [{\n"
         "                'role': 'user',\n"
         "                'content': [\n"
         "                    {'type': 'image'},\n"
         "                    {'type': 'text', 'text': OLMOCR_PROMPT},\n"
         "                ],\n"
         "            }]\n"
         "            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
         "            inputs = processor(text=[text], images=[img], padding=True, return_tensors='pt').to(model.device)\n"
         "            with Timer('infer') as t, torch.no_grad():\n"
         "                gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=TEMP > 0,\n"
         "                                     temperature=TEMP if TEMP > 0 else 1.0)\n"
         "            out = processor.batch_decode(\n"
         "                gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]\n"
         "            rec.full_text = out\n"
         "            rec.raw_output = out\n"
         "            rec.inference_time_s = t.elapsed\n"
         "        except Exception as e:\n"
         "            rec.error = f'{type(e).__name__}: {e}'\n"
         "            traceback.print_exc()\n"
         "        w.write(rec.to_dict())\n"
         "print('готово →', out_path)"),
    md("## 5. Освободить GPU"),
    code("del model; cuda_free(); print(gpu_info())"),
]


# ============================================================================
# 05_monkeyocr.ipynb
# ============================================================================
nb05 = [
    *inference_intro(
        "05 · MonkeyOCR",
        "**MonkeyOCR** — модель с парадигмой **Structure → Recognition → Relation**. "
        "Чекпоинт: [`echo840/MonkeyOCR`](https://huggingface.co/echo840/MonkeyOCR)."
    ),
    md("## 2. Установка"),
    code("!pip install -q -r requirements.txt"),
    code(COLAB_PROLOG),
    md("## 3. Конфиг и модель"),
    code("from src.utils import load_config, JsonlWriter, Timer, cuda_free, gpu_info, already_processed_ids\n"
         "from src.io_records import PredictionRecord\n"
         "cfg = load_config('configs/monkeyocr.yaml')\n"
         "cfg"),
    code("import torch\n"
         "from transformers import AutoModel, AutoTokenizer\n"
         "MODEL_REPO = cfg['model']['hf_repo']\n"
         "tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)\n"
         "model = AutoModel.from_pretrained(\n"
         "    MODEL_REPO,\n"
         "    trust_remote_code=True,\n"
         "    torch_dtype=getattr(torch, cfg['model']['torch_dtype']),\n"
         "    device_map=cfg['model']['device_map'],\n"
         ").eval()\n"
         "print(gpu_info())"),
    md("## 4. Subset"),
    common_load_subset(),
    md("## 5. Инференс\n\n"
       "MonkeyOCR имеет встроенный метод `model.chat_full_page(...)` "
       "(см. README модели). Если интерфейс изменится — заменить на актуальный."),
    code("import traceback\n"
         "from pathlib import Path\n"
         "from PIL import Image\n"
         "\n"
         "out_path = Path(cfg['output']['results_dir']) / 'predictions.jsonl'\n"
         "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
         "done = already_processed_ids(out_path)\n"
         "\n"
         "with JsonlWriter(out_path) as w:\n"
         "    for gt in subset:\n"
         "        if gt.page_id in done: continue\n"
         "        img_path = DATA_ROOT / gt.image_path\n"
         "        if not img_path.exists(): continue\n"
         "        rec = PredictionRecord(page_id=gt.page_id, model='monkeyocr')\n"
         "        try:\n"
         "            img = Image.open(img_path).convert('RGB')\n"
         "            with Timer('infer') as t, torch.no_grad():\n"
         "                if hasattr(model, 'chat_full_page'):\n"
         "                    out = model.chat_full_page(tokenizer, img,\n"
         "                                               max_new_tokens=cfg['inference']['max_new_tokens'])\n"
         "                else:\n"
         "                    out = model.chat(tokenizer, img,\n"
         "                                     query='Convert this page to markdown',\n"
         "                                     max_new_tokens=cfg['inference']['max_new_tokens'])\n"
         "            rec.full_text = out if isinstance(out, str) else str(out)\n"
         "            rec.raw_output = rec.full_text\n"
         "            rec.inference_time_s = t.elapsed\n"
         "        except Exception as e:\n"
         "            rec.error = f'{type(e).__name__}: {e}'\n"
         "            traceback.print_exc()\n"
         "        w.write(rec.to_dict())\n"
         "print('готово →', out_path)"),
    md("## 6. Освободить GPU"),
    code("del model; cuda_free(); print(gpu_info())"),
]


# -------------------- запись --------------------
save("01_setup_and_dataset.ipynb", nb01)
save("02_deepseek_ocr.ipynb", nb02)
save("03_mplug_docowl.ipynb", nb03)
save("04_olmocr.ipynb", nb04)
save("05_monkeyocr.ipynb", nb05)
