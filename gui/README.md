# Transformer-MT Desktop GUI

Cross-platform desktop app that wraps the published HuggingFace
checkpoints (`euswbnix/transformer-wmt14-en{fr,de}-{base,big}`)
behind a click-to-translate interface. No terminal required.

## For end users (when packaged)

1. Download the installer for your OS from the GitHub Releases page
2. Run it; on first launch the app prompts to download a 240 MB
   en→fr Base model from HuggingFace Hub
3. Type/paste English text on the left, click **Translate**
4. Switch the dropdown to en→de, Big, or fine-tuned variants on demand;
   each downloads the first time you use it

Default model = en→fr Base v1.1 (60M params, FP32, ~240 MB,
sacrebleu test 35.31). Other variants are lazy-loaded per dropdown.

Long inputs are split into sentences with `pysbd` and translated
sentence-by-sentence, then reassembled preserving paragraph breaks —
this is the standard production-MT pattern and keeps inputs within
the model's trained 256-token limit. Output sentences with unusual
length ratios are flagged in the status bar as possible
mistranslations.

## For developers

```bash
# from project root (so src/ is importable)
pip install -r requirements.txt
pip install -r gui/requirements_gui.txt
python -m gui
```

### Layout

```
gui/
├── main.py            # entry point
├── translator.py      # main window + threaded workers
├── inference.py       # checkpoint load + sentence-segmented translate
├── downloader.py      # huggingface_hub wrapper, ~/.cache/transformer-mt/
├── model_registry.py  # available models; FT extension hooked at runtime
└── requirements_gui.txt
```

### Adding a new model

Append a `ModelEntry` to `BASE_MODELS` in `model_registry.py`. Files
listed in `files=[...]` are fetched from `hf_repo` on first use.

### Fine-tuned (FT) variants

The Advanced menu's *Show fine-tuned models* toggle imports
`gui_ft_extension.ft_models`. That package lives in the
[Machine-Translation-FT](https://github.com/Euswbnix/Machine-Translation-FT)
repo and is not bundled by default; users who install both repos
side-by-side get the extra dropdown entries automatically.

### Building a standalone binary

```bash
pip install pyinstaller>=6.0
# placeholder spec — to be added with platform-specific tweaks
pyinstaller --name TransformerMT --windowed --onefile gui/main.py
```

PyInstaller specs per OS will be added under `gui/build/` once
release tooling is in place.
