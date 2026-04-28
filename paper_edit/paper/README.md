# Paper

LaTeX source for the paper. Compiles with stock `pdflatex`/`bibtex` or
on Overleaf.

## Build

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or upload the `paper/` directory to Overleaf as a new project.

## Layout

```
paper/
├── main.tex                  # top-level, includes all sections
├── references.bib            # bibliography
├── sections/
│   ├── abstract.tex
│   ├── 01_introduction.tex   # done
│   ├── 02_related_work.tex   # done (stub-quality, expand as needed)
│   ├── 03_setup.tex          # placeholder
│   ├── 04_enfr_v1.tex        # placeholder
│   ├── 05_enfr_v2.tex        # placeholder
│   ├── 06_ende.tex           # placeholder
│   ├── 07_sft.tex            # done (full draft)
│   ├── 08_methodology_note.tex   # placeholder
│   └── 09_conclusion.tex     # placeholder
└── figures/                  # PNGs (generate with scripts in main repos)
    └── sft_curves.png        # from Machine-Translation-SFT/scripts/plot_sft_curves.py
```

## Style

Currently using stock `article` class with single-column layout for
draftability. Switch to `\twocolumn` in main.tex once content is
stable, or swap in `acl2024.sty` for ACL/EMNLP submission.

For arXiv-only release: keep stock article class; add CC-BY note.

## TODO

- [ ] Draft §3 setup (port from main `Machine_translation` README)
- [ ] Draft §4 en-fr v1.1 (Base 35.31 + Big 35.87 success cases)
- [ ] Draft §5 en-fr v2 (Base/Big both fail on noisy data)
- [ ] Draft §6 en-de (Base > Big at 4.5M scale; same pattern)
- [ ] Draft §8 methodology note (spike-guard rate not a clean noise meter)
- [ ] Draft §9 conclusion
- [ ] Generate figures: training curves for v1/v2/en-de, 2x2 results bar plot
- [ ] Final pass: number tables, cross-references, citation completeness
