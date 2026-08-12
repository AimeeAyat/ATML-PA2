# ATML-PA2

A collection of notebooks, scripts, and outputs for Assignment 2 of the Advanced Topics in Machine Learning (ATML) course — focused on domain generalization and domain adaptation experiments and visualizations.

## Overview

This repository contains experiments, visualizations, and analysis for several domain adaptation / domain generalization approaches (e.g., DANN, DAN, CDAN, ERM, GroupDRO, IRM) using Jupyter notebooks and helper scripts. It includes results, visualization assets, and scripts to analyze and compare experimental runs.

Key components:
- Notebooks for model training, evaluation, and visualization
- Results and output directories containing metrics and figures
- Analysis scripts to parse and summarize results from DomainBed and GroupDRO runs

## Repository structure

Top-level directories and important files:
- CDAN_visualizations/         — visualizations for CDAN experiments
- Clip Results/                — CLIP-related results and outputs
- DANN_visualizations/         — visualizations for DANN experiments
- DAN_visualizations/          — visualizations for DAN experiments
- DG_NoteBooks/                — domain generalization notebooks
- DG_OUTPUTs/                  — saved outputs from DG experiments
- ERM domainbed/               — ERM experiments configured for DomainBed
- baseline_souce_only/         — baseline (source-only) experiments
- clip notebooks/              — notebooks using CLIP
- concept_shift/               — concept shift experiments/notebooks
- irm results/                 — IRM experiment outputs
- pseudo visulization/         — pseudo-label visualizations
- sam notebook/                — SAM-related notebook(s)
- task_1/                      — Task 1 notebooks and materials
- analyze_domainbed_results.py — script to analyze DomainBed result files
- analyze_groupdro_results.py  — script to analyze GroupDRO result files
- .gitignore, .gitattributes

(Directories may contain notebooks (.ipynb), dataset references, and image assets.)

## Dependencies

These projects were developed in Python and with Jupyter Notebooks. Exact package versions may vary between experiments. Example dependencies:

- Python 3.8+
- jupyterlab or notebook
- numpy, pandas, matplotlib, seaborn
- torch (PyTorch) and torchvision
- domainbed (if using DomainBed experiments)
- scikit-learn
- tqdm
- scikit-image (if image processing used)
- CLIP (openai/CLIP) for CLIP-based notebooks (optional)

Recommend using a virtual environment (conda) and a requirements file. Example commands:

Using conda:
```bash
conda create -n atml-p2 python=3.9
conda activate atml-p2
pip install -r requirements.txt
```

If you don't have a requirements.txt, create one with the libraries above, or run:
```bash
pip install jupyterlab numpy pandas matplotlib seaborn torch torchvision scikit-learn tqdm
```

And install DomainBed and CLIP as needed:
```bash
pip install domainbed
pip install git+https://github.com/openai/CLIP.git
```

## How to run

1. Clone the repository:
```bash
git clone https://github.com/rabiaaslam92/ATML-PA2.git
cd ATML-PA2
```

2. Start Jupyter:
```bash
jupyter lab
# or
jupyter notebook
```

3. Open the notebook you want to run. Suggested order:
   - `task_1/` notebooks for assignment-specific experiments
   - `DG_NoteBooks/` for domain generalization experiments and baseline comparisons
   - `clip notebooks/` for CLIP-based experiments and evaluation
   - Visualization directories contain notebooks or scripts to produce the figures in the corresponding output folders.

4. If notebooks expect datasets, place datasets into a `data/` folder (create one) and update any path variables inside the notebooks. Check notebook metadata/comments for expected dataset names and structure.

## Reproducing experiments and analyzing results

- DomainBed experiments / ERM or other DomainBed configs:
  - See `ERM domainbed/` for configuration; use the `analyze_domainbed_results.py` script to summarize runs saved by DomainBed.
  - Typical workflow:
    1. Run DomainBed experiments (following DomainBed instructions).
    2. Save outputs to `DG_OUTPUTs/` (or other folder).
    3. Run:
       ```bash
       python analyze_domainbed_results.py --input <path_to_domainbed_results> --output summary.csv
       ```
    4. Inspect the generated summary CSV or plots.

- GroupDRO:
  - Use `analyze_groupdro_results.py` to parse and summarize results produced by GroupDRO experiments.

- Visualizations:
  - Visualizations are organized by method (DANN_visualizations, CDAN_visualizations, DAN_visualizations, etc.). Open the notebooks or scripts in those folders to reproduce plots. They typically read results from the corresponding results folders.

## Notes & tips

- Many notebooks are exploratory — run cells sequentially and watch for notebook-specific configuration cells at the top (paths, device selection, hyperparameters).
- If running on a GPU-enabled environment, ensure PyTorch detects the GPU (e.g., `torch.cuda.is_available()`).
- For large datasets or long training runs, prefer running core training scripts on a remote GPU instance and using notebooks for analysis/visualization.

## License

Add a license for reuse (e.g., MIT). If you want, I can add an MIT license file.

## Contact

Repository owner: rabiaaslam92 (GitHub)
