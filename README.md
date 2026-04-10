# Early Detection of Memory Decline in MCI

[![Paper Preprint](https://img.shields.io/badge/Zenodo-Paper%20Preprint-636B2F?logo=zenodo)](https://zenodo.org/records/19244475)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MNE](https://img.shields.io/badge/MNE--Python-EEG-00897B)
![License](https://img.shields.io/badge/License-MIT-yellow)

Independent undergraduate research by **Shlok Khare**, University of California, Davis.

---

## Overview

This project presents a multimodal neurophysiological framework for characterising pre-symptomatic divergence between two distinct genetic risk pathways for Alzheimer's disease  (**APOE ε4** and **PICALM rs3851179**)  in a cohort of middle-aged adults prior to any clinical diagnosis.

Rather than treating Alzheimer's risk as a binary flag, this work directly compares the resting-state EEG signatures of two biologically distinct pathways: APOE ε4, which impairs amyloid clearance via lipid metabolism, and PICALM, which disrupts clathrin-mediated endocytosis. The central question is whether these two pathways produce measurably different pre-symptomatic brain states and whether multimodal fusion with fMRI Default Mode Network (DMN) connectivity adds discriminative value beyond EEG alone.

The framework jointly models resting-state EEG biomarkers (temporal complexity, spectral features, Hjorth parameters) with fMRI DMN connectivity features extracted from the same subjects at the same session, enabling genuine within-subject cross-modal analysis rather than dataset-level fusion. A parallel strand of the work examines structural MRI-based classification of CDR-labelled cognitive decline using a separate longitudinal cohort, grounding the EEG findings in an established neurodegeneration framework.

---

## Abstract

*For the full methodology, results, and discussion, see the [paper preprint](https://zenodo.org/records/19244475).*

We propose a multimodal framework to detect pre-symptomatic neurophysiological divergence between APOE ε4 and PICALM genetic risk carriers for Alzheimer's disease, using resting-state EEG and fMRI data from the PEARL-Neuro dataset (OpenNeuro ds004796, N=78). An EEGNet classifier trained under leave-one-subject-out cross-validation achieves strong subject-level discrimination, revealing a bimodal fold structure that reflects genuine individual variability in pre-symptomatic onset timing rather than model instability. SHAP analysis and EEGNet spatial filter extraction converge independently on bilateral temporal and frontal channels as primary discriminators. DMN connectivity emerges as the top individual SHAP feature despite showing no linear correlation with EEG features, motivating late fusion experiments that demonstrate genuine cross-modal complementarity in precisely the regime where EEG is maximally uncertain. A companion ResNet18 classifier on structural MRI from the OASIS-2 longitudinal dataset (N=150 baseline sessions) provides an independent CDR-based anchor for neuroanatomical attribution via Grad-CAM.

---

## Datasets

**PEARL-Neuro** — [OpenNeuro ds004796](https://openneuro.org/datasets/ds004796). Resting-state EEG and fMRI from the Nencki Institute. Openly available via DataLad. Labels derived from APOE haplotype and PICALM rs3851179 genotype columns; no pre-made group column exists in the dataset.

**OASIS-2** — [OASIS Brains](https://www.oasis-brains.org/). Longitudinal structural MRI with CDR-based cognitive labels. Openly available from Washington University.

> Note: PEARL-Neuro structural MRI data requires a Data Use Agreement countersigned by an institutional legal representative (Nencki Institute). This pathway is unavailable for independent researchers without faculty affiliation and is documented as a project limitation.

---

## Repository Structure

```
├── cli/
│   ├── classifier.py        # AlzheimerRiskClassifier — simulate + model modes
│   └── main.py              # Interactive CLI demo
├── data/
│   ├── metadata/            # Manifests, feature matrices, processing summaries
│   ├── processed/           # EEG epochs, features, fMRI features, MRI tensors
│   └── splits/              # LOSO and k-fold split definitions
├── extract/
│   ├── xtpneuro.py          # PEARL-Neuro preprocessing pipeline (EEG + fMRI)
│   └── xtoasis2.py          # OASIS-2 preprocessing pipeline (MRI)
├── notebook/
│   ├── preprocess.ipynb     # Feature extraction and label derivation
│   ├── model.ipynb          # EEGNet, ResNet18, LR baselines, EDA
│   ├── analysis.ipynb       # SHAP, spatial filters, significance tests, Grad-CAM
│   └── fusion.ipynb         # Late + intermediate fusion, bimodal subgroup analysis
├── results/
│   ├── checkpoints/         # EEGNet (77 folds) + ResNet18 v2 (5 folds) checkpoints
│   ├── figures/             # All publication figures
│   └── *.csv                # Fold-level results, SHAP scores, fusion metrics
├── report.md                # Full experimental log and results documentation
├── roadmap.md               # Research questions, paper structure, task plan
└── README.md
```

---

## CLI Demo

A self-contained interactive simulator is included for exploring the EEG + fMRI fusion pipeline without requiring checkpoints or raw data. It runs from research-derived distributions in simulate mode, or accepts a real EEGNet `.pt` checkpoint for model mode.

```bash
pip install numpy torch          # torch only needed for --model mode
python cli/main.py               # simulate mode
python cli/main.py --model results/checkpoints/eegnet_fold2_best.pt
```

---

## Dependencies

Core: `mne`, `mne-bids`, `torch`, `scikit-learn`, `nilearn`, `nibabel`, `shap`, `numpy`, `pandas`, `scipy`

FSL (`bet`, `flirt`) is required for OASIS-2 skull stripping and MRI registration. DataLad is required to fetch PEARL-Neuro raw files.

---

## Citation

If you use this code or build on this work, please cite:

```
Khare, S. (2026). Early Detection of Memory Decline in MCI:
A Multimodal EEG and Neuroimaging Framework for Pre-Symptomatic
Genetic Risk Stratification. University of California, Davis.
https://github.com/shlokye16/psychm2d2
```

---

## License

Dataset usage is subject to the terms of OpenNeuro (PEARL-Neuro) and OASIS (OASIS-2) respectively.
