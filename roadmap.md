# Research Roadmap
### Early Detection of Memory Decline in MCI — PEARL-Neuro + OASIS-2

---

## Research Theme

> "Pre-symptomatic neurophysiological divergence between APOE ε4 and PICALM genetic risk pathways for Alzheimer's disease: a multimodal EEG and neuroimaging analysis."

## Novelty Statement

> "This work presents a multimodal neurophysiological framework for characterising genetic risk pathways toward memory decline, jointly modelling resting-state EEG biomarkers of functional brain divergence and fMRI DMN connectivity features — comparing pre-symptomatic electrophysiological signatures between APOE ε4 and PICALM rs3851179 genetic risk carriers, mapping the relationship between EEG functional biomarkers and MRI structural atrophy across a validated CDR-labelled cohort, and demonstrating that DMN connectivity provides independent, complementary discriminative information in precisely the regime where EEGNet's temporal complexity features are maximally uncertain."

## Five Novel Contributions

1. **Two-pathway genetic comparison** — direct EEG signature comparison between APOE ε4 and PICALM risk carriers in a pre-symptomatic cohort, grounding classification in distinct biological mechanisms rather than a binary at-risk flag.
2. **Same-subject cross-modal analysis** — EEG functional biomarkers and fMRI DMN connectivity from identical individuals at the same timepoint, enabling genuine biological correlation rather than dataset-level fusion.
3. **Feature divergence by pathway (RQ2)** — temporal complexity and spectral entropy identified as primary discriminators, with convergent support from SHAP and EEGNet spatial filters.
4. **Cross-modal complementarity via bimodal subgroup** — DMN connectivity is the single top SHAP feature and captures independent variance from EEG. Within the 24 subjects where EEGNet is maximally uncertain (epoch-level F1≈0.47–0.50), late fusion raises P(PICALM_risk) from ~0.50 to ~0.70–0.75, demonstrating genuine probability calibration improvement where EEG alone is insufficient.
5. **Pre-symptomatic framing** — PEARL-Neuro participants are middle-aged genetic risk carriers before any clinical MCI diagnosis. 57.8% detection rate in e3/e4 carriers with bimodal structure reflecting genuine individual variability in onset timing.

---

## Research Questions

**RQ1 — EEG-Only Classification**
Can resting-state EEG spectral and connectivity features distinguish APOE ε4 carriers from PICALM rs3851179 carriers at a statistically meaningful threshold, prior to any clinical diagnosis?

**RQ2 — Biomarker Divergence by Pathway** *(most original contribution)*
Which EEG biomarkers diverge most between the two genetic risk groups?

**RQ3 — Neuroimaging Contribution**
Does fMRI DMN connectivity or structural MRI atrophy improve classification when combined with EEG?

**RQ4 — Neuroanatomical Attribution**
Which brain regions does Grad-CAM identify as most predictive of CDR-based decline?

**RQ5 — Cross-Modal Biological Correlation**
Does EEG alpha coherence correlate with fMRI DMN connectivity in the same subjects?

### RQ Coverage in Paper

| RQ | Introduction | Methods | Results | Discussion |
|---|---|---|---|---|
| RQ1 | State it | EEG model design | EEGNet accuracy + bimodal structure | Clinical threshold context |
| RQ2 | State it | Feature engineering | SHAP + spatial filters + ablation | APOE vs PICALM neurological mechanism |
| RQ3 | State it | Fusion model design | Aggregate ceiling + bimodal subgroup calibration + ResNet + LR MMSE ablation | When neuroimaging adds value |
| RQ4 | Mention | Grad-CAM setup | Heatmap regions | Hippocampal atrophy literature |
| RQ5 | Mention | Cross-modal correlation | Pearson/Spearman + power analysis | DMN disruption theory + independent variance finding |

---

## Task Plan

### Phase 1 — Data Acquisition

- Download PEARL-Neuro (OpenNeuro ds004796) via DataLad
- Download OASIS-2 (wustl.edu/oasisbrains, PART1 + PART2)
- Verify manifest and subject counts
- Delete raw data after preprocessing (free storage)

### Phase 2 — Preprocessing

**PEARL-Neuro EEG**
- Load BrainVision files via mne-bids
- Band-pass filter 0.5–45 Hz (FIR), 50 Hz notch, downsample to 500 Hz
- Re-reference to average, run 15-component ICA
- Epoch to 4s, reject epochs >150 µV
- QC: exclude subjects with >50% epoch rejection or missing .fif

**PEARL-Neuro fMRI**
- Concatenate AP+PA phase-encoding runs
- Extract ROI timeseries using Harvard-Oxford 48-ROI atlas
- Compute Pearson correlation matrix → 4 DMN scalar features per subject

**PEARL-Neuro structural MRI** [skip — DUA required, institutional pathway unavailable]

**PEARL-Neuro labels**
- Derive from genotype columns (no pre-made group column exists)
- APOE_risk: APOE_haplotype contains "e4"
- PICALM_risk: PICALM_rs3851179 ≠ "G/G" AND no APOE ε4
- Verify age matching across groups

**OASIS-2**
- FSL BET skull-stripping (-f 0.5 -R)
- FLIRT 12-DOF registration to MNI152 2mm
- z-score normalisation, ±3σ clipping
- Export 128³ float32 tensors (.pt); verify orientation (Dim 2 = axial)
- Extract atlas-based intensity features per ROI
- Exclude vol_* (uniform in MNI) and hippo_asymmetry_index (MNI artifact)
- Derive labels from CDR: 0.0 = Control, 0.5 = MCI, ≥1.0 = AD

**Splits**
- PEARL-Neuro: LOSO (77 folds)
- OASIS-2: k=5 stratified fold

### Phase 3 — EDA and Baselines

- Compute EEG topographic maps per frequency band (theta, alpha, beta, delta, gamma)
- Run LR baseline on full feature matrix (LOSO)
- Run spatial targeted LR (top 20 topographically-selected features)
- Run EEG feature ablation (remove feature groups, measure ΔAUC)
- OASIS-2: feature significance testing; MMSE ablation across configurations

### Phase 4 — Model Training

**PEARL-Neuro**
- EEGNet: F1=8, D=2, F2=16, dropout=0.5, sfreq=500; train LOSO, 77 folds
- APOE e3/e4 subgroup analysis on perfect vs failed folds (age, epoch count)

**OASIS-2**
- ResNet18: verify tensor orientation before training; use Dim 2 (axial)
- Threshold optimisation: sweep t from 0.24 to 0.50, select by MCI recall / ctrl specificity tradeoff
- Grad-CAM on layer4.1.conv2; verify anatomical coherence

**Within-subject EEG–structural MRI (PEARL-Neuro)** [skip — DUA required]

### Phase 5 — Interpretability

- SHAP (LinearExplainer, background ≤50 samples): feature importance + topomaps
- EEGNet spatial filter extraction from best-fold checkpoint (Fold 2)
- Cross-modal correlations (RQ5): Pearson/Spearman + power analysis
- Statistical significance: paired t-test + Wilcoxon across all model pairs

### Phase 6 — Fusion

- Late fusion: LR C=0.1 class-balanced on [EEGNet scalar prob + DMN scalars] (5–6 dims)
- Intermediate fusion: LR C=0.1 on [EEGNet penultimate embedding + DMN scalars] (~996 dims)
- Both trained OOS-corrected under LOSO (N=68, 10 excluded for missing fMRI)
- Bimodal subgroup analysis: partition by epoch-level fold F1 (≤0.50 = chance, 1.0 = perfect)
- DMN null distribution test: confirm DMN does not explain bimodal structure (MW U)

### Phase 7 — Paper Draft

See Paper Structure below.

---

## Paper Structure

```
Abstract
├── Background + problem (genetic risk detection gap, pre-symptomatic window)
├── Methods summary (PEARL-Neuro EEG+fMRI, APOE vs PICALM, OASIS-2 CDR)
├── Key results (57.8% e3/e4 detection, ResNet AUC 0.6762, DMN calibration
│   improvement in uncertain folds, ceiling effect framing)
└── Significance statement

Introduction
├── Motivation (clinical need, genetic risk pathway divergence)
├── Limitations of prior work (EEG mostly binary AD vs HC, no genetic pathway comparison)
├── Approach (EEG + fMRI, same subjects, two genetic pathways, OASIS-2 anchor)
└── Research questions (RQ1–RQ5, RQ6 optional)

Related Work
├── EEG-based cognitive decline detection (Cassani 2018, Ieracitano 2019, Babiloni 2016)
├── APOE ε4 and PICALM genetic risk in AD (Corder 1993, Harold 2009)
├── MRI-based MCI/AD classification (Wen 2020, Marcus 2010)
├── Multimodal approaches (prior EEG+MRI/fMRI fusion — note the gap)
└── Pre-symptomatic detection literature

Methods
├── Data
│   ├── PEARL-Neuro: preprocessing, label derivation, demographics
│   ├── Note: no healthy controls; classification is within-risk-group
│   ├── Note: 10 subjects excluded from fusion (absent fMRI), N=68 for fusion
│   └── OASIS-2: preprocessing, CDR labels, tensor orientation (Dim 2 = axial)
├── Feature Engineering
│   ├── EEG: hjorth, spectral entropy, coherence, band power (band powers removed by variance filter — document)
│   └── fMRI: Harvard-Oxford 48-ROI DMN connectivity (4 scalars)
├── Model Architecture
│   ├── EEGNet (LOSO, raw epochs)
│   ├── ResNet18 v2 (axial, k=5 fold, threshold t=0.30)
│   ├── LR baseline (OASIS-2, MMSE ablation)
│   └── Fusion: late (5–6 dims) + intermediate (~996 dims), LR C=0.1 class-balanced
└── Evaluation Protocol
    ├── LOSO for PEARL-Neuro, k=5 stratified fold for OASIS-2
    └── OOS-corrected fusion; bimodal subgroup defined by epoch-level F1

Results
├── RQ1 — EEGNet bimodal structure + 57.8% e3/e4 detection rate
├── RQ2 — SHAP (temporal hjorth + spectral entropy) + EEGNet spatial filter convergence
├── RQ3 — Two beats:
│   ├── Aggregate: ceiling effect, both fusion p<0.01 vs EEGNet, late > intermediate (dimensionality)
│   ├── Bimodal subgroup: 24 uncertain subjects, late fusion calibration improvement,
│   │   DMN null distribution (MW p=0.386)
│   ├── ResNet18 v2: AUC 0.6762, MCI recall 63.5% at t=0.30
│   └── LR MMSE ablation: imaging_only 0.678 vs combined 0.835
├── RQ4 — Grad-CAM posterior-inferior shift (MCI) on axial slices
└── RQ5 — All hypotheses non-significant, DMN independently discriminative, power analysis

Discussion
├── e3/e4 heterogeneous onset — age not explanatory, individual variability framing
├── Temporal complexity as APOE vs PICALM discriminator — neurological interpretation
├── MMSE dominance in OASIS-2 — honest neuroimaging claim
├── DMN ceiling effect + bimodal complementarity — information present, expressible only
│   where EEGNet is uncertain; motivates larger-N future work
├── Null RQ5 — power limitation, not decoupling; DMN still informative via classification
└── Ethical implications of pre-symptomatic genetic risk stratification

Limitations
├── PEARL-Neuro structural MRI inaccessible — institutional DUA required; no individual
│   researcher pathway (confirmed Nencki Institute, April 2026); within-subject
│   EEG–structural MRI correlation is future work
├── N=68 fusion / N=77 EEG / N=150 OASIS-2 — underpowered for subgroup claims
└── AD n=13 OASIS-2; converter null result underpowered

Future Work
Conclusion
```

---

## Required Reading

| Paper | Relevance |
|---|---|
| Jöbsis et al., 2023 | PEARL-Neuro dataset paper |
| Marcus et al., 2010 | OASIS-2 dataset paper |
| Jeong, 2004 | EEG biomarkers in AD — grounds theta + complexity features |
| Czigler et al., 2008 | Alpha slowing as early EEG biomarker |
| Babiloni et al., 2016 | EEG coherence in MCI/AD — grounds coherence features |
| Cassani et al., 2018 | Systematic review — benchmark map |
| Ieracitano et al., 2019 | CNN on EEG spectrograms ~88% — EEGNet benchmark |
| Lawhern et al., 2018 | EEGNet architecture |
| Wen et al., 2020 | ResNet on MRI ~90% — ResNet18 benchmark |
| Jack et al., 2018 | ATN framework — grounds nWBV as neurodegeneration marker |
| Petersen et al., 2014 | MCI definition — CDR 0.5 proxy justification |
| Corder et al., 1993 | Original APOE ε4 AD risk paper |
| Harold et al., 2009 | Original PICALM GWAS paper |
| Selvaraju et al., 2017 | Grad-CAM |
| Lundberg & Lee, 2017 | SHAP |

### Benchmark Numbers

- Cassani et al. 2018: ~85–90% Acc, binary AD vs HC (different task)
- Ieracitano et al. 2019: ~88% Acc, binary AD vs HC
- Wen et al. 2020: ~90%+ on ADNI MRI (different dataset)
- No prior benchmark for APOE vs PICALM EEG comparison — task is novel
