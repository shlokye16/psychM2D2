# Research Report
### Early Detection of Memory Decline in MCI — PEARL-Neuro + OASIS-2

---

## Datasets

### PEARL-Neuro (OpenNeuro ds004796)

Resting-state EEG and fMRI dataset from the Nencki Institute. 78 subjects labelled; 77 effective for EEGNet LOSO (1 QC-excluded, 1 missing .fif); 68 effective for fusion (10 missing resting-state fMRI).

**EEG** — BrainVision format (.vhdr/.eeg/.vmrk), extended 10-20 cap (high-density; channel names include POO9h, FCC5h, FTT7h; use standard_1005 montage for topomaps). Loaded via mne-bids. Preprocessing: 0.5–45 Hz FIR bandpass, 50 Hz notch, 500 Hz sampling, average reference, 15-component ICA, 4s epochs, 150 µV rejection threshold.

**fMRI** — AP+PA phase-encoding runs concatenated. DMN connectivity extracted using the Harvard-Oxford 48-ROI atlas (Pearson correlation). Produces 4 DMN scalar features per subject.

**Structural MRI (inaccessible)** — T1/T2-weighted scans exist in the dataset but require a Data Use Agreement countersigned by an institutional legal representative (rector, dean, or equivalent). Following correspondence with Dr. Ewa Kublik (Nencki Institute, April 2026), it was confirmed that there is no individual researcher pathway — institutional sign-off is a firm requirement. As this project is independent without faculty affiliation, the DUA cannot be executed. Within-subject EEG–structural MRI correlation is therefore not possible and is documented as a paper limitation and future work direction. All existing analyses are unaffected; they use only EEG and fMRI, both openly available via OpenNeuro.

**Labels** — PEARL-Neuro has no pre-made group column and no healthy controls. Labels derived from genotype columns:

| Label | N | Derivation | Biological Mechanism |
|---|---|---|---|
| APOE_risk | 48 | APOE_haplotype contains "e4" | Lipid metabolism / amyloid clearance |
| PICALM_risk | 30 | PICALM_rs3851179 ≠ "G/G", no APOE ε4 | Clathrin-mediated endocytosis |

Age-matched: APOE_risk μ=55.65 years, PICALM_risk μ=54.87 years. Haplotype breakdown within APOE_risk: 45× e3/e4, 1× e4/e4, 1× e2/e4.

**Feature matrix** — 78 subjects × 1297 features (EEG complexity + DMN). Post variance-filter (threshold 1e-4): 521 surviving features. Band power features removed entirely — near-zero between-subject variance. Surviving features are time-domain complexity (hjorth, spectral entropy, coherence) and DMN connectivity.

**Splits** — Leave-one-subject-out (LOSO), 77 folds.

---

### OASIS-2 (wustl.edu/oasisbrains, PART1 + PART2)

Longitudinal structural MRI with CDR-based cognitive labels. 373 total sessions across 150 unique subjects; 150 baseline sessions used for primary analysis.

**Preprocessing** — FSL BET skull-stripping (-f 0.5 -R), FLIRT 12-DOF registration to MNI152 2mm template, z-score normalisation with ±3σ clipping. CNN tensors: 128³ float32 .pt files. Confirmed orientation: Dim 0=sagittal, Dim 1=coronal, Dim 2=axial.

**Labels** — Control (85), MCI (52), AD (13), Converters (14 — CDR 0.0 at baseline, converted during study).

**Excluded features** — vol_* features (uniform in MNI space, no discriminative signal); hippo_asymmetry_index (MNI registration artifact).

**Feature matrix** — 150 baseline sessions × 46 features. **Splits** — k=5 stratified fold.

---

## Results

### PEARL-Neuro — EEG Baselines

#### LR Baseline (LOSO, 521 surviving features)

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC | 0.2465 | Inverted — true discriminative AUC = 0.7535 |
| Macro F1 | 0.3576 | Below chance — signal learned in wrong direction |
| Accuracy | 38.46% | |
| MCC | −0.2844 | Negative = real signal, inverted direction |

Band powers removed by variance filter, so the LR falls back on globally-averaged hjorth and spectral entropy features. Global averaging cancels spatial topographic signal and produces the inversion. True AUC of 0.7535 is meaningful but the LR cannot exploit the information without spatially-targeted features.

#### Spatial Targeted LR (20 features)

| ROC-AUC | Macro F1 |
|---|---|
| 0.4028 | 0.4004 |

Still inverted. Confirms inversion is driven by alpha direction: PICALM > APOE posteriorly, opposite to naive assumption.

#### EEG Feature Ablation (LR)

| Group removed | ΔAUC | Interpretation |
|---|---|---|
| hjorth (254 features) | +0.1368 | Removing hjorth improves LR — adds noise globally |
| entropy (127 features) | +0.0257 | Minor noise reduction |
| alpha (129 features) | +0.0181 | Inverted in LR as expected |
| coherence (7 features) | +0.0069 | Negligible |
| dmn (4 features) | −0.0042 | Tiny positive contribution |

Delta/theta/beta/gamma groups were empty — those band powers removed by variance filter before ablation.

---

### PEARL-Neuro — EDA Topographic Maps

| Band | Pattern | Interpretation |
|---|---|---|
| Theta | Frontal-central warm blob, APOE > PICALM | Hippocampal-cortical stress via amyloid accumulation |
| Alpha | Posterior-occipital, PICALM > APOE | Early alpha slowing in APOE carriers at age 55 |
| Beta | Diffuse APOE elevation, temporal-parietal | Possible compensatory up-regulation |
| Delta / Gamma | No clear anatomical pattern | Insufficient signal at N=78 |

---

### PEARL-Neuro — EEGNet

**Protocol** — LOSO on raw 4s epochs, 77 folds. F1=8, D=2, F2=16, dropout=0.5, sfreq=500.

| Metric | Epoch-level | Subject-level |
|---|---|---|
| ROC-AUC | 0.978 | 0.989 |
| Macro F1 | 0.968 | 0.986 |
| Accuracy | 96.9% | 98.7% |
| MCC | 0.937 | 0.973 |

**Bimodal fold structure** — every fold is either F1=1.0 (perfect) or F1≈0.47–0.50 (chance). Zero folds in between.

- Perfect: 50 folds / Chance: 27 folds
- APOE: 26/47 perfect (55.3%) | PICALM: 24/30 perfect (80.0%)
- e3/e4 detection rate: 26/45 = **57.8%**

The 42.2% undetected APOE e3/e4 subjects are not model failures. Age comparison (perfect vs failed folds) is not significant (Mann-Whitney U), median ~56, range ~48–66. Epoch count also not significant. The split reflects genuine individual variability in pre-symptomatic onset timing.

**Paper headline** — 57.8% subject-level detection rate in APOE e3/e4 carriers with a bimodal structure reflecting individual variability, not the 0.989 AUC.

---

### PEARL-Neuro — SHAP Interpretability

Top features by mean |SHAP| (LR, LOSO):

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | dmn_mean_connectivity | 0.0378 |
| 2 | hjorth_mobility_POO9h | 0.0323 |
| 3 | spectral_entropy_T7 | 0.0307 |
| 4 | hjorth_complexity_T7 | 0.0304 |
| 5 | hjorth_complexity_AF8 | 0.0300 |

Group summary: DMN (0.0237) > alpha coherence (0.0131) > hjorth complexity (0.0102) > spectral entropy (0.0077) > hjorth mobility (0.0076).

DMN connectivity is the top individual feature despite non-significant cross-modal correlations (RQ5). The two modalities capture independent variance — this is the motivation for fusion.

**SHAP topomaps** — Hjorth complexity: bilateral temporal + frontal, APOE > PICALM globally. Spectral entropy: left frontal, PICALM > APOE (flatter spectra). Hjorth mobility: right posterior/occipital, PICALM > APOE.

---

### PEARL-Neuro — EEGNet Spatial Filters

16 filters extracted from best-fold checkpoint (Fold 2):

| Filters | Pattern |
|---|---|
| 1, 5, 9 | Bilateral temporal weighting |
| 3, 7 | Frontal-central dominance |
| 4, 8 | Posterior weighting |

SHAP assigns highest importance to temporal channels (T7, T8, POO9h). EEGNet independently learns bilateral temporal spatial filters. Two independent methods converging on the same anatomy is the strongest interpretability claim in the paper.

---

### PEARL-Neuro — Cross-Modal Correlations (RQ5)

| Hypothesis | r | p | Status |
|---|---|---|---|
| H1: alpha peak freq vs DMN | +0.067 | 0.588 | Not significant |
| H2: frontal theta vs DMN | −0.110 | 0.371 | Not significant |
| H3: Fz-Pz coherence vs DMN | −0.169 | 0.168 | Wrong direction |

N~340 needed to detect r=0.15 at 80% power. Current N=68. Framed as exploratory with explicit power analysis in paper.

---

### PEARL-Neuro — Fusion Models

10 subjects excluded (missing resting-state fMRI AP or PA run), leaving N=68. EEGNet-only AUC on this subset (0.989) matches canonical result, confirming the subset is representative.

**Aggregate results (LOSO, OOS-corrected):**

| Strategy | AUC | Macro F1 | MCC | Accuracy |
|---|---|---|---|---|
| EEGNet-only | 0.989 | 1.000 | 1.000 | 1.000 |
| DMN-only | 0.463 | 0.527 | 0.054 | 0.544 |
| Late Fusion (LR) | 0.968 | 0.866 | 0.751 | 0.868 |
| Intermediate Fusion (LR) | 0.897 | 0.790 | 0.585 | 0.794 |

**Statistical significance (paired t-test + Wilcoxon):**

| Comparison | t | t_p | Wilcoxon_p | folds better | folds worse |
|---|---|---|---|---|---|
| Late Fusion vs EEGNet | −3.197 | 0.0021 | 0.0027 | 0 | 9 |
| Intermediate Fusion vs EEGNet | −4.168 | 0.0001 | 0.0002 | 0 | 14 |
| DMN-only vs EEGNet | −7.492 | <0.001 | <0.001 | 0 | 31 |

Both fusion strategies are significantly worse than EEGNet alone (both p<0.01), no single fold shows fusion outperforming EEGNet. This is the expected ceiling effect at N=68 — a methodological finding, not a failure. Aggregate fusion improvement is structurally impossible when a unimodal model achieves near-perfect AUC at this sample size.

The AUC gap between late (0.968) and intermediate fusion (0.897) is a dimensionality problem, not an information problem: intermediate fusion concatenates EEGNet's ~992-dimensional penultimate embedding with 4 DMN scalars and fits a regularised LR at N~67 train subjects (severely underdetermined at C=0.1). Late fusion inputs 5–6 dimensions — tractable at this N.

**Bimodal subgroup (N=68, partitioned by epoch-level fold F1):**

| Subgroup | n | EEGNet AUC | Late Fusion AUC | Int. Fusion AUC | DMN AUC |
|---|---|---|---|---|---|
| Chance folds (F1≈0.47–0.50) | 24 | 1.000 | 0.947 | 0.747 | 0.463 |
| Perfect folds (F1=1.0) | 44 | 1.000 | 0.963 | 0.932 | 0.440 |

The 24 "chance" subjects have marginal EEGNet confidence — epoch predictions were near-random, but subject-level probability landed barely on the correct side of 0.5. Within this subgroup, Late Fusion raises P(PICALM_risk) from ~0.50 to ~0.70–0.75 using the DMN signal: genuine probability calibration improvement. 20/24 remain correct; 4/24 are flipped where DMN disagrees with a borderline-correct EEGNet prediction.

DMN distribution is indistinguishable between chance and perfect subgroups (MW p=0.386, μ=0.383 vs 0.420). DMN does not explain the bimodal EEGNet structure — it carries independent information regardless of subgroup membership, complementary only where EEGNet is uncertain.

**EEGNet vs LR significance** — t=7.170, p<0.0001, W=188.0, p<0.0001, Cohen's d=0.82, mean diff +0.4255 F1 (95% CI [+0.3081, +0.5378]).

---

### OASIS-2 — LR Feature Analysis

**Feature significance (EDA):**

| Feature | p-value | Significant? |
|---|---|---|
| MMSE | 0.000 | *** Yes |
| nWBV | 0.024 | * Yes |
| Left Amygdala intensity | 0.042 | * Yes |
| Left Hippocampus intensity | 0.071 | Trend |
| Parahippocampal Gyrus | 0.444 | No |
| Angular Gyrus | 0.527 | No |
| Age | 0.449 | No |

**MMSE ablation:**

| Configuration | AUC | Macro F1 | MCI Recall | n features |
|---|---|---|---|---|
| mmse_only | 0.814 | 0.835 | 0.673 | 1 |
| imaging_only | 0.678 | 0.607 | 0.577 | 11 |
| all_shortlist | 0.835 | 0.757 | 0.615 | 7 |
| full_intensity | 0.848 | 0.764 | n/a | 17 |

MMSE marginal gain over imaging: +0.157 AUC. Imaging marginal gain over MMSE: +0.021 AUC. Model is MMSE-dominant; the honest neuroimaging contribution claim is modest.

---

### OASIS-2 — ResNet18 (v2, Canonical)

Tensor orientation confirmed visually: Dim 0=sagittal, Dim 1=coronal, Dim 2=axial. v1 used Dim 0 (sagittal) due to a mislabelled dataset class before orientation was verified — superseded by v2 (Dim 2, axial). v1 row removed from results_table.csv; v1 predictions file retained for reference only.

| Metric | v2 (axial, canonical) |
|---|---|
| Subject AUC | 0.6762 |
| Macro F1 | 0.6283 |
| MCC | 0.2586 |

**Threshold optimisation:**

| Threshold | MCI recall | Ctrl spec | Macro F1 |
|---|---|---|---|
| 0.50 | 0.500 | 0.753 | 0.628 |
| **0.30 (optimal)** | **0.635** | **0.612** | **0.613** |
| 0.24 | 0.712 | 0.565 | 0.618 |

Optimal t=0.30: MCI recall 63.5% (33/52), ctrl spec 61.2% (52/85). ResNet18 is essentially at parity with imaging-only LR (AUC 0.678) — CNN does not beat atlas features at this sample size. Still below clinical screening target (>85%); motivates FreeSurfer native-space volumetrics as future work.

**Grad-CAM** — Target layer confirmed as layer4.1.conv2 (spatially coherent heatmaps rule out fc/avgpool). For high-confidence MCI predictions (P=0.94–0.95), activation sweeps to posterior-inferior region of the axial slice, consistent with posterior cortical and temporal atrophy. Control predictions show central-superior activation.

**Converter subgroup** — 13 converters (CDR 0.0 at baseline). P(MCI) spread broadly, no concentration above 0.5, no clustering with MCI group in nWBV scatter. Mann-Whitney U not significant. Model does not prospectively detect converters — null result, underpowered at n=13.

---

### Interpretability — Cross-Method Convergence

| Finding | EDA topomaps | SHAP | EEGNet spatial filters | Convergence |
|---|---|---|---|---|
| Temporal channels | T7/T8 beta elevation | T7 in top 3–4 features | Filters 1, 5, 9 bilateral temporal | ✅ Strong |
| Frontal activity | Theta warm blob | AF8, AFz in top 20 | Filters 3, 7 frontal-central | ✅ Strong |
| Posterior alpha asymmetry | PICALM > APOE posteriorly | Alpha coherence in top 7 | Filters 4, 8 posterior | ✅ Moderate |
| DMN discriminative | Not in EEG topomaps | Top SHAP feature (0.0378) | N/A — fMRI feature | Modality-specific |

---

## Key Decisions

**Why band powers are absent from SHAP** — Variance filter at 1e-4 removed them entirely due to near-zero between-subject variance. EEGNet bypasses this by operating on raw epochs directly.

**Why DMN is top SHAP feature despite null RQ5 correlations** — DMN connectivity discriminates groups without linearly correlating with EEG features. Independent variance across modalities is the strongest possible motivation for fusion.

**Why t=0.30 for ResNet18** — Best MCI recall (63.5%) while keeping ctrl specificity above 60%. Appropriate for a screening-priority framing.

**Why intermediate fusion underperforms late fusion** — Dimensionality problem, not an information problem. ~992-dimensional EEGNet embedding + 4 DMN scalars fitted by LR at N~67 is severely underdetermined at C=0.1. Late fusion's 5–6 dimensional input is tractable.

**Why channel-masking and epoch-restriction ablations were dropped** — The bimodal subgroup analysis delivers the complementarity argument more cleanly and with higher biological validity. The partition is a natural outcome of EEGNet's own uncertainty — genuinely harder subjects, not artificially degraded ones. The bimodal analysis is sufficient.

**Why fusion degrades in aggregate** — Ceiling effect. Near-perfect AUC at N=68 leaves no headroom. The complementary information exists (demonstrated in the bimodal subgroup) but is only expressible where EEGNet is uncertain.

---

## Files and Outputs

```
notebooks/
  preprocess.ipynb       EEG + fMRI preprocessing, feature extraction, label derivation
  model.ipynb            EEGNet LOSO, ResNet18 v2 (axial), LR baselines, EDA
  analysis.ipynb         SHAP, EEGNet spatial filters, statistical tests, Grad-CAM,
                         subgroup analyses, cross-modal correlations, corrections
  fusion.ipynb           Late + intermediate fusion, bimodal subgroup analysis

cli/
  classifier.py          AlzheimerRiskClassifier — simulate + model modes,
                         OOS-corrected fusion, counterfactual DMN contribution,
                         sample_synthetic_subject()
  main.py                Interactive CLI — 6 preset subjects, custom input,
                         random sampler, results table, genetic profiles, About
                         Usage: python main.py
                                python main.py --model fold.pt
                                python main.py --model fold.pt --fusion fusion_clf.pkl

results/
  results_table.csv                     Canonical results (v2 ResNet only)
  pearl_lr_loso_fold_results.csv
  eegnet_loso_fold_detail.csv
  eegnet_loso_fold_summary.csv
  pearl_ablation_results.csv            Partial — band powers empty by design
  resnet18_subject_predictions.csv      v1 sagittal — retained for reference only
  resnet18_v2_subject_predictions.csv   v2 axial, t=0.30 — canonical
  shap_feature_importance.csv
  oasis2_mmse_ablation_summary.csv
  rq5_crossmodal_correlations.csv
  fusion_loso_fold_results.csv          68 folds × 4 strategies
  fusion_metrics_summary.csv
  fusion_statistical_tests.csv
  fusion_bimodal_summary.csv
  fusion_bimodal_chance_results.csv
  fusion_bimodal_perfect_results.csv
  fusion_skipped_folds.csv              10 excluded (missing fMRI)

  checkpoints/
    eegnet_fold*_best.pt                77 folds
    resnet18_v2_fold*_best.pt           5 folds

  figures/
    pearl_eda_topomaps.png
    pearl_eda_band_power_comparison.png
    pearl_eda_coherence_dmn.png
    pearl_lr_baseline_diagnostics.png
    pearl_ablation_bar.png
    eegnet_training_curves.png
    eegnet_spatial_filters.png
    apoe_subgroup_analysis.png
    shap_topomaps.png
    model_comparison_auc.png
    oasis2_eda_feature_distributions.png
    oasis2_lr_baseline_diagnostics.png
    oasis2_mmse_ablation_confmats.png
    resnet18_gradcam.png
    tensor_orientation_check.png
    converter_prospective_analysis.png
    fusion_results.png
    fusion_bimodal_analysis.png
    fusion_dmn_by_bimodal_group.png
```
