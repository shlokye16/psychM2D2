"""
main.py — Alzheimer's Genetic Risk Classifier Demo
Interactive CLI for exploring EEG + fMRI multimodal classification.

Run:
    python main.py                          # simulate mode (default)
    python main.py --model path/to/fold.pt  # real EEGNet checkpoint

No external dependencies beyond numpy in simulate mode.
PyTorch required only for --model mode.
"""

import sys
import argparse
from classifier import (
    AlzheimerRiskClassifier,
    RESEARCH_DISTRIBUTIONS,
    RESULTS_SUMMARY,
    EEGNET_LOSO_SUMMARY,
    FUSION_BIMODAL_SUMMARY,
)

# ── Preset subject profiles ──────────────────────────────────────────────────
# Drawn from characterised research distributions, covering the key scenarios.

PRESETS = {
    "1": {
        "name": "Typical APOE ε4 carrier — confident EEG fold",
        "desc": (
            "Strong frontal theta elevation, suppressed posterior alpha — classic APOE ε4 "
            "neurophysiological signature. EEGNet is high-confidence. This is the 57.8% "
            "case: a subject whose pre-symptomatic EEG divergence has already emerged."
        ),
        "frontal_theta":   2.55,
        "posterior_alpha": 1.55,
        "temporal_hjorth": 1.19,
        "dmn_connectivity": 0.36,
        "true_label": "APOE_risk",
    },
    "2": {
        "name": "Typical PICALM carrier — confident EEG fold",
        "desc": (
            "Elevated posterior alpha, flatter spectra, lower temporal complexity — "
            "the PICALM endocytosis pathway signature. PICALM subjects have a cleaner "
            "EEG profile, resulting in the 80.0% detection rate in the LOSO analysis."
        ),
        "frontal_theta":   1.78,
        "posterior_alpha": 2.32,
        "temporal_hjorth": 1.04,
        "dmn_connectivity": 0.46,
        "true_label": "PICALM_risk",
    },
    "3": {
        "name": "APOE ε4 carrier — uncertain fold (EEG near-chance, DMN agrees)",
        "desc": (
            "Frontal theta only marginally elevated, posterior alpha not yet suppressed. "
            "This is the 42.2% case — no EEG divergence signature yet. Low DMN "
            "connectivity (0.33) provides a weak reinforcing signal in the same direction, "
            "but neither modality is definitive. Both agree on APOE_risk at low confidence."
        ),
        "frontal_theta":   2.12,
        "posterior_alpha": 1.97,
        "temporal_hjorth": 1.10,
        "dmn_connectivity": 0.33,
        "true_label": "APOE_risk",
    },
    "4": {
        "name": "PICALM carrier — uncertain fold, DMN flips the call",
        "desc": (
            "EEG features are near-ambiguous, landing marginally in APOE territory. "
            "But this subject has notably high DMN connectivity (0.65, well above PICALM "
            "mean of 0.420). Late Fusion leverages this independent signal and flips the "
            "prediction to PICALM_risk — which is correct. This is the paper's core "
            "complementarity finding: DMN rescues borderline EEG calls in the uncertain regime."
        ),
        "frontal_theta":   2.14,
        "posterior_alpha": 1.99,
        "temporal_hjorth": 1.10,
        "dmn_connectivity": 0.65,
        "true_label": "PICALM_risk",
    },
    "5": {
        "name": "Conflicting signals — EEG says APOE, DMN disagrees",
        "desc": (
            "Strong APOE signature in the EEG (elevated frontal theta, suppressed alpha). "
            "But DMN connectivity is atypically high (0.65, PICALM direction). This mirrors "
            "the 4/24 cases in the bimodal subgroup where DMN disagreed with a borderline-correct "
            "EEGNet prediction. Fusion weighs the conflicting evidence — EEGNet dominates here "
            "because it is confident, but DMN visibly pulls the probability down."
        ),
        "frontal_theta":   2.42,
        "posterior_alpha": 1.70,
        "temporal_hjorth": 1.16,
        "dmn_connectivity": 0.65,
        "true_label": "APOE_risk",
    },
    "6": {
        "name": "Both modalities ambiguous — classifier at limits",
        "desc": (
            "EEG features are exactly at the population mean. DMN connectivity is at "
            "population mean. Neither modality provides discriminative signal. This is "
            "the fundamental limit of pre-symptomatic detection at N=78: some subjects "
            "have not yet developed a detectable neural divergence, and the honest "
            "answer is that the classifier cannot call it."
        ),
        "frontal_theta":   2.10,
        "posterior_alpha": 2.00,
        "temporal_hjorth": 1.10,
        "dmn_connectivity": 0.40,
        "true_label": None,
    },
}

# ── Display helpers ──────────────────────────────────────────────────────────

WIDTH = 68

def hr(char="─"):
    print(char * WIDTH)

def header(title: str, char="═"):
    print()
    print(char * WIDTH)
    print(f"  {title}")
    print(char * WIDTH)

def prob_bar(p: float, width: int = 24) -> str:
    filled = int(round(p * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {p:.3f}"

def confidence_bar(c: float, width: int = 24) -> str:
    filled = int(round(c * width))
    colour = "█" if c >= 0.30 else "▒"
    return "[" + colour * filled + "░" * (width - filled) + f"] {c:.3f}"

def wrap(text: str, indent: str = "  ", width: int = WIDTH - 2):
    """Word-wrap a string to width, prefixing each line with indent."""
    words = text.split()
    line  = indent
    for w in words:
        if len(line) + len(w) + 1 > width:
            print(line.rstrip())
            line = indent + w + " "
        else:
            line += w + " "
    if line.strip():
        print(line.rstrip())


def print_result(result: dict, preset: dict = None):
    name  = preset["name"] if preset else "Custom subject"
    label = preset.get("true_label") if preset else None

    header(f"SUBJECT: {name}")

    if preset and preset.get("desc"):
        print()
        wrap(preset["desc"])

    print()
    hr()
    print()

    # ── Input features ───────────────────────────────────────────────────────
    if preset:
        print("  INPUT FEATURES")
        print(f"    Frontal theta log-power   : {preset['frontal_theta']:.2f}  "
              f"(APOE mean {RESEARCH_DISTRIBUTIONS['APOE_risk']['frontal_theta_mean']:.2f}, "
              f"PICALM mean {RESEARCH_DISTRIBUTIONS['PICALM_risk']['frontal_theta_mean']:.2f})")
        print(f"    Posterior alpha log-power : {preset['posterior_alpha']:.2f}  "
              f"(APOE mean {RESEARCH_DISTRIBUTIONS['APOE_risk']['posterior_alpha_mean']:.2f}, "
              f"PICALM mean {RESEARCH_DISTRIBUTIONS['PICALM_risk']['posterior_alpha_mean']:.2f})")
        print(f"    Temporal Hjorth complexity: {preset['temporal_hjorth']:.2f}  "
              f"(APOE mean {RESEARCH_DISTRIBUTIONS['APOE_risk']['hjorth_complexity_mean']:.2f}, "
              f"PICALM mean {RESEARCH_DISTRIBUTIONS['PICALM_risk']['hjorth_complexity_mean']:.2f})")
        print(f"    DMN mean connectivity     : {preset['dmn_connectivity']:.3f}  "
              f"(APOE mean {RESEARCH_DISTRIBUTIONS['APOE_risk']['dmn_connectivity_mean']:.3f}, "
              f"PICALM mean {RESEARCH_DISTRIBUTIONS['PICALM_risk']['dmn_connectivity_mean']:.3f})")
        print()
        hr()
        print()

    # ── Step 1: EEGNet ───────────────────────────────────────────────────────
    print("  STEP 1 — EEGNet  (temporal EEG complexity features)")
    print()
    print(f"    P(APOE_risk)   {prob_bar(result['eeg_probability'])}")
    print(f"    P(PICALM_risk) {prob_bar(1 - result['eeg_probability'])}")
    print()
    print(f"    Confidence     {confidence_bar(result['confidence'])}")
    regime_icon = "⚠  UNCERTAIN" if result["regime"] == "uncertain" else "✓  CONFIDENT"
    print(f"    Regime         {regime_icon}  (threshold: {AlzheimerRiskClassifier.UNCERTAIN_THRESHOLD:.2f})")
    print()
    print(f"    EEGNet call →  {result['prediction']}")
    print()
    hr()
    print()

    # ── Step 2: Late Fusion ──────────────────────────────────────────────────
    print("  STEP 2 — Late Fusion  (EEGNet + fMRI DMN connectivity)")
    print()
    print(f"    P(APOE_risk)   {prob_bar(result['fusion_probability'])}")
    print(f"    P(PICALM_risk) {prob_bar(1 - result['fusion_probability'])}")
    print()
    dmn_delta  = result["dmn_contribution"]
    delta_str  = f"{dmn_delta:+.4f}"
    direction  = (
        "→ DMN pushed toward PICALM_risk" if dmn_delta < -0.01 else
        "→ DMN pushed toward APOE_risk"   if dmn_delta >  0.01 else
        "→ DMN near-neutral"
    )
    print(f"    DMN shift      {delta_str}  {direction}")
    print()
    agree     = result["prediction"] == result["fusion_prediction"]
    flip_note = "" if agree else "  ← FUSION FLIPPED THE CALL"
    print(f"    Fusion call →  {result['fusion_prediction']}{flip_note}")
    print()
    hr()
    print()

    # ── True label ───────────────────────────────────────────────────────────
    if label:
        eeg_correct    = "✓" if result["prediction"]        == label else "✗"
        fusion_correct = "✓" if result["fusion_prediction"] == label else "✗"
        print(f"  TRUE LABEL  :  {label}")
        print(f"  EEGNet      :  {eeg_correct}  {result['prediction']}")
        print(f"  Fusion      :  {fusion_correct}  {result['fusion_prediction']}")
        print()
        hr()
        print()

    # ── Interpretation ───────────────────────────────────────────────────────
    print("  INTERPRETATION")
    print()
    wrap(result["interpretation"])
    print()

    # ── Bimodal callout ──────────────────────────────────────────────────────
    if result["regime"] == "uncertain":
        hr("·")
        print()
        print("  ★  BIMODAL UNCERTAIN REGIME")
        print()
        n_chance  = FUSION_BIMODAL_SUMMARY["chance_subjects"]
        n_fusion  = FUSION_BIMODAL_SUMMARY["n_subjects"]
        n_eeg     = EEGNET_LOSO_SUMMARY["chance_folds"]
        n_eeg_tot = EEGNET_LOSO_SUMMARY["n_folds"]
        wrap(
            f"In the real LOSO experiment, {n_eeg}/{n_eeg_tot} EEGNet folds were near-chance. "
            f"Of the {n_fusion} subjects with both EEG and fMRI data, {n_chance} fell in this "
            f"uncertain regime. Within that subgroup, Late Fusion raised P(PICALM_risk) from "
            f"~0.50 to ~0.70\u20130.75 using the DMN signal alone, demonstrating genuine "
            f"cross-modal complementarity in exactly the regime it matters most."
        )
        print()

    hr("═")


def print_about():
    header("ABOUT THIS RESEARCH")
    loso_n    = EEGNET_LOSO_SUMMARY["n_subjects_effective"]
    fusion_n  = FUSION_BIMODAL_SUMMARY["n_subjects"]
    perf_f    = EEGNET_LOSO_SUMMARY["perfect_folds"]
    chance_f  = EEGNET_LOSO_SUMMARY["chance_folds"]
    total_f   = EEGNET_LOSO_SUMMARY["n_folds"]
    chance_s  = FUSION_BIMODAL_SUMMARY["chance_subjects"]
    print(f"""
  WHAT IS BEING CLASSIFIED?

  Two Alzheimer's genetic risk profiles in pre-symptomatic adults (~age 55):

    APOE ε4 carriers    Impaired amyloid clearance via lipid metabolism.
                        Most common genetic risk factor (~61% of burden).
                        N=48 in this study (45× e3/e4, 1× e4/e4, 1× e2/e4).

    PICALM carriers     Disrupted clathrin-mediated endocytosis pathway.
                        Secondary genetic risk marker, rs3851179 non-G/G.
                        N=30. No APOE ε4 overlap.

  Neither group is symptomatic. There are no healthy controls — this is a
  within-risk-group discrimination task. Age-matched (APOE 55.65, PICALM 54.87).

  ──────────────────────────────────────────────────────────────────────

  HOW THE PIPELINE WORKS

  1. EEGNet (deep learning on raw EEG epochs)
     Trained on 4-second resting-state EEG epochs, 64 channels, 500 Hz.
     Learns temporal complexity features directly from raw signal.
     Validated LOSO (leave-one-subject-out) across {loso_n} subjects.
     Subject-level AUC: 0.989 — but with a bimodal fold structure.

  2. fMRI Default Mode Network (DMN) connectivity
     4 Pearson correlation features from Harvard-Oxford 48-ROI atlas.
     Top SHAP feature across all models (mean |SHAP| = 0.0378).
     Does NOT linearly correlate with EEG features — captures independent
     biological variance. This is the motivation for fusion.

  3. Late Fusion (logistic regression, LR C=0.1)
     Inputs: EEGNet probability scalar + DMN connectivity scalar (5-6 dims).
     Aggregate AUC: 0.968 — worse than EEGNet alone (ceiling effect at N={fusion_n}).
     But in the bimodal-uncertain subgroup (N={chance_s}), DMN shifts P(PICALM_risk)
     from ~0.50 to ~0.70-0.75 — genuine complementarity.

  ──────────────────────────────────────────────────────────────────────

  THE BIMODAL FINDING

  EEGNet's folds are bimodal — every fold is either perfect (F1=1.0) or
  near-chance (F1≈0.47-0.50). Zero folds in between.

    EEGNet LOSO ({total_f} folds total):
      Perfect folds  : {perf_f} / {total_f}   (APOE 26/47 = 55.3%, PICALM 24/30 = 80.0%)
      Chance folds   : {chance_f} / {total_f}

    Fusion bimodal ({fusion_n} subjects with EEG + fMRI):
      Uncertain (chance) : {chance_s} subjects — DMN provides complementary calibration
      Confident (perfect): {FUSION_BIMODAL_SUMMARY['perfect_subjects']} subjects

  The 42.2% undetected APOE e3/e4 subjects are not model failures.
  Age analysis (Mann-Whitney U) confirmed no age effect — these are
  individuals whose pre-symptomatic EEG divergence has not yet emerged.
  Individual variability in biological onset timing, independent of age
  and recording quality.

  ──────────────────────────────────────────────────────────────────────

  CROSS-METHOD CONVERGENCE (interpretability)

  Three independent methods agreed on the same anatomy:

    EEG topomaps      Frontal theta warm blob, bilateral temporal beta elevation
    SHAP analysis     T7/T8 in top 3-4 features, AF8/AFz in top 20
    EEGNet filters    Filters 1,5,9 bilateral temporal; Filters 3,7 frontal

  Two independent methods converging on the same anatomy is the strongest
  defensible interpretability claim in the paper.

  ──────────────────────────────────────────────────────────────────────

  DATASET

  PEARL-Neuro (OpenNeuro ds004796) — resting-state EEG + fMRI
  N=78 labelled subjects ({loso_n} effective for EEGNet; {fusion_n} with paired fMRI for fusion)
  BrainVision EEG cap (extended 10-20, standard_1005 montage)
  Preprocessing: 0.5-45 Hz FIR, 50 Hz notch, 15-component ICA, 4s epochs
  fMRI: AP+PA concatenated, Harvard-Oxford 48-ROI Pearson connectivity
""")
    hr("═")


def print_results_table():
    loso_n   = EEGNET_LOSO_SUMMARY["n_subjects_effective"]
    fusion_n = FUSION_BIMODAL_SUMMARY["n_subjects"]
    perf_f   = EEGNET_LOSO_SUMMARY["perfect_folds"]
    chance_f = EEGNET_LOSO_SUMMARY["chance_folds"]
    total_f  = EEGNET_LOSO_SUMMARY["n_folds"]
    chance_s = FUSION_BIMODAL_SUMMARY["chance_subjects"]
    perf_s   = FUSION_BIMODAL_SUMMARY["perfect_subjects"]

    header(f"EXPERIMENTAL RESULTS (N={loso_n} EEGNet LOSO | N={fusion_n} fusion)")
    print()
    print(f"  {'Strategy':<28} {'AUC':>7} {'F1':>7} {'MCC':>7} {'Acc':>7}")
    hr()
    for strategy, m in RESULTS_SUMMARY.items():
        marker = " ← research baseline" if strategy == "EEGNet_only" else ""
        print(f"  {strategy:<28} {m['AUC']:>7.3f} {m['F1']:>7.3f} {m['MCC']:>7.3f} {m['Acc']:>7.3f}{marker}")
    print()
    print(f"  Note: 10 subjects excluded from fusion — missing resting-state fMRI.")
    print(f"  EEGNet AUC on fusion subset (0.989) matches canonical N={loso_n} result.")
    print()
    hr()
    print()
    print("  EEGNET LOSO BIMODAL STRUCTURE")
    print(f"  (N={loso_n} subjects, {total_f} LOSO folds)")
    print()
    print(f"  Perfect folds (F1=1.0)       : {perf_f} / {total_f}")
    print(f"  Chance folds  (F1≈0.47-0.50) : {chance_f} / {total_f}")
    print(f"  APOE e3/e4 detection rate    : {EEGNET_LOSO_SUMMARY['e34_detection_rate']*100:.1f}%  (26/45)")
    print(f"  PICALM detection rate        : {EEGNET_LOSO_SUMMARY['PICALM_detection_rate']*100:.1f}%  (24/30)")
    print()
    hr()
    print()
    print("  FUSION BIMODAL SUBGROUP")
    print(f"  (N={fusion_n} subjects with paired fMRI)")
    print()
    print(f"  Uncertain subjects (epoch F1≈0.47-0.50) : {chance_s}")
    print(f"  Confident subjects (epoch F1=1.0)       : {perf_s}")
    print(f"  Late Fusion AUC — uncertain subgroup    : {FUSION_BIMODAL_SUMMARY['chance_late_fusion_AUC']}")
    print(f"  Late Fusion AUC — confident subgroup    : {FUSION_BIMODAL_SUMMARY['perfect_late_fusion_AUC']}")
    print(f"  DMN null distribution (MW p)            : {FUSION_BIMODAL_SUMMARY['dmn_null_p']}  (DMN does not explain bimodal structure)")
    print()
    wrap(FUSION_BIMODAL_SUMMARY["dmn_shift_description"])
    print()
    hr("═")


def print_profiles():
    header("GENETIC RISK PROFILES")
    profiles = AlzheimerRiskClassifier.get_profile_description()
    for label, p in profiles.items():
        print()
        print(f"  {label}  (N={p['N']})")
        hr()
        print(f"  Full name  : {p['full_name']}")
        print(f"  Mechanism  : {p['mechanism']}")
        print(f"  Age mean   : {p['age_mean']}")
        print()
        print(f"  EEG signature:")
        wrap(p["eeg_signature"], indent="    ")
        print()
        print(f"  fMRI DMN   : {p['fmri_dmn']}")
        print(f"  Detection  : {p['eegnet_detection']}")
        print(f"  SHAP top   : {p['shap_top_feature']}")
        print()
    hr("═")


def custom_input_flow(clf: AlzheimerRiskClassifier):
    header("CUSTOM SUBJECT — Enter feature values")
    print()
    print("  Feature guide:")
    print("  frontal_theta    log band power at Fz/FCz, ~6 Hz    APOE > PICALM")
    print("  posterior_alpha  log band power at O1/O2/POO9h       PICALM > APOE")
    print("  temporal_hjorth  Hjorth complexity at T7/T8          APOE > PICALM")
    print("  dmn_connectivity mean Pearson r, 48-ROI Harvard-Oxford")
    print()
    print("  Population means: theta=2.1, alpha=2.0, hjorth=1.10, dmn=0.40")
    print("  (Press Enter to use population mean for any feature)")
    print()

    def ask(prompt: str, default: float, lo: float, hi: float) -> float:
        while True:
            raw = input(f"  {prompt} [default={default}]: ").strip()
            if raw == "":
                return default
            try:
                val = float(raw)
                if lo <= val <= hi:
                    return val
                print(f"  → Please enter a value between {lo} and {hi}")
            except ValueError:
                print("  → Please enter a number.")

    ft     = ask("Frontal theta log-power    (range 0.5–4.0)", 2.10, 0.5, 4.0)
    pa     = ask("Posterior alpha log-power  (range 0.5–4.0)", 2.00, 0.5, 4.0)
    hjorth = ask("Temporal Hjorth complexity (range 0.8–1.5)", 1.10, 0.8, 1.5)
    dmn    = ask("DMN mean connectivity      (range 0.0–1.0)", 0.40, 0.0, 1.0)

    result = clf.predict(
        frontal_theta=ft,
        posterior_alpha=pa,
        temporal_hjorth=hjorth,
        dmn_connectivity=dmn,
    )
    custom_preset = {
        "name":             "Custom subject",
        "desc":             None,
        "frontal_theta":    ft,
        "posterior_alpha":  pa,
        "temporal_hjorth":  hjorth,
        "dmn_connectivity": dmn,
        "true_label":       None,
    }
    print_result(result, custom_preset)


def sample_random_flow(clf: AlzheimerRiskClassifier):
    import random
    header("RANDOM SYNTHETIC SUBJECT")
    print()
    label    = random.choice(["APOE_risk", "PICALM_risk"])
    seed     = random.randint(0, 9999)
    print(f"  Sampling from {label} distribution  (seed={seed})")
    print()
    features = clf.sample_synthetic_subject(label, seed=seed)
    result   = clf.predict(
        frontal_theta=features["frontal_theta"],
        posterior_alpha=features["posterior_alpha"],
        temporal_hjorth=features["temporal_hjorth"],
        dmn_connectivity=features["dmn_connectivity"],
    )
    preset = {
        "name":      f"Synthetic {label} (seed {seed})",
        "desc":      f"Randomly sampled from the {label} research distribution.",
        **features,
        "true_label": label,
    }
    print_result(result, preset)


# ── Main loop ────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║  MULTIMODAL ALZHEIMER'S GENETIC RISK CLASSIFIER                     ║
║  EEG Temporal Complexity + fMRI DMN Connectivity Fusion             ║
║  PEARL-Neuro (OpenNeuro ds004796) · N=78 · LOSO Protocol            ║
║  UC Davis Undergraduate Research · 2026                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

MENU = """
  MENU
  ─────────────────────────────────────────────────────────────────────
  Preset subjects:
    [1]  Typical APOE ε4 carrier         (confident EEG fold)
    [2]  Typical PICALM carrier          (confident EEG fold)
    [3]  APOE ε4 carrier                 (uncertain fold — bimodal)
    [4]  PICALM carrier                  (uncertain fold — DMN rescues)
    [5]  Conflicting signals             (EEG vs DMN disagree)
    [6]  Both modalities ambiguous       (classifier at limits)
  Other:
    [7]  Enter custom feature values
    [8]  Sample random synthetic subject
    [r]  View full results table
    [p]  View genetic risk profiles
    [a]  About this research
    [q]  Quit
  ─────────────────────────────────────────────────────────────────────
"""


def main():
    parser = argparse.ArgumentParser(description="Alzheimer's Genetic Risk Classifier Demo")
    parser.add_argument(
        "--model", metavar="CHECKPOINT",
        help="Path to EEGNet state_dict .pt file. If omitted, runs in simulate mode."
    )
    parser.add_argument(
        "--fusion", metavar="FUSION_WEIGHTS",
        help="Path to pickled late fusion LogisticRegression (optional, model mode only)."
    )
    args = parser.parse_args()

    print(BANNER)

    if args.model:
        print(f"  Loading EEGNet checkpoint: {args.model}")
        clf = AlzheimerRiskClassifier(
            mode="model",
            checkpoint_path=args.model,
            fusion_weights_path=args.fusion,
        )
        print("  Running in MODEL mode — actual EEGNet checkpoint.")
        print("  (Supply --epoch path/to/epoch.npy to classify real EEG data)\n")
    else:
        clf = AlzheimerRiskClassifier(mode="simulate")
        print("  Running in SIMULATE mode.")
        print("  EEG features are scalar inputs; EEGNet behaviour is approximated")
        print("  from the research-derived distributions (LOSO N=77 analysis).")
        print("  Pass --model path/to/checkpoint.pt to use the real EEGNet.\n")

    while True:
        print(MENU)
        choice = input("  Choose: ").strip().lower()

        if choice == "q":
            print("\n  Exiting. Thanks for exploring the research.\n")
            break
        elif choice in PRESETS:
            p = PRESETS[choice]
            result = clf.predict(
                frontal_theta=p["frontal_theta"],
                posterior_alpha=p["posterior_alpha"],
                temporal_hjorth=p["temporal_hjorth"],
                dmn_connectivity=p["dmn_connectivity"],
            )
            print_result(result, p)
        elif choice == "7":
            custom_input_flow(clf)
        elif choice == "8":
            sample_random_flow(clf)
        elif choice == "r":
            print_results_table()
        elif choice == "p":
            print_profiles()
        elif choice == "a":
            print_about()
        else:
            print("\n  Invalid choice — pick a number from the menu or a letter command.\n")


if __name__ == "__main__":
    main()
