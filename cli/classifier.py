"""
classifier.py — Alzheimer's Genetic Risk Classifier
Multimodal EEG + fMRI DMN Fusion Pipeline

Distinguishes APOE ε4 risk vs PICALM risk genetic profiles
from resting-state neurophysiological signals.

Two modes:
  simulate — uses research-derived distributions (no checkpoints needed)
  model    — loads actual EEGNet .pt checkpoint + fitted fusion weights

Research: PEARL-Neuro (OpenNeuro ds004796), N=78 labelled (77 effective for EEGNet LOSO).
EEGNet subject-level AUC: 0.989 | Late Fusion AUC: 0.968 (N=68, fusion subset)
Key finding: DMN connectivity complements EEG in the bimodal-uncertain regime.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Research-derived distribution parameters (from LOSO analysis) ────────────

RESEARCH_DISTRIBUTIONS = {
    "APOE_risk": {
        "n": 48,
        "frontal_theta_mean":    2.30,   # log band power, Fz/FCz channels
        "frontal_theta_std":     0.40,
        "posterior_alpha_mean":  1.80,   # log band power, O1/O2/POO9h channels
        "posterior_alpha_std":   0.50,
        "hjorth_complexity_mean": 1.12,  # temporal channels T7/T8
        "hjorth_complexity_std":  0.08,
        "dmn_connectivity_mean": 0.383,  # Harvard-Oxford 48-ROI mean Pearson r
        "dmn_connectivity_std":  0.12,
        "eegnet_detection_rate": 0.578,  # 57.8% — e3/e4 subgroup (26/45)
    },
    "PICALM_risk": {
        "n": 30,
        "frontal_theta_mean":    1.90,
        "frontal_theta_std":     0.30,
        "posterior_alpha_mean":  2.20,
        "posterior_alpha_std":   0.40,
        "hjorth_complexity_mean": 1.08,
        "hjorth_complexity_std":  0.07,
        "dmn_connectivity_mean": 0.420,
        "dmn_connectivity_std":  0.11,
        "eegnet_detection_rate": 0.800,  # 80.0% (24/30)
    },
}

# Research results reference table (N=68 for all fusion strategies)
RESULTS_SUMMARY = {
    "EEGNet_only":         {"AUC": 0.989, "F1": 0.986, "MCC": 0.973, "Acc": 0.987},
    "DMN_only":            {"AUC": 0.463, "F1": 0.527, "MCC": 0.054, "Acc": 0.544},
    "Late_Fusion":         {"AUC": 0.968, "F1": 0.866, "MCC": 0.751, "Acc": 0.868},
    "Intermediate_Fusion": {"AUC": 0.897, "F1": 0.790, "MCC": 0.585, "Acc": 0.794},
}

# EEGNet LOSO results — full 77-fold experiment (one fold per subject)
EEGNET_LOSO_SUMMARY = {
    "n_subjects_effective": 77,    # 78 labelled, 1 excluded (missing .fif / QC)
    "n_folds":              77,
    "perfect_folds":        50,    # F1 = 1.0
    "chance_folds":         27,    # F1 ≈ 0.47–0.50
    "APOE_perfect":         26,    # out of 47 APOE folds
    "APOE_chance":          21,
    "APOE_n_folds":         47,
    "PICALM_perfect":       24,    # out of 30 PICALM folds
    "PICALM_chance":         6,    # remainder (4 APOE folds unaccounted = e4/e4 + e2/e4)
    "PICALM_n_folds":       30,
    "e34_detection_rate":   0.578, # 26/45 e3/e4 subjects
    "PICALM_detection_rate": 0.800,
    "subject_AUC":          0.989,
    "epoch_AUC":            0.978,
}

# Fusion bimodal subgroup — N=68 (10 subjects excluded: missing resting-state fMRI)
# The bimodal partition is defined by epoch-level fold F1 (≤0.50 = chance),
# not by subject-level correctness.
FUSION_BIMODAL_SUMMARY = {
    "n_subjects":           68,
    "n_excluded_fmri":      10,    # absent AP or PA fMRI run
    "chance_subjects":      24,    # epoch-level F1 ≈ 0.47–0.50 (marginal EEGNet confidence)
    "perfect_subjects":     44,    # epoch-level F1 = 1.0
    # Late Fusion AUC within each subgroup
    "chance_late_fusion_AUC":   0.947,
    "perfect_late_fusion_AUC":  0.963,
    # DMN null result: chance vs perfect subgroups indistinguishable on DMN
    "dmn_null_p":           0.386, # Mann-Whitney U, MW p-value
    "dmn_chance_mean":      0.383,
    "dmn_perfect_mean":     0.420,
    # What fusion does in the uncertain regime
    "dmn_shift_description": (
        "In the 24 subjects where EEGNet's epoch-level predictions were near-chance "
        "(F1≈0.47–0.50), Late Fusion raises P(PICALM_risk) from ~0.50 to ~0.70–0.75 "
        "using the DMN signal alone. 20/24 remain correctly classified; 4/24 are "
        "flipped to incorrect where the DMN signal disagreed with a borderline-correct "
        "EEGNet prediction."
    ),
}

# Kept for backwards compatibility — points to the fusion bimodal frame
BIMODAL_SUMMARY = {
    "total_subjects":       78,
    "eegnet_subjects":      77,
    "fusion_subjects":      68,
    "perfect_folds":        50,               # EEGNet LOSO (77 folds)
    "chance_folds":         27,               # EEGNet LOSO (77 folds)
    "chance_subjects_fusion": 24,             # Fusion bimodal frame (N=68)
    "perfect_subjects_fusion": 44,            # Fusion bimodal frame (N=68)
    "APOE_detection_rate":  0.578,
    "PICALM_detection_rate": 0.800,
    "chance_fold_late_fusion_AUC": 0.947,
    "chance_fold_dmn_shift_description": FUSION_BIMODAL_SUMMARY["dmn_shift_description"],
}


# ── EEGNet architecture (PyTorch) ────────────────────────────────────────────

if TORCH_AVAILABLE:
    class EEGNet(nn.Module):
        """
        EEGNet implementation as used in the research pipeline.
        Trained parameters: F1=8, D=2, F2=16, dropout=0.5, sfreq=500.
        Checkpoint keys: raw state_dict() — no wrapper dict.
        """

        def __init__(
            self,
            n_classes: int = 2,
            n_channels: int = 64,
            n_times: int = 2000,
            sfreq: int = 500,
            F1: int = 8,
            D: int = 2,
            F2: int = None,
            dropout: float = 0.5,
        ):
            super().__init__()
            if F2 is None:
                F2 = F1 * D
            temp_kern = (sfreq // 2) | 1  # nearest odd number to sfreq/2

            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, temp_kern), padding=(0, temp_kern // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
            )

            sep_kern = 16
            self.block2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, sep_kern), padding=(0, sep_kern // 2),
                          groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
            )

            n_flat = F2 * (n_times // 32)
            self.classifier = nn.Linear(n_flat, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x.unsqueeze(1)
            x = self.block1(x)
            x = self.block2(x)
            return self.classifier(x.flatten(1))

        def get_embedding(self, x: "torch.Tensor") -> "torch.Tensor":
            """Return penultimate-layer embedding (used in intermediate fusion)."""
            x = x.unsqueeze(1)
            x = self.block1(x)
            x = self.block2(x)
            return x.flatten(1)


# ── Main classifier class ────────────────────────────────────────────────────

class AlzheimerRiskClassifier:
    """
    Distinguishes APOE ε4 risk vs PICALM risk genetic profiles.

    Parameters
    ----------
    mode : str
        'simulate' — no checkpoints needed, uses research distributions
        'model'    — loads actual EEGNet .pt checkpoint
    checkpoint_path : str, optional
        Path to EEGNet state_dict .pt file (required when mode='model')
    fusion_weights_path : str, optional
        Path to pickled sklearn LogisticRegression (late fusion classifier)
    n_channels : int
        Number of EEG channels (default 64)
    n_times : int
        Epoch length in samples at 500 Hz (default 2000 = 4 seconds)
    sfreq : int
        Sampling frequency in Hz (default 500)
    """

    UNCERTAIN_THRESHOLD = 0.30  # confidence below this = uncertain/bimodal regime

    def __init__(
        self,
        mode: str = "simulate",
        checkpoint_path: str = None,
        fusion_weights_path: str = None,
        n_channels: int = 64,
        n_times: int = 2000,
        sfreq: int = 500,
    ):
        self.mode = mode
        self.n_channels = n_channels
        self.n_times = n_times
        self.sfreq = sfreq

        if mode == "model":
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch is required for model mode. Install with: pip install torch"
                )
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when mode='model'")
            self._load_model(checkpoint_path, fusion_weights_path)
        else:
            self._init_simulation()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _load_model(self, checkpoint_path: str, fusion_weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eegnet = EEGNet(
            n_classes=2,
            n_channels=self.n_channels,
            n_times=self.n_times,
            sfreq=self.sfreq,
        )
        state = torch.load(checkpoint_path, map_location=self.device)
        self.eegnet.load_state_dict(state)
        self.eegnet.eval()
        self.eegnet.to(self.device)

        self.fusion_clf = None
        if fusion_weights_path is not None:
            import pickle
            with open(fusion_weights_path, "rb") as f:
                self.fusion_clf = pickle.load(f)

    def _init_simulation(self):
        """
        Calibrate simulation from research distributions.

        The simulated EEGNet uses a linear discriminant projection over the
        three most informative EEG features identified by SHAP analysis:
          - frontal theta (APOE > PICALM, weight +)
          - posterior alpha (PICALM > APOE, weight -)
          - temporal hjorth complexity (APOE > PICALM, weight +)

        Fusion weights approximate the fitted logistic regression from the paper
        (late fusion, LR C=0.1, class_weight='balanced').
        """
        # Feature projection weights (relative importance from SHAP)
        self._eeg_weights = np.array([0.45, -0.35, 0.20])
        self._eeg_scale = 2.1  # logistic gain

        # Late fusion: [EEGNet_prob, DMN_mean_connectivity] → P(APOE_risk)
        # Approximate from paper's late fusion AUC=0.968 behaviour
        self._fusion_coef      = np.array([2.8, -1.5])  # EEG prob, DMN (negative: high DMN → PICALM)
        self._fusion_intercept = -0.55

    # ── Core prediction ─────────────────────────────────────────────────────

    def predict(
        self,
        frontal_theta: float = None,
        posterior_alpha: float = None,
        temporal_hjorth: float = None,
        dmn_connectivity: float = None,
        raw_epoch: "np.ndarray" = None,
    ) -> dict:
        """
        Run the full EEG + fMRI late fusion pipeline.

        Simulate mode — supply scalar EEG features + DMN connectivity:
            frontal_theta    : log band power at Fz/FCz (~6 Hz)
            posterior_alpha  : log band power at O1/O2/POO9h (~10 Hz)
            temporal_hjorth  : Hjorth complexity at T7/T8
            dmn_connectivity : DMN mean Pearson r (Harvard-Oxford 48 ROI)

        Model mode — supply raw epoch + DMN connectivity:
            raw_epoch        : np.ndarray (n_channels, n_times)
            dmn_connectivity : same as above

        Returns
        -------
        dict with keys:
            eeg_probability    : float — P(APOE_risk) from EEGNet alone
            fusion_probability : float — P(APOE_risk) after late fusion
            confidence         : float — EEGNet confidence [0=uncertain, 1=certain]
            regime             : str   — 'uncertain' | 'confident'
            prediction         : str   — EEGNet-only call
            fusion_prediction  : str   — fusion call
            dmn_contribution   : float — how much DMN shifted the probability
            interpretation     : str   — plain-language summary
        """
        # Step 1: EEGNet
        if self.mode == "model" and raw_epoch is not None:
            eeg_prob, confidence = self._run_eegnet(raw_epoch)
        else:
            if any(v is None for v in [frontal_theta, posterior_alpha, temporal_hjorth]):
                raise ValueError(
                    "Simulate mode requires frontal_theta, posterior_alpha, "
                    "and temporal_hjorth. See AlzheimerRiskClassifier.predict() docs."
                )
            eeg_prob, confidence = self._simulate_eegnet(
                frontal_theta, posterior_alpha, temporal_hjorth
            )

        # Step 2: Late fusion
        DMN_POPULATION_MEAN = 0.40
        if dmn_connectivity is None:
            dmn_connectivity = DMN_POPULATION_MEAN
        fusion_prob = self._run_fusion(eeg_prob, dmn_connectivity)

        # Step 3: Counterfactual DMN contribution
        # How much did the actual DMN value shift the prediction vs population mean?
        # This isolates the DMN contribution from the logistic calibration shift.
        fusion_at_dmn_mean = self._run_fusion(eeg_prob, DMN_POPULATION_MEAN)
        dmn_contribution   = fusion_prob - fusion_at_dmn_mean

        regime           = "uncertain" if confidence < self.UNCERTAIN_THRESHOLD else "confident"
        prediction       = "APOE_risk"   if eeg_prob    >= 0.5 else "PICALM_risk"
        fusion_prediction = "APOE_risk"  if fusion_prob >= 0.5 else "PICALM_risk"

        interpretation = self._interpret(
            eeg_prob, fusion_prob, dmn_contribution, regime, prediction, fusion_prediction
        )

        return {
            "eeg_probability":    round(float(eeg_prob), 4),
            "fusion_probability": round(float(fusion_prob), 4),
            "confidence":         round(float(confidence), 4),
            "regime":             regime,
            "prediction":         prediction,
            "fusion_prediction":  fusion_prediction,
            "dmn_contribution":   round(float(dmn_contribution), 4),
            "interpretation":     interpretation,
        }

    # ── Internal pipeline steps ──────────────────────────────────────────────

    def _simulate_eegnet(
        self,
        frontal_theta: float,
        posterior_alpha: float,
        temporal_hjorth: float,
    ) -> tuple:
        """
        Simulate EEGNet output from scalar EEG features.

        Uses a calibrated linear projection over the three SHAP-top features,
        scaled to approximate the bimodal confidence distribution observed in
        the real LOSO results (AUC 0.989, bimodal fold structure).
        """
        apoe_mu   = RESEARCH_DISTRIBUTIONS["APOE_risk"]
        picalm_mu = RESEARCH_DISTRIBUTIONS["PICALM_risk"]

        pooled_means = np.array([
            (apoe_mu["frontal_theta_mean"]      + picalm_mu["frontal_theta_mean"])     / 2,
            (apoe_mu["posterior_alpha_mean"]    + picalm_mu["posterior_alpha_mean"])   / 2,
            (apoe_mu["hjorth_complexity_mean"]  + picalm_mu["hjorth_complexity_mean"]) / 2,
        ])
        pooled_stds = np.array([
            (apoe_mu["frontal_theta_std"]      + picalm_mu["frontal_theta_std"])      / 2,
            (apoe_mu["posterior_alpha_std"]    + picalm_mu["posterior_alpha_std"])    / 2,
            (apoe_mu["hjorth_complexity_std"]  + picalm_mu["hjorth_complexity_std"]) / 2,
        ])

        features = np.array([frontal_theta, posterior_alpha, temporal_hjorth])
        z        = (features - pooled_means) / (pooled_stds + 1e-8)

        score    = np.dot(self._eeg_weights, z)
        eeg_prob = 1.0 / (1.0 + np.exp(-self._eeg_scale * score))

        confidence = min(abs(eeg_prob - 0.5) * 2, 1.0)
        return float(eeg_prob), float(confidence)

    def _run_eegnet(self, raw_epoch: "np.ndarray") -> tuple:
        """Run actual EEGNet on a raw (n_channels, n_times) epoch."""
        x = torch.FloatTensor(raw_epoch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits   = self.eegnet(x)
            probs    = torch.softmax(logits, dim=1)
            eeg_prob = probs[0, 0].item()  # class 0 = APOE_risk
        confidence = min(abs(eeg_prob - 0.5) * 2, 1.0)
        return eeg_prob, confidence

    def _run_fusion(self, eeg_prob: float, dmn_connectivity: float) -> float:
        """
        Late fusion: logistic regression over [EEGNet_prob, DMN_connectivity].

        If a fitted fusion_clf is loaded (model mode), use it directly.
        Otherwise fall back to the approximate simulation weights.
        """
        features = np.array([[eeg_prob, dmn_connectivity]])

        if self.mode == "model" and self.fusion_clf is not None:
            fusion_prob = self.fusion_clf.predict_proba(features)[0, 0]
        else:
            logit       = np.dot(self._fusion_coef, features[0]) + self._fusion_intercept
            fusion_prob = 1.0 / (1.0 + np.exp(-logit))

        return float(np.clip(fusion_prob, 1e-6, 1 - 1e-6))

    # ── Interpretation ───────────────────────────────────────────────────────

    def _interpret(
        self,
        eeg_prob: float,
        fusion_prob: float,
        dmn_contribution: float,
        regime: str,
        prediction: str,
        fusion_prediction: str,
    ) -> str:
        lines = []

        if regime == "confident":
            lines.append(
                f"EEGNet is confident in this classification "
                f"(P={eeg_prob:.3f}, {'well above' if eeg_prob > 0.7 else 'well below'} 0.5). "
                f"The subject's temporal EEG complexity pattern is a strong match for "
                f"{prediction}."
            )
            if abs(dmn_contribution) > 0.05:
                direction = "reinforces" if (
                    (fusion_prob > 0.5) == (eeg_prob > 0.5)
                ) else "slightly contradicts"
                lines.append(
                    f"DMN connectivity {direction} the EEG call "
                    f"(shift: {dmn_contribution:+.3f}), "
                    f"but is not decisive here — this is the expected behaviour when "
                    f"EEGNet is already certain."
                )
            else:
                lines.append(
                    "DMN connectivity is near-neutral, consistent with the paper's "
                    "finding that fMRI adds marginal value when EEGNet is confident."
                )
        else:
            lines.append(
                f"EEGNet is uncertain (P={eeg_prob:.3f}, near-chance). "
                f"This subject falls in the bimodal-uncertain regime — a pre-symptomatic "
                f"EEG signature has not yet emerged, reflecting individual variability "
                f"in onset timing, not a model failure."
            )
            if abs(dmn_contribution) > 0.08:
                lines.append(
                    f"DMN connectivity is the primary discriminator here. "
                    f"Fusion shifts P(APOE_risk) by {dmn_contribution:+.3f}, "
                    f"{'pushing toward APOE_risk' if dmn_contribution > 0 else 'pushing toward PICALM_risk'}. "
                    f"This is the complementarity argument from the paper: DMN carries "
                    f"independent variance precisely when EEG is maximally uncertain."
                )
                if prediction != fusion_prediction:
                    lines.append(
                        f"Note: fusion flipped the call from {prediction} → {fusion_prediction}. "
                        f"This reflects a case where the DMN signal overrides a borderline EEGNet prediction."
                    )
            else:
                lines.append(
                    "DMN connectivity is also near-neutral. Both modalities are ambiguous "
                    "for this subject — the classifier cannot make a high-confidence call."
                )

        return " ".join(lines)

    # ── Utility methods ──────────────────────────────────────────────────────

    def sample_synthetic_subject(self, label: str, seed: int = None) -> dict:
        """
        Sample a synthetic subject from the research-derived distributions.

        Parameters
        ----------
        label : str — 'APOE_risk' or 'PICALM_risk'
        seed  : int — optional random seed for reproducibility

        Returns
        -------
        dict of sampled feature values
        """
        if label not in RESEARCH_DISTRIBUTIONS:
            raise ValueError(f"label must be 'APOE_risk' or 'PICALM_risk', got '{label}'")

        rng = np.random.default_rng(seed)
        d   = RESEARCH_DISTRIBUTIONS[label]

        return {
            "label":            label,
            "frontal_theta":    float(rng.normal(d["frontal_theta_mean"],    d["frontal_theta_std"])),
            "posterior_alpha":  float(rng.normal(d["posterior_alpha_mean"],  d["posterior_alpha_std"])),
            "temporal_hjorth":  float(rng.normal(d["hjorth_complexity_mean"], d["hjorth_complexity_std"])),
            "dmn_connectivity": float(rng.normal(d["dmn_connectivity_mean"], d["dmn_connectivity_std"])),
        }

    @staticmethod
    def get_research_summary() -> dict:
        """Return the full results table and bimodal summary from the paper."""
        return {
            "results":       RESULTS_SUMMARY,
            "eegnet_loso":   EEGNET_LOSO_SUMMARY,
            "fusion_bimodal": FUSION_BIMODAL_SUMMARY,
            "distributions": RESEARCH_DISTRIBUTIONS,
        }

    @staticmethod
    def get_profile_description() -> dict:
        """Return plain-language descriptions of the two genetic risk profiles."""
        return {
            "APOE_risk": {
                "N": 48,
                "full_name": "APOE ε4 carriers (mostly e3/e4, N=45; e4/e4, N=1; e2/e4, N=1)",
                "mechanism": "Lipid metabolism disruption / impaired amyloid clearance",
                "age_mean":  55.65,
                "eeg_signature": (
                    "Elevated frontal theta (~6 Hz) at Fz/FCz, reduced posterior alpha "
                    "at O1/O2/POO9h, higher temporal Hjorth complexity at T7/T8. "
                    "Consistent with hippocampal-cortical stress from amyloid accumulation."
                ),
                "fmri_dmn":        "Mean connectivity μ=0.383 (slightly lower than PICALM group)",
                "eegnet_detection": "57.8% of e3/e4 subjects (26/45) — bimodal structure",
                "shap_top_feature": "dmn_mean_connectivity (0.0378) then hjorth_mobility_POO9h",
            },
            "PICALM_risk": {
                "N": 30,
                "full_name": "PICALM rs3851179 non-G/G carriers, no APOE ε4",
                "mechanism": "Disrupted clathrin-mediated endocytosis pathway",
                "age_mean":  54.87,
                "eeg_signature": (
                    "Elevated posterior alpha at O1/O2/POO9h (PICALM > APOE), "
                    "higher left-frontal spectral entropy at AF8, "
                    "lower temporal Hjorth complexity. Flatter spectral profiles."
                ),
                "fmri_dmn":        "Mean connectivity μ=0.420 (slightly higher than APOE group)",
                "eegnet_detection": "80.0% (24/30) — cleaner EEG signature",
                "shap_top_feature": "spectral_entropy_T7 and hjorth_mobility_POO9h (posterior)",
            },
        }
