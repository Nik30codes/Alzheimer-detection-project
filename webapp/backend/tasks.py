"""Registry of the models the API can serve.

Nothing about any particular task is hardcoded elsewhere. Enabling the four-stage
model later is a matter of flipping `enabled` to True -- the routes, the React UI and
the Grad-CAM path all read the class list from here.

Every displayed number is READ FROM THE RECORDED RESULT JSON, never typed in. That is
deliberate: an earlier version of this app displayed "four-way accuracy is around 60%",
a figure that later turned out to be the scanner-cohort shortcut rather than diagnostic
skill. Numbers living in a template go stale silently; numbers read from the
experiment's own output cannot.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(ROOT, "reports")

CLASS_LABELS = {
    "CN": "Cognitively Normal",
    "AD": "Alzheimer's Disease",
    "EMCI": "Early Mild Cognitive Impairment",
    "LMCI": "Late Mild Cognitive Impairment",
}

TASKS = {
    "ad_vs_cn": {
        "label": "Alzheimer's vs Cognitively Normal",
        "short": "AD vs CN",
        "classes": ["CN", "AD"],          # index order must match training
        "arch": "mobilenetv2",
        "checkpoint": "mobilenetv2_ADvsCN.pt",
        # CROSS-VALIDATED metrics, not the single-split ones. The single 75-subject
        # split reported 82.7% / AUC 0.9055; cross-validating the identical
        # configuration over all 501 subjects gives 70.9% / 0.7845, and the two AUC
        # intervals do not overlap. The split was simply favourable. Showing the
        # optimistic number to a visitor would misrepresent the model.
        "metrics_file": "mobilenetv2_ADvsCN_cv_result.json",
        "positive_class": "AD",
        "status": "validated",
        "enabled": True,
        "description": (
            "Trained on 501 ADNI subjects split evenly across two scanner generations, "
            "so the model cannot separate the classes by recognising the scanner. "
            "Figures below are 5-fold cross-validated over all 501 subjects, not a "
            "single held-out split — an earlier single-split estimate ran about 12 "
            "points optimistic. Also validated by training on one scanner generation "
            "and testing on the other."
        ),
    },
    "four_stage": {
        "label": "Four-stage classification",
        "short": "CN / EMCI / LMCI / AD",
        "classes": ["CN", "AD", "EMCI", "LMCI"],
        "arch": "mobilenetv2",
        "checkpoint": "mobilenetv2_v3go2_f1_init-mobilenetv2_ADvsCN.pt",
        "metrics_file": "mobilenetv2_v3go2_f1_init-mobilenetv2_ADvsCN_result.json",
        "positive_class": None,
        "status": "experimental",
        # OFF until four-stage clears its 36.6% baseline significantly. Separating early
        # from late MCI is defined in ADNI by a memory-test cutoff rather than anatomy,
        # and the model is not yet reliably better than guessing. Serving stage
        # predictions now would look authoritative without being so.
        "enabled": False,
        "description": (
            "Predicts all four stages. Currently NOT reliably better than always "
            "guessing the most common class."
        ),
    },
}


def load_metrics(task):
    """Pull the honest numbers for a task straight from its result JSON."""
    path = os.path.join(REPORTS, task["metrics_file"])
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        r = json.load(f)

    out = {"source_file": task["metrics_file"]}
    if "roc_auc" in r:                                   # binary task
        out.update({
            "accuracy": r.get("accuracy"),
            "accuracy_ci": r.get("accuracy_95CI"),
            "auc": r.get("roc_auc"),
            "auc_ci": r.get("roc_auc_95CI"),
            "baseline": r.get("majority_baseline"),
            "n_test": r.get("n_test_subjects"),
            "threshold": r.get("decision_threshold"),
            "verdict": r.get("verdict"),
        })
        # Confusion matrix + macro-F1, recomputed by scripts/compute_cv_confusion.py
        # from the exact out-of-fold predictions behind the headline above (same
        # threshold, same subjects) -- additive only, never overwrites the file above.
        conf_path = os.path.join(REPORTS, "mobilenetv2_ADvsCN_cv_confusion.json")
        if os.path.exists(conf_path):
            with open(conf_path) as cf:
                conf = json.load(cf)
            out.update({
                "confusion_matrix": conf.get("confusion_matrix"),
                "confusion_classes": conf.get("classes"),
                "macro_f1": conf.get("macro_f1"),
            })
    else:                                                # multi-class task
        out.update({
            "accuracy": r.get("subject_level_accuracy"),
            "macro_f1": r.get("subject_level_macro_f1"),
            "baseline": 0.366,
            "n_test": 93,
            "verdict": "Not significantly above baseline",
        })
    return out


def enabled_tasks():
    return {k: v for k, v in TASKS.items() if v.get("enabled")}


def get_task(task_id):
    t = TASKS.get(task_id)
    return t if (t and t.get("enabled")) else None
