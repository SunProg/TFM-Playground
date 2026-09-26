"""Build the IID and valid target-site adaptation tables from saved results."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "native" / "heart_sites_group_feature.md"
OUTPUT = HERE / "appendix_f_table.tex"
ADAPTATION = HERE.parent / "native" / "heart_sites_adaptation_summary.json"


def section_rows(section: str) -> dict[str, tuple[str, str]]:
    text = SOURCE.read_text().split(f"## {section}", 1)[1]
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| model |"))
    table = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("| \n").split("|")]
        table.append((cells[0], cells[3], cells[4]))
    if len(table) != 20:
        raise ValueError(f"Expected 20 models in {section}, found {len(table)}")
    return {model: (hidden, given) for model, hidden, given in table}


iid = section_rows("1. IID folds")
adaptation_rows = json.loads(ADAPTATION.read_text())
adaptation = {}
for row in adaptation_rows:
    name = row["model"].replace("native-", "nat-", 1)
    if name in adaptation:
        raise ValueError(f"Duplicate adaptation model: {name}")
    if any(row[f"{condition}_folds"] != 15 for condition in ("site_hidden", "site_shuffled", "site_true")):
        raise ValueError(f"Incomplete 15-fold adaptation result: {name}")
    adaptation[name] = row
if iid.keys() != adaptation.keys():
    raise ValueError(f"IID/adaptation model sets differ: {iid.keys() ^ adaptation.keys()}")

def label(model: str) -> str:
    if model.startswith("nat-"):
        return model.replace("nat-original-", "Native single-regime, ").replace("nat-rg_z-fixed-", "Native fixed, ").replace("nat-rg_z-curriculum-", "Native curriculum, ").replace("small", "2 layers").replace("medium", "4 layers").replace("large", "6 layers")
    return {
        "rf": "Random forest",
        "logreg": "Logistic regression",
        "hgb": "Histogram gradient boosting",
        "catboost": "CatBoost",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "tabpfn-v2.2": "TabPFN v2.2",
        "tabpfn-v2.6": "TabPFN v2.6",
        "tabpfn-v3": "TabPFN v3",
        "tabicl-v1": "TabICL v1",
        "tabicl-v2": "TabICL v2",
    }[model]


order = [
    *[f"nat-{family}-{size}" for size in ("small", "medium", "large") for family in ("original", "rg_z-fixed", "rg_z-curriculum")],
    "rf", "logreg", "hgb", "catboost", "xgboost", "lightgbm",
    "tabicl-v1", "tabicl-v2", "tabpfn-v2.2", "tabpfn-v2.6", "tabpfn-v3",
]
out = [
    r"\begin{table}[htbp]\centering\small",
    r"\caption{Merged heart-site excess CE under repeated stratified IID folds (five folds, two repeats). All sites occur in support. Hidden uses 13 clinical attributes; given adds the observed site. Lower values are better. TabICL v1.1 was not evaluated in this case study.}\label{tab:heart_iid}",
    r"\begin{tabular}{@{}lrr@{}}\toprule",
    r"Model & Site hidden & Site given\\\midrule",
]
for model in order:
    out.append(label(model) + " & " + " & ".join(iid[model]) + r"\\")
out += [
    r"\bottomrule\end{tabular}\end{table}",
    r"\begin{table}[htbp]\centering\small",
    r"\caption{Target-site adaptation excess CE over 15 matched folds: five stratified 20-shot draws for each of three target sites. Support also contains all rows from the other two sites; queries are the remaining target-site rows. Shuffled permutes support-site tags while keeping tag counts and query tags fixed. True and shuffled have equal input width. Lower CE is better; negative true-minus-shuffled values favor aligned site tags.}\label{tab:heart_adapt}",
    r"\begin{tabular}{@{}lrrrr@{}}\toprule",
    r"Model & Hidden & Shuffled & True & True$-$shuffled\\\midrule",
]
for model in order:
    row = adaptation[model]
    values = (
        row["site_hidden_excess_cross_entropy"],
        row["site_shuffled_excess_cross_entropy"],
        row["site_true_excess_cross_entropy"],
        row["true_minus_shuffled_excess_cross_entropy"],
    )
    out.append(label(model) + " & " + " & ".join(f"{value:+.4f}" for value in values) + r"\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
OUTPUT.write_text("\n".join(out) + "\n")
print(f"Wrote {OUTPUT} with {len(order)} models")
