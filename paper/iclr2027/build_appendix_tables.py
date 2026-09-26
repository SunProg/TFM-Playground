"""Render appendix tables from the documented, rounded native-prior results.

Run from the repository root with `python3 paper/iclr2027/build_appendix_tables.py`.
This script performs no experiment and deliberately labels derived summaries as
descriptive. The source markdown carries the 36 selected BeyondArena tasks.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "native" / "native_prior_results.md"
OUTPUT = ROOT / "appendix_tables.tex"


def latex(text: str) -> str:
    return text.replace("_", r"\_\allowbreak{}")


def parse_table(section: str) -> tuple[list[str], list[dict[str, str]]]:
    raw = SOURCE.read_text()
    match = re.search(rf"^### A1 — {section}\s*$", raw, re.M)
    if not match:
        raise ValueError(section)
    lines = raw[match.end() :].splitlines()
    table = []
    for line in lines:
        if line.startswith("|"):
            table.append([cell.strip().replace("**", "") for cell in line.strip("|\n ").split("|")])
        elif table:
            break
    header = table[0]
    rows = [dict(zip(header, cells, strict=True)) for cells in table[2:]]
    if len(rows) != 36:
        raise ValueError(f"{section}: {len(rows)} tasks")
    return header, rows


_, ce = parse_table("excess_cross_entropy")
_, acc = parse_table("accuracy_gain")
_, auc = parse_table("macro_ovr_auc")
assert [r["task"] for r in ce] == [r["task"] for r in acc] == [r["task"] for r in auc]

labels = {
    "nat-o-L": "Native single-regime",
    "nat-rgF-L": "Native fixed mixture",
    "nat-rgC-L": "Native curriculum",
    "rf": "Random forest",
    "tabicl-v2": "TabICL v2",
    "tabpfn-v3": "TabPFN v3",
}
slices = [
    ("Bin. IID (26)", lambda r: r["cls"] == "bin" and r["regime"] == "IID"),
    ("Bin. group (1)", lambda r: r["cls"] == "bin" and r["regime"] == "G"),
    ("Multi IID (6)", lambda r: r["cls"] != "bin" and r["regime"] == "IID"),
    ("Multi group (2)", lambda r: r["cls"] != "bin" and r["regime"] == "G"),
    ("Multi time (1)", lambda r: r["cls"] != "bin" and r["regime"] == "T"),
]

out: list[str] = []


def add(line: str = "") -> None:
    out.append(line)


add(r"\begingroup\small\setlength\LTleft{0pt}\setlength\LTright{0pt}")
add(r"\begin{longtable}{@{}lrrr@{}}")
add(r"\caption{BeyondArena equal-task summaries from the original evaluation aggregates. Overall excess CE covers 36 tasks, binary accuracy gain 27 tasks, and multiclass accuracy gain nine tasks. Values are rounded; accuracy gains are percentage points.}\label{tab:baoverall}\\")
add(r"\toprule")
add(r"Model & Overall excess CE $\downarrow$ & Binary gain & Multiclass gain\\\midrule")
overall = [
    ("Native single-regime", "-.224", "9.4", "20.0"),
    ("Native fixed mixture", "-.221", "9.3", "20.2"),
    ("Native curriculum", "-.220", "9.2", "19.9"),
    ("Random forest", "-.225", "9.7", "21.7"),
    ("TabICL v2", "-.273", "11.2", "23.0"),
    ("TabPFN v3", "-.267", "11.0", "22.8"),
]
for row in overall:
    add(" & ".join(row) + r"\\")
add(r"\bottomrule\end{longtable}\endgroup")

for metric, rows, label, title, scale in [
    ("CE", ce, "tab:splitce", "Excess CE (nats; lower is better)", 1),
    ("accuracy", acc, "tab:splitacc", "Accuracy gain (percentage points; higher is better)", 100),
]:
    add(r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\setlength\LTleft{0pt}\setlength\LTright{0pt}")
    add(r"\begin{longtable}{@{}lrrrrr@{}}")
    add(rf"\caption{{BeyondArena split summaries, {title}. Equal-task means are calculated from task-level results rounded to three decimals and are descriptive. Grouped and temporal slices contain only one or two tasks each.}}\label{{{label}}}\\")
    add(r"\toprule")
    add("Model & " + " & ".join(name for name, _ in slices) + r"\\\midrule")
    for key, model in labels.items():
        vals = []
        for _, predicate in slices:
            chosen = [float(r[key]) for r in rows if predicate(r)]
            vals.append(f"{sum(chosen)/len(chosen)*scale:+.3f}" if scale == 1 else f"{sum(chosen)/len(chosen)*scale:+.2f}")
        add(model + " & " + " & ".join(vals) + r"\\")
    add(r"\bottomrule\end{longtable}\endgroup")

add(r"\begingroup\footnotesize\setlength\LTleft{0pt}\setlength\LTright{0pt}")
add(r"\begin{longtable}{@{}p{6.3cm}lrrr@{}}")
add(r"\caption{Selected BeyondArena task inventory. Training rows and features describe the task-level evaluation manifest; G = grouped and T = temporal. Task names follow the source results and are truncated there where necessary.}\label{tab:inventory}\\")
add(r"\toprule Task & Split & Train $n$ & $d$ & Majority\\\midrule\endfirsthead")
add(r"\toprule Task & Split & Train $n$ & $d$ & Majority\\\midrule\endhead")
add(r"\bottomrule\endfoot")
for r in ce:
    split = r["regime"]
    add(f"{latex(r['task'])} & {split} & {r['n']} & {r['d']} & {r['maj']}" + r"\\")
add(r"\end{longtable}\endgroup")

metric_info = [
    ("Excess cross-entropy (nats; lower is better)", ce, "ce"),
    ("Accuracy gain (fraction; higher is better)", acc, "acc"),
    ("Macro one-versus-rest AUC (higher is better)", auc, "auc"),
]
panels = [
    ("small/medium native", ["nat-o-S", "nat-rgF-S", "nat-rgC-S", "nat-o-M", "nat-rgF-M", "nat-rgC-M"], ["O-S", "F-S", "C-S", "O-M", "F-M", "C-M"]),
    ("large native and reference", ["nat-o-L", "nat-rgF-L", "nat-rgC-L", "tabpfn-v3", "tabicl-v2", "rf", "logreg"], ["O-L", "F-L", "C-L", "PFN3", "ICL2", "RF", "LR"]),
]
for metric_name, rows, short in metric_info:
    for panel_name, keys, headings in panels:
        add(r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}")
        add(r"\setlength\LTleft{0pt}\setlength\LTright{0pt}")
        spec = r"@{}p{3.95cm}l" + "r" * len(keys) + r"@{}"
        add(rf"\begin{{longtable}}{{{spec}}}")
        add(rf"\caption{{Per-task BeyondArena {metric_name}: {panel_name} models. O/F/C are native single-regime/fixed/curriculum; S/M/L indicate size. Results are rounded task aggregates.}}\label{{tab:{short}{'sm' if len(keys)==6 else 'large'}}}\\")
        add(r"\toprule Task & Split & " + " & ".join(headings) + r"\\\midrule")
        add(r"\endfirsthead")
        add(r"\multicolumn{" + str(len(keys) + 2) + r"}{l}{\emph{Continued from previous page}}\\\toprule")
        add(r"Task & Split & " + " & ".join(headings) + r"\\\midrule\endhead")
        add(r"\bottomrule\endfoot")
        for index, r in enumerate(rows):
            if index in (27,):
                add(r"\midrule")
            vals = [r[k] for k in keys]
            add(latex(r["task"]) + " & " + r["regime"] + " & " + " & ".join(vals) + r"\\")
        add(r"\end{longtable}\endgroup")

OUTPUT.write_text("\n".join(out) + "\n")
print(f"Wrote {OUTPUT} with {len(ce)} tasks and {len(out)} lines")
