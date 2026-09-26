"""Format the documented paired regime-tag contrasts; no evaluation is run."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent / "native" / "observed_regime_information_condition_by_mechanism_summary.json"
rows = json.loads(source.read_text())["paired_contrasts"]
lookup = {
    (r["size"], r["model"], r["mechanism"], r["family"], r["information"], r["reference"]): r
    for r in rows
}


def fmt(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}"


def measure(row: dict, key: str, interval: str, decimals: int) -> str:
    lo, hi = row[interval]
    return f"{fmt(row[key], decimals)} [{fmt(lo, decimals)}, {fmt(hi, decimals)}]"


out = []
for metric, label, unit, diff_key, interval_key, digits, hidden_digits in (
    ("cross-entropy", "tab:tagcontrasts", "nats", "cross_entropy_difference", "cross_entropy_difference_95_interval", 3, 4),
    ("accuracy", "tab:tagaccuracy", "percentage points", "accuracy_difference_pp", "accuracy_difference_95_interval_pp", 2, 2),
):
    out.extend(
        [
            r"\begingroup\footnotesize\setlength{\tabcolsep}{4pt}\setlength\LTleft{0pt}\setlength\LTright{0pt}",
            r"\begin{longtable}{@{}lllp{4.4cm}r@{}}",
            rf"\caption{{Within-checkpoint {metric} contrasts ({unit}) on multiregime multiclass episodes. True minus shuffled isolates aligned membership at matched input width; shuffled minus hidden captures an appended corrupted tag. Brackets are 95\% within-cell episode-bootstrap intervals (5,000 replicates), not training-seed intervals. Each mechanism-by-routing stratum has 1,440 episodes in 180 cells. O/F/C denote original/fixed/curriculum and S/M/L denote size.}}\label{{{label}}}\\",
            r"\toprule Mechanism & Routing & Model/size & True$-$shuffled [95\% interval] & Shuffled$-$hidden\\\midrule\endfirsthead",
            r"\multicolumn{5}{l}{\emph{Continued from previous page}}\\\toprule Mechanism & Routing & Model/size & True$-$shuffled [95\% interval] & Shuffled$-$hidden\\\midrule\endhead",
            r"\bottomrule\endfoot",
        ]
    )
    for mechanism in ("r_z", "g_z"):
        for family in ("persistent", "soft_gate"):
            for size in ("small", "medium", "large"):
                for model, short in (("original", "O"), ("rg_z-fixed", "F"), ("rg_z-curriculum", "C")):
                    true = lookup[(size, model, mechanism, family, "true", "shuffled")]
                    shuffled = lookup[(size, model, mechanism, family, "shuffled", "hidden")]
                    out.append(
                        " & ".join(
                            (
                                "$" + mechanism + "$",
                                "persistent" if family == "persistent" else "soft gate",
                                f"{short}/{size[0].upper()}",
                                measure(true, diff_key, interval_key, digits),
                                fmt(shuffled[diff_key], hidden_digits),
                            )
                        )
                        + r"\\"
                    )
            out.append(r"\midrule")
    out.pop()  # End-foot supplies the rule on the final longtable page.
    out.append(r"\end{longtable}\endgroup")
target = HERE / "appendix_d_tables.tex"
target.write_text("\n".join(out) + "\n")
print(f"Wrote {target} with 36 model/mechanism/routing rows")
