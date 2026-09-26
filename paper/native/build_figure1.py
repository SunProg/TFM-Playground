"""Build the conceptual Figure 1 for the native-prior multiregime paper."""

from html import escape
from pathlib import Path


OUT = Path(__file__).parent
W, H = 1680, 1320
parts: list[str] = []


def raw(markup: str) -> None:
    parts.append(markup)


def rect(x, y, w, h, fill="#fff", stroke="none", sw=1, rx=0, **attrs):
    extra = " ".join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in attrs.items())
    raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def txt(x, y, value, size=24, color="#26384a", weight=400, anchor="start", **attrs):
    extra = " ".join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in attrs.items())
    raw(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" {extra}>{escape(value)}</text>')


def equation_text(x, y, parts, size=20, color="#26384a", weight=600, anchor="middle", math_font=False):
    family = ' font-family="STIX Two Math, Cambria Math, Times New Roman, serif"' if math_font else ""
    content = []
    for part in parts:
        if isinstance(part, tuple):
            base, sub = part
            sub_markup = (
                f'<tspan font-size="75%" baseline-shift="sub">{escape(sub)}</tspan>'
                if sub is not None else ""
            )
            var_family = "" if math_font else ' font-family="Times New Roman, Georgia, serif"'
            content.append(
                f'<tspan{var_family} font-style="italic">'
                f'{escape(base)}{sub_markup}</tspan>'
            )
        else:
            content.append(f'<tspan>{escape(part)}</tspan>')
    raw(
        f'<text x="{x}" y="{y}"{family} xml:space="preserve" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{"".join(content)}</text>'
    )


def line(x1, y1, x2, y2, color="#98a8b7", sw=2, dash=None, marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')


def circle(cx, cy, r, fill, stroke="none", sw=1):
    raw(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def panel(x, y, width, height, title, letter):
    rect(x, y, width, height, "#f8fafc", "#dce5ec", 2, 22)
    rect(x + 22, y + 20, 44, 40, "#e7eef5", "none", 0, 12)
    txt(x + 44, y + 48, letter, 23, "#36566e", 700, "middle")
    txt(x + 82, y + 48, title, 27, "#1f3447", 700)


raw(f'''<svg xmlns="http://www.w3.org/2000/svg" width="7.2in" height="5.66in" viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#73879a"/></marker>
  <marker id="arrow-teal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#278b88"/></marker>
  <style>text {{ font-family: Arial, Helvetica, sans-serif; }}</style>
</defs>
<rect width="{W}" height="{H}" fill="#ffffff"/>
''')

# Restrained, color-blind-friendly palette: colors encode regimes, not classes.
navy = "#294e69"
muted = "#5d7182"
teal = "#268b88"
orange = "#d58a36"
violet = "#7968a9"
pale_teal = "#e5f3f1"
pale_orange = "#fbf0df"
pale_violet = "#eeeaf6"
border = "#d7e0e7"

panel(32, 40, 790, 610, "Sample a task", "A")
panel(858, 40, 790, 610, "Score and label mechanisms", "B")
panel(32, 670, 790, 610, "Route rows to regimes", "C")
panel(858, 670, 790, 610, "Pretrain and predict", "D")

# Panel A: one generated table split into labelled support and unlabelled query.
rect(62, 130, 730, 88, "#edf3f7", "#d8e2e9", 1.5, 15)
rect(82, 148, 54, 52, "#ffffff", "#cad8e2", 1.5, 12)
circle(98, 165, 4, teal); circle(120, 165, 4, orange)
circle(98, 184, 4, violet); circle(120, 184, 4, navy)
txt(154, 165, "Native TabICL SCM prior", 24, navy, 700)
txt(154, 197, "generates features X and continuous score(s)", 20, muted)
line(427, 222, 427, 249, "#8194a4", 2.2, marker="arrow")

rect(62, 260, 730, 350, "#ffffff", border, 1.6, 16)
txt(86, 296, "One synthetic episode with multiple regimes", 23, navy, 700)
txt(86, 326, "Regime colors are generator labels; they are not model inputs.", 19, muted)
rect(84, 344, 686, 34, "#f1f5f8", "none", 0, 7)
for x, label in [(175, "x₁"), (330, "x₂"), (500, "Y"), (690, "split")]:
    txt(x, 368, label, 18, muted, 700, "middle")
table_rows = [
    (402, pale_teal, teal, "−0.8", "0.2", "0", "support"),
    (434, pale_teal, teal, "−0.4", "0.7", "1", "support"),
    (466, pale_orange, orange, "0.1", "−0.6", "0", "support"),
    (498, pale_orange, orange, "0.5", "0.3", "1", "support"),
    (530, pale_violet, violet, "0.8", "−0.1", "—", "query"),
    (562, pale_violet, violet, "1.1", "0.6", "—", "query"),
]
for y, wash, regime_color, x1, x2, label, split in table_rows:
    rect(84, y - 22, 686, 28, wash, "none", 0, 6)
    rect(84, y - 22, 6, 28, regime_color, "none", 0, 3)
    txt(175, y, x1, 18, "#354b5e", 400, "middle", font_family="monospace")
    txt(330, y, x2, 18, "#354b5e", 400, "middle", font_family="monospace")
    txt(500, y, label, 19, navy if label != "—" else "#a5b1bc", 700, "middle")
    txt(690, y, split, 17, muted, 600, "middle")
txt(427, 592, "Support Y is given; query Y is hidden.", 19, navy, 700, "middle")

# Panel B: explicit score values, separately sampled cut points, and output labels.
txt(1253, 119, "Example numbers are schematic; class IDs may be permuted.", 17, muted, 500, "middle")

def score_band(x, y, width, height, cut1, cut2, score):
    cuts = (cut1, cut2)
    labels = ("0", "1", "2")
    fills = ("#e7edf2", "#dce5ec", "#e7edf2")
    start = 0.0
    for end, label, fill in zip((cut1, cut2, 1.0), labels, fills):
        part_w = width * (end - start)
        rect(x + width * start, y, part_w, height, fill, "#b8c7d2", 1, 3)
        txt(x + width * (start + end) / 2, y + height * 0.68, label, 14, navy, 700, "middle")
        start = end
    for cut in cuts:
        line(x + width * cut, y - 3, x + width * cut, y + height + 3, "#71879a", 1.5, "3 2")
    marker_x = x + width * score
    line(marker_x, y - 8, marker_x, y + height + 2, "#203f56", 2)
    circle(marker_x, y - 8, 4, "#203f56")

def mechanism_row(y, regime, score_text, score_value, cut1, cut2, class_id, fill, edge, ink):
    rect(894, y, 712, 58, fill, edge, 1, 10)
    rect(904, y + 13, 93, 31, "#ffffff", edge, 1, 8)
    txt(950, y + 34, regime, 16, ink, 700, "middle")
    rect(1005, y + 13, 38, 31, "#ffffff", edge, 1, 8)
    txt(1024, y + 34, "X", 16, navy, 700, "middle")
    line(1047, y + 29, 1064, y + 29, "#73879a", 1.5, marker="arrow")
    rect(1070, y + 10, 132, 36, "#ffffff", edge, 1, 8)
    equation_text(1136, y + 33, score_text, 17, ink, 600, "middle", math_font=True)
    line(1205, y + 29, 1219, y + 29, "#73879a", 1.5, marker="arrow")
    score_band(1224, y + 16, 248, 26, cut1, cut2, score_value)
    line(1474, y + 29, 1481, y + 29, "#73879a", 1.5, marker="arrow")
    rect(1483, y + 13, 112, 31, "#ffffff", edge, 1, 8)
    txt(1539, y + 34, f"Y = {class_id}", 16, navy, 700, "middle")

# r_z has a different SCM score column and a new label assignment draw per regime.
rect(880, 138, 746, 232, "#ffffff", "#b9d8d5", 1.7, 16)
rect(900, 153, 76, 34, pale_teal, "none", 0, 10)
equation_text(938, 177, [("r", "z")], 22, "#176e6b", 700, "middle", math_font=True)
txt(992, 177, "SCM gives each regime its own score and cut-point draw", 19, navy, 700)
mechanism_row(196, "Regime 1", [("s", "1"), "(", ("X", None), ") = .62"], .62, .30, .80, 1, pale_teal, "#c9e1de", "#176e6b")
mechanism_row(261, "Regime 2", [("s", "2"), "(", ("X", None), ") = .30"], .30, .20, .40, 1, pale_teal, "#c9e1de", "#176e6b")
txt(1253, 351, "Rule settings match; sampled cut points are not shared.", 16, muted, 500, "middle")

# g_z computes one score and branches it through different regime-specific mappings.
rect(880, 385, 746, 239, "#ffffff", "#ead5b6", 1.7, 16)
rect(900, 400, 76, 34, pale_orange, "none", 0, 10)
equation_text(938, 424, [("g", "z")], 22, "#9a5e14", 700, "middle", math_font=True)
txt(992, 424, "One shared score; each regime uses its own cut points", 18, navy, 700)

rect(900, 486, 55, 36, "#ffffff", "#efdec4", 1, 8)
txt(927, 510, "X", 17, navy, 700, "middle")
line(959, 504, 980, 504, "#73879a", 1.7, marker="arrow")
rect(987, 479, 152, 50, pale_orange, "#efdec4", 1.2, 9)
txt(1063, 499, "one shared score", 15, "#9a5e14", 700, "middle")
equation_text(1063, 518, [("s", None), "(", ("X", None), ") = .62"], 17, "#9a5e14", 600, "middle", math_font=True)
line(1139, 504, 1156, 504, "#73879a", 1.7)
line(1156, 474, 1156, 562, "#73879a", 1.7)
line(1156, 474, 1170, 474, "#73879a", 1.7, marker="arrow")
line(1156, 562, 1170, 562, "#73879a", 1.7, marker="arrow")

for y, regime, c1, c2, result in [
    (447, "Regime 1", .30, .80, 1),
    (535, "Regime 2", .20, .40, 2),
]:
    rect(1175, y, 88, 52, "#ffffff", "#efdec4", 1, 8)
    txt(1219, y + 32, regime, 14, "#9a5e14", 700, "middle")
    score_band(1272, y + 13, 211, 25, c1, c2, .62)
    line(1484, y + 26, 1490, y + 26, "#73879a", 1.5, marker="arrow")
    rect(1492, y + 10, 112, 31, "#ffffff", "#efdec4", 1, 8)
    txt(1548, y + 31, f"Y = {result}", 16, navy, 700, "middle")
txt(1253, 610, "Same score (.62); different cuts give different labels.", 16, muted, 500, "middle")

# Panel C: routing is independent of the score/label mechanism above.
raw('<g transform="translate(0 10)">')
txt(427, 737, "Either routing method can pair with either mechanism.", 19, muted, 500, "middle")

rect(58, 746, 360, 504, "#ffffff", "#b9d8d5", 1.7, 16)
rect(80, 766, 116, 36, pale_teal, "none", 0, 10)
txt(138, 791, "soft gate", 20, "#176e6b", 700, "middle")
txt(80, 837, "Observed x₁ sets the regime.", 21, navy, 700)
txt(80, 869, "Sort support x₁; split its ranks into bins.", 17, muted)
txt(80, 895, "Query x₁ uses those same boundaries.", 17, muted)
txt(238, 956, "ordered x₁", 17, muted, 600, "middle")
line(103, 1000, 375, 1000, "#73879a", 2)
line(193, 987, 193, 1013, "#a4b2bd", 1.6, "4 4")
line(284, 987, 284, 1013, "#a4b2bd", 1.6, "4 4")
for x, color in [(121, teal), (145, teal), (169, teal), (214, orange), (238, orange), (262, orange), (305, violet), (329, violet), (353, violet)]:
    circle(x, 1000, 8, color)
txt(238, 1042, "support ranks define the bins", 16, "#176e6b", 600, "middle")
rect(82, 1101, 312, 122, "#f3f8f7", "#d4e6e3", 1.2, 12)
txt(102, 1132, "Support x₁  →  assign Z", 17, navy, 600)
txt(102, 1164, "Query x₁     →  same Z cuts", 17, navy, 600)
txt(102, 1201, "Z is determined by observed X₁.", 16, muted)

rect(434, 746, 360, 504, "#ffffff", "#d7d0e7", 1.7, 16)
rect(456, 766, 126, 36, pale_violet, "none", 0, 10)
txt(519, 791, "persistent", 20, "#61518d", 700, "middle")
txt(456, 837, "Group identity sets the regime.", 21, navy, 700)
txt(456, 869, "Rows in a group keep one assignment.", 17, muted)
txt(456, 895, "The group ID is not included in X.", 17, muted)
raw('<g transform="translate(0 -16)">')
txt(614, 956, "latent groups", 17, muted, 600, "middle")
for x, color, label in [(506, teal, "Z=1"), (614, orange, "Z=2"), (722, violet, "Z=3")]:
    line(x, 977, x, 1051, "#bbc7d0", 1.5)
    circle(x, 990, 10, color)
    circle(x, 1038, 10, color)
    txt(x, 1076, label, 15, "#61518d", 600, "middle")
raw('</g>')
rect(458, 1101, 312, 122, "#f6f4fa", "#e2dced", 1.2, 12)
txt(478, 1132, "same group  →  same regime", 17, navy, 600)
txt(478, 1164, "Group ID is not included in X.", 16, muted)
txt(478, 1201, "Membership stays latent.", 16, muted)
raw('</g>')

# Panel D: training mixtures and the complete support/query conditioning.
raw('<g transform="translate(0 10)">')
txt(1253, 737, "Same in-context prediction task for all schedules", 19, muted, 500, "middle")

def schedule_card(x, label, detail, kind):
    rect(x, 746, 228, 126, "#ffffff", border, 1.4, 14)
    txt(x + 18, 778, label, 20, navy, 700)
    txt(x + 18, 805, detail, 16, muted)
    bx, by, bw, bh = x + 18, 837, 192, 12
    rect(bx, by, bw, bh, "#e7edf2", "none", 0, 6)
    if kind == "fixed":
        rect(bx + bw * .7, by, bw * .3, bh, "#7c9ab0", "none", 0, 6)
    elif kind == "curriculum":
        raw(f'<path d="M {bx} {by + 11} L {bx} {by + 9} L {bx + bw} {by} L {bx + bw} {by + 12} Z" fill="#7c9ab0" opacity="0.9"/>')

schedule_card(883, "Native control", "single-regime tasks", "native")
schedule_card(1125, "Fixed mixture", "30% multiregime", "fixed")
schedule_card(1367, "Curriculum", "share ramps 0 → 50%", "curriculum")

txt(1253, 913, "Prediction uses labelled examples plus the query features", 20, navy, 700, "middle")

rect(883, 949, 184, 142, "#ffffff", border, 1.4, 13)
txt(975, 977, "Model input", 19, navy, 700, "middle")
rect(898, 992, 154, 36, pale_teal, "none", 0, 8)
equation_text(975, 1016, ["support: ", ("X", "s"), ", ", ("Y", "s")], 17, navy, 600, "middle")
rect(898, 1038, 154, 36, "#edf3f7", "none", 0, 8)
equation_text(975, 1062, ["query: ", ("X", "q"), " only"], 17, navy, 600, "middle")
line(1072, 1020, 1094, 1020, "#8194a4", 2, marker="arrow")
rect(1102, 966, 194, 108, "#eaf1f5", "#cad8e2", 1.4, 14)
txt(1199, 1009, "NanoTabPFN", 21, navy, 700, "middle")
txt(1199, 1041, "in-context predictor", 16, muted, 500, "middle")
line(1301, 1020, 1323, 1020, "#8194a4", 2, marker="arrow")
rect(1332, 949, 272, 142, "#ffffff", border, 1.4, 13)
txt(1468, 978, "Query prediction", 19, navy, 700, "middle")
equation_text(1468, 1016, [("p", None), "(", ("Y", "q"), " ∣ ", ("X", "s"), ", ", ("Y", "s"), ", ", ("X", "q"), ")"], 20, navy, 500, "middle", math_font=True)
txt(1468, 1052, "query labels are not supplied", 16, muted, 500, "middle")

rect(883, 1110, 721, 93, "#f3f6f8", "#d5dfe6", 1.3, 13)
txt(905, 1143, "Primary prediction input omits the regime ID.", 18, navy, 700)
txt(905, 1175, "Soft-gate membership may be inferred from X; persistent group membership remains latent.", 16, muted)
raw('</g>')

raw('</svg>')
svg_path = OUT / "figure1_native_multiregime.svg"
svg_path.write_text("\n".join(parts), encoding="utf-8")
print(svg_path)
