"""Render ``report.md`` as a LaTeX PDF with a one-page health snapshot.

    python -m risklib pdf reports/report.md [--config portfolio.yaml] [--no-compile]

or set ``output.pdf: true`` in the config to build it at the end of a report run.

Page 1 is a health snapshot: eight headline tiles and a traffic-light
checklist, each check showing its value, the rule that coloured it and a short
reading. Page 2 summarizes VaR/ES, risk contributors, exposures, sectors,
stress and backtest results. The remaining pages render the full report with
charts, and every checklist item links to its detail section.

The module only reads ``report.md`` and the charts beside it, so the PDF
always matches the Markdown. Checklist thresholds are in
:data:`DEFAULT_THRESHOLDS` and can be overridden under ``pdf.thresholds``.
Compilation uses ``pdflatex`` or ``latexmk`` when one is on the PATH;
otherwise only ``report.tex`` is written.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

# Health-checklist thresholds (fractions, not percent).
DEFAULT_THRESHOLDS: dict[str, dict] = {
    "var_limit": {"limit": 0.05, "amber_at": 0.80},  # 1-day 99% VaR vs limit; amber at 80% of it
    "top_risk_share": {"amber": 0.20, "red": 0.30},  # largest holding's share of VaR
    "effective_n_risk": {"amber": 10.0, "red": 5.0},  # effective number of positions by risk
    "top_sector_risk": {"amber": 0.40, "red": 0.60},  # largest sector's share of VaR
    "beta_mkt": {"lo": 0.8, "hi": 1.2, "lo2": 0.5, "hi2": 1.5},  # green inside [lo, hi], amber inside [lo2, hi2]
    "worst_stress": {"amber": -0.20, "red": -0.35},  # worst historical replay
    "drawdown_1y": {"amber": -0.15, "red": -0.25},  # trailing 1-year max drawdown
    "vol_regime": {"amber": 1.10, "red": 1.30},  # FHS VaR / historical VaR
    "proxy_weight": {"amber": 0.10, "red": 0.25},  # weight in holdings > 50% proxy-backfilled
}


# ---------------------------------------------------------------------------
# Markdown parsing (the subset report.py emits)
# ---------------------------------------------------------------------------
@dataclass
class Block:
    kind: str  # heading | para | table | image | quote | rule
    text: str = ""
    level: int = 0
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    path: str = ""


def parse_markdown(md: str) -> list[Block]:
    lines = md.splitlines()
    out: list[Block] = []
    i, para = 0, []

    def flush():
        if para:
            out.append(Block("para", " ".join(s.strip() for s in para)))
            para.clear()

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            flush(); i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            flush(); out.append(Block("heading", m.group(2).strip(), level=len(m.group(1)))); i += 1; continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m:
            flush(); out.append(Block("image", m.group(1), path=m.group(2))); i += 1; continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?[\s:\-|]+\|?$", lines[i + 1].strip()):
            flush()
            hdr = [c.strip() for c in s.strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append(Block("table", header=hdr, rows=rows)); i = j; continue
        if s.startswith(">"):
            flush(); out.append(Block("quote", s.lstrip("> ").strip())); i += 1; continue
        if re.match(r"^-{3,}$", s):
            flush(); out.append(Block("rule")); i += 1; continue
        para.append(s); i += 1
    flush()
    return out


def _bare(heading: str) -> str:
    """Heading text without a leading section number such as "4. "."""
    return re.sub(r"^\d+\.\s*", "", heading)


class Report:
    """Parsed report.md with tables indexed by their nearest heading."""

    def __init__(self, md: str):
        self.md = md
        self.blocks = parse_markdown(md)
        self.tables: list[tuple[str, Block]] = []
        heading = ""
        for b in self.blocks:
            if b.kind == "heading":
                heading = b.text
            elif b.kind == "table":
                self.tables.append((heading, b))

    def table(self, prefix: str) -> Block | None:
        """First table under a heading starting with ``prefix`` (section numbers optional)."""
        for h, b in self.tables:
            if h.startswith(prefix) or _bare(h).startswith(prefix):
                return b
        return None

    def rows(self, prefix: str) -> list[dict]:
        b = self.table(prefix)
        if b is None:
            return []
        return [dict(zip(b.header, r)) for r in b.rows]

    def row(self, prefix: str, key: str, col0: str | None = None) -> dict | None:
        for r in self.rows(prefix):
            k = r.get(col0) if col0 else next(iter(r.values()))
            if k is not None and k.startswith(key):
                return r
        return None

    def search(self, pattern: str, cast=str, default=None):
        m = re.search(pattern, self.md)
        if not m:
            return default
        try:
            return cast(m.group(1).replace(",", ""))
        except ValueError:
            return default


def _f(x) -> float | None:
    try:
        return float(str(x).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# LaTeX escaping and formatting
# ---------------------------------------------------------------------------
_UNI = {
    "→": r"$\to$", "−": r"$-$", "–": "--", "—": "---", "≈": r"$\approx$", "≤": r"$\le$", "≥": r"$\ge$",
    "×": r"$\times$", "√": r"$\surd$", "σ": r"$\sigma$", "ν": r"$\nu$", "γ": r"$\gamma$", "α": r"$\alpha$",
    "β": r"$\beta$", "λ": r"$\lambda$", "μ": r"$\mu$", "ρ": r"$\rho$", "Σ": r"$\Sigma$", "δ": r"$\delta$",
    "²": r"\textsuperscript{2}", "₂": r"$_2$", "±": r"$\pm$", "…": r"\ldots{}", "’": "'", "‘": "`",
    "“": "``", "”": "''", "é": r"\'e", "ü": r'\"u', "ö": r'\"o', "·": r"$\cdot$", "†": r"$^\dagger$",
}
_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(s: str) -> str:
    out = []
    for ch in str(s):
        if ch in _SPECIAL:
            out.append(_SPECIAL[ch])
        elif ch in _UNI:
            out.append(_UNI[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append("?")  # unknown non-ASCII never breaks the build
    return "".join(out)


def inline(s: str) -> str:
    """Markdown inline code and bold to LaTeX, with escaping."""
    parts = re.split(r"(`[^`]+`)", s)
    res = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            code = tex_escape(p[1:-1]).replace(r"\_", r"\_\allowbreak{}").replace(".", r".\allowbreak{}")
            res.append(r"\texttt{" + code + "}")
            continue
        for sg in re.split(r"(\*\*[^*]+\*\*)", p):
            if sg.startswith("**") and sg.endswith("**") and len(sg) > 4:
                res.append(r"\textbf{" + tex_escape(sg[2:-2]) + "}")
            else:
                res.append(tex_escape(sg))
    return "".join(res)


def num_cell(s: str) -> str:
    """Escape a table cell, typesetting a leading minus as a true minus."""
    s = s.strip()
    if re.match(r"^-\d", s):
        return r"$-$" + tex_escape(s[1:])
    return inline(s)


def _is_num(s: str) -> bool:
    return bool(re.match(r"^-?[\d,]*\.?\d+(e-?\d+)?%?$", s.strip())) or s.strip() == ""


def pct(x, d=1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}\\%"


def signed_pct(x, d=1) -> str:
    if x is None:
        return "n/a"
    return (r"$-$" if x < 0 else "") + f"{abs(100 * x):.{d}f}\\%"


def money(x) -> str:
    return "n/a" if x is None else r"\$" + f"{x:,.0f}"


def label_for(heading: str) -> str:
    return "sec:" + re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:48]


def heading_labels(rep: Report) -> dict[str, str]:
    """A unique LaTeX label for every heading."""
    out, used = {}, set()
    for b in rep.blocks:
        if b.kind == "heading" and b.text not in out:
            lab, k = label_for(b.text), 2
            while lab in used:
                lab, k = f"{label_for(b.text)}-{k}", k + 1
            used.add(lab)
            out[b.text] = lab
    return out


def resolve_link(prefix: str, labels: Mapping[str, str]) -> str:
    """Label of the first heading starting with ``prefix`` ('' if none)."""
    return next((lab for h, lab in labels.items() if h.startswith(prefix) or _bare(h).startswith(prefix)), "")


# ---------------------------------------------------------------------------
# Health snapshot
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    value: str
    status: str  # green | amber | red | info
    rule: str
    reading: str
    link: str = ""  # heading prefix of the detail section


def _band(x, amber, red, higher_is_worse=True) -> str:
    if x is None:
        return "info"
    if higher_is_worse:
        return "red" if x >= red else "amber" if x >= amber else "green"
    return "red" if x <= red else "amber" if x <= amber else "green"


def extract(rep: Report) -> dict:
    """Snapshot numbers read from the report tables (None where absent)."""
    d: dict = {}
    d["name"] = rep.search(r"(?m)^# Risk report: (.+)$", str, "Portfolio")
    d["start"] = rep.search(r"Data: (\d{4}-\d{2}-\d{2})")
    d["end"] = rep.search(r"→ (\d{4}-\d{2}-\d{2})")
    d["n_days"] = rep.search(r"\((\d+) trading days", int)
    d["n_assets"] = rep.search(r"trading days, (\d+) holdings", int)
    d["value"] = rep.search(r"Value ([\d,]+)", float)
    d["generated"] = rep.search(r"Generated ([\d\- :]+)\.")

    def var_row(prefix, h=1, a=0.99):
        for r in rep.rows("VaR / ES"):
            if r.get("method", "").startswith(prefix) and _f(r.get("horizon")) == h and abs((_f(r.get("alpha")) or 0) - a) < 1e-9:
                return r
        return None

    fhs = var_row("Filtered historical") or var_row("Historical")
    d["var_method"] = "FHS" if var_row("Filtered historical") else "Historical"
    d["var"] = _f(fhs["VaR"]) if fhs else None
    d["es"] = _f(fhs["ES"]) if fhs else None
    d["var_lo"] = _f(fhs.get("VaR_lo")) if fhs else None
    d["var_hi"] = _f(fhs.get("VaR_hi")) if fhs else None
    d["var_usd"] = _f(fhs.get("VaR_$")) if fhs else None
    d["es_usd"] = _f(fhs.get("ES_$")) if fhs else None
    hs = var_row("Historical")
    nm = var_row("Parametric normal")
    d["var_hs"] = _f(hs["VaR"]) if hs else None
    d["var_norm"] = _f(nm["VaR"]) if nm else None
    d["es_norm"] = _f(nm["ES"]) if nm else None
    d["es_hs"] = _f(hs["ES"]) if hs else None
    f10 = var_row("Filtered historical", 10) or var_row("Parametric normal", 10)
    d["var10"] = _f(f10["VaR"]) if f10 else None

    rp = rep.row("Risk profile", "ann_vol")
    d["ann_vol"] = _f(rp.get("daily_rebalanced")) if rp else None
    mdd = rep.row("Risk profile", "max_drawdown")
    d["mdd"] = _f(mdd.get("daily_rebalanced")) if mdd else None
    t1 = rep.row("Trailing-window", "1y")
    d["mdd_1y"] = _f(t1.get("max_drawdown")) if t1 else None
    d["vol_1y"] = _f(t1.get("ann_vol")) if t1 else None

    beta = rep.row("Portfolio exposures", "MKT")
    d["beta"] = _f(beta.get("exposure")) if beta else None
    d["exposures"] = [(r["factor"], _f(r["exposure"])) for r in rep.rows("Portfolio exposures")]
    r2 = rep.search(r"explained by factors: \*\*([\d.]+)%", float)
    d["r2"] = r2 / 100 if r2 is not None else None
    idio = rep.row("Variance decomposition", "idiosyncratic")
    d["idio_pct"] = _f(idio.get("pct_of_total")) / 100 if idio else None

    d["effn_w"] = rep.search(r"by weight \*\*([\d.]+)\*\*", float)
    d["effn_r"] = rep.search(r"by risk \*\*([\d.]+)\*\*", float)

    cv = rep.rows("Component VaR")
    cv_sorted = sorted(cv, key=lambda r: -(_f(r.get("pct_of_var")) or 0))
    d["top_risk"] = [(r["ticker"], _f(r["weight"]), (_f(r["pct_of_var"]) or 0) / 100) for r in cv_sorted[:5]]
    d["hedges"] = [(r["ticker"], (_f(r["pct_of_var"]) or 0) / 100) for r in cv if (_f(r.get("pct_of_var")) or 0) < 0]
    d["div_benefit"] = rep.search(r"Diversification benefit: ([\d.]+)%", float)

    sec = rep.rows("Sectors")
    sec_sorted = sorted(sec, key=lambda r: -(_f(r.get("risk_%")) or _f(r.get("weight_%")) or 0))
    d["sectors"] = [(r["sector"], (_f(r.get("weight_%")) or 0) / 100, (_f(r.get("risk_%")) or 0) / 100 if r.get("risk_%") else None) for r in sec_sorted]

    prox = rep.rows("Proxy-backfilled")
    heavy = {r["ticker"] for r in prox if (_f(r.get("pct_of_sample_proxied")) or 0) > 50}
    w = {r["ticker"]: (_f(r.get("weight_%")) or 0) / 100 for r in rep.rows("1. Portfolio")}
    d["proxy_heavy"] = sorted(heavy)
    d["proxy_weight"] = sum(w.get(t, 0) for t in heavy) if prox else 0.0

    hist = rep.rows("Historical replay")
    valid = [(r["scenario"], _f(r["portfolio_return"])) for r in hist if sum(_f(r.get(k)) or 0 for k in ("direct", "backfilled", "proxied")) > 0]
    d["stress_hist"] = valid
    d["worst_stress"] = min(valid, key=lambda x: x[1]) if valid else (None, None)
    hypo = [(r["scenario"], _f(r["portfolio_return"])) for r in rep.rows("Hypothetical factor shocks")]
    d["stress_hypo"] = hypo
    d["worst_hypo"] = min(hypo, key=lambda x: x[1]) if hypo else (None, None)

    bt = rep.rows("VaR backtest")
    d["bt_n"] = _f(bt[0]["n_days"]) if bt else None
    best = rep.search(r"passing conditional coverage: \*\*([\w_]+)\*\*")
    d["bt_best"] = best
    d["bt_best_row"] = next((r for r in bt if r["method"] == best), None)
    d["bt_rows"] = bt
    d["bt_normal"] = next((r for r in bt if r["method"] == "normal"), None)

    wf = {r["portfolio"]: r for r in rep.rows("Walk-forward")}
    d["wf_cur"] = wf.get("current")
    d["wf_rp"] = wf.get("ledoit_wolf/risk_parity")

    return d


def evaluate(d: dict, th: Mapping | None = None) -> list[Check]:
    """The ten health checks, coloured by the (overridable) thresholds."""
    t = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    for k, v in (th or {}).items():
        t.setdefault(k, {}).update(v)
    checks: list[Check] = []

    b = d.get("bt_best_row")
    if b:
        cc, esp = _f(b.get("cc_p")), _f(b.get("es_p"))
        zone = b.get("traffic_light_full") or b.get("traffic_light") or ""
        ok = sum(x is not None and x > 0.05 for x in (cc, esp))
        status = "green" if ok == 2 and zone != "red" else "amber" if ok >= 1 else "red"
        checks.append(Check("Risk model validity", f"{tex_escape(b['method'])}: {float(b['breach_rate_%']):.2f}\\% breaches",
                            status, r"coverage \& ES tests $p>0.05$",
                            f"Best model passes {ok}/2 tests; Basel zone {tex_escape(zone or 'n/a')} over {int(d['bt_n'] or 0):,} days.", "VaR backtest"))

    lim = t["var_limit"]["limit"]
    util = d["var"] / lim if d.get("var") is not None else None
    checks.append(Check("Daily VaR vs limit", f"{pct(d.get('var'), 2)} ({pct(util, 0)} of limit)",
                        _band(util, t["var_limit"]["amber_at"], 1.0), f"limit {pct(lim, 1)}; amber at {pct(t['var_limit']['amber_at'], 0)}",
                        f"1-day 99\\% {d.get('var_method', 'VaR')}: a one-in-a-hundred day costs about {money(d.get('var_usd'))}.", "VaR / ES"))

    tr = d.get("top_risk") or []
    if tr:
        tk, wt, sh = tr[0]
        c = t["top_risk_share"]
        checks.append(Check("Largest risk contributor", f"{tex_escape(tk)}: {pct(sh)} of VaR",
                            _band(sh, c["amber"], c["red"]), f"amber $\\ge${pct(c['amber'], 0)}, red $\\ge${pct(c['red'], 0)}",
                            f"{pct(wt)} of capital carries {pct(sh)} of risk.", "Component VaR"))

    c = t["effective_n_risk"]
    checks.append(Check("Diversification (effective positions)", f"{d['effn_r']:.1f} by risk / {d['effn_w']:.1f} by weight" if d.get("effn_r") else "n/a",
                        _band(d.get("effn_r"), c["amber"], c["red"], higher_is_worse=False), f"amber $<${c['amber']:.0f}, red $<${c['red']:.0f}",
                        "Number of equal-sized independent bets the book behaves like.", "Concentration"))

    secs = d.get("sectors") or []
    if secs and secs[0][2] is not None:
        s, sw, sr = secs[0]
        c = t["top_sector_risk"]
        unknown = s.strip().lower() in ("unknown", "n/a", "")
        checks.append(Check("Sector concentration", f"{tex_escape(s)}: {pct(sr)} of VaR",
                            "info" if unknown else _band(sr, c["amber"], c["red"]), f"amber $\\ge${pct(c['amber'], 0)}, red $\\ge${pct(c['red'], 0)}",
                            "Sectors not classified; set data.lookup\\_sectors or a sector per holding." if unknown else f"{pct(sw)} of capital.", "Sectors"))

    bt = d.get("beta")
    c = t["beta_mkt"]
    if bt is not None:
        st = "green" if c["lo"] <= bt <= c["hi"] else "amber" if c["lo2"] <= bt <= c["hi2"] else "red"
        checks.append(Check("Market beta", f"{bt:.2f}", st, f"green {c['lo']}--{c['hi']}, amber {c['lo2']}--{c['hi2']}",
                            f"Factors explain {pct(d.get('r2'))} of variance.", "Portfolio exposures"))

    nm, v = d.get("worst_stress", (None, None))
    c = t["worst_stress"]
    checks.append(Check("Worst historical stress", f"{signed_pct(v)}", _band(v, c["amber"], c["red"], higher_is_worse=False),
                        f"amber $\\le${signed_pct(c['amber'], 0)}, red $\\le${signed_pct(c['red'], 0)}",
                        f"{tex_escape(nm or 'n/a')} replayed at today's weights.", "Historical replay"))

    c = t["drawdown_1y"]
    checks.append(Check("1-year max drawdown", signed_pct(d.get("mdd_1y")), _band(d.get("mdd_1y"), c["amber"], c["red"], higher_is_worse=False),
                        f"amber $\\le${signed_pct(c['amber'], 0)}, red $\\le${signed_pct(c['red'], 0)}",
                        f"Full-sample max drawdown {signed_pct(d.get('mdd'))}.", "Trailing-window"))

    ratio = d["var"] / d["var_hs"] if d.get("var") and d.get("var_hs") and d.get("var_method") == "FHS" else None
    c = t["vol_regime"]
    reading = "n/a" if ratio is None else ("calmer than the sample average" if ratio < 0.95 else "near the sample average" if ratio < c["amber"] else "more turbulent than average")
    checks.append(Check("Volatility regime", "n/a" if ratio is None else f"{ratio:.2f}$\\times$ average", _band(ratio, c["amber"], c["red"]),
                        f"FHS / historical VaR; amber $\\ge${c['amber']}", f"Current conditions are {reading}.", "VaR / ES"))

    c = t["proxy_weight"]
    pw = d.get("proxy_weight")
    heavy = ", ".join(d.get("proxy_heavy") or []) or "none"
    checks.append(Check("Data quality (proxied history)", f"{pct(pw)} of capital", _band(pw, c["amber"], c["red"]),
                        f"weight in names $>$50\\% proxied; amber $\\ge${pct(c['amber'], 0)}",
                        f"Mostly-proxied: {tex_escape(heavy)}.", "Proxy-backfilled"))
    return checks


# ---------------------------------------------------------------------------
# LaTeX document
# ---------------------------------------------------------------------------
PREAMBLE = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.65in,headheight=13pt,headsep=8pt,footskip=20pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathpazo}
\usepackage[scaled=0.88]{helvet}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,longtable,array,colortbl}
\usepackage[table]{xcolor}
\usepackage[most]{tcolorbox}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[colorlinks=true,linkcolor=ink,urlcolor=accent]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{3pt}\setlength{\emergencystretch}{2em}
\setlist{nosep,leftmargin=1.2em}
\definecolor{ink}{HTML}{1F2A44}\definecolor{accent}{HTML}{2A78D6}\definecolor{muted}{HTML}{6B6A66}
\definecolor{okc}{HTML}{0CA30C}\definecolor{warnc}{HTML}{E0A100}\definecolor{badc}{HTML}{D03B3B}\definecolor{infoc}{HTML}{898781}
\definecolor{tile}{HTML}{F3F5F8}\definecolor{rowalt}{HTML}{F7F8FA}
\newcommand{\dotc}[1]{\tikz[baseline=-0.6ex]\fill[#1] (0,0) circle (3.2pt);}
\newcommand{\stok}{\dotc{okc}}\newcommand{\stwarn}{\dotc{warnc}}\newcommand{\stbad}{\dotc{badc}}\newcommand{\stinfo}{\dotc{infoc}}
\titleformat{\section}{\sffamily\Large\bfseries\color{ink}}{}{0pt}{}[{\color{accent}\titlerule[0.8pt]}]
\titleformat{\subsection}{\sffamily\large\bfseries\color{ink}}{}{0pt}{}
\titleformat{\subsubsection}{\sffamily\normalsize\bfseries\color{ink}}{}{0pt}{}
\titlespacing*{\section}{0pt}{8pt}{5pt}\titlespacing*{\subsection}{0pt}{7pt}{3pt}\titlespacing*{\subsubsection}{0pt}{5pt}{2pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\sffamily\footnotesize\color{ink}\textbf{risklib} risk report --- __NAME__}
\fancyhead[R]{\sffamily\footnotesize\color{muted}__DATES__}
\fancyfoot[C]{\sffamily\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\newtcolorbox{kpi}[1]{enhanced,colback=tile,colframe=tile,arc=2pt,left=6pt,right=6pt,top=4pt,bottom=4pt,
  fontupper=\sffamily\raggedright,title={\sffamily\scriptsize\color{muted}#1},coltitle=muted,colbacktitle=tile,
  attach title to upper,after title={\par\vspace{1pt}}}
\newtcolorbox{note}{enhanced,breakable,colback=white,colframe=accent!60,arc=1.5pt,left=5pt,right=5pt,top=3pt,bottom=3pt,fontupper=\small}
\renewcommand{\arraystretch}{1.12}
\begin{document}
"""


def _status_cmd(s: str) -> str:
    return {"green": r"\stok", "amber": r"\stwarn", "red": r"\stbad"}.get(s, r"\stinfo")


def _tile(title: str, big: str, sub: str, status: str | None = None) -> str:
    dot = (_status_cmd(status) + r"\,") if status else ""
    return (r"\begin{kpi}{" + title + "}" + dot + r"{\Large\bfseries\color{ink} " + big + r"}\\[1pt]"
            r"{\scriptsize\color{muted} " + sub + r"}\end{kpi}")


def _small_table(header: list[str], rows: list[list[str]], align: str) -> str:
    L = [r"{\small\rowcolors{2}{white}{rowalt}\begin{tabular}{" + align + r"}\toprule",
         " & ".join(r"\textbf{" + h + "}" for h in header) + r"\\\midrule"]
    for r in rows:
        L.append(" & ".join(r) + r"\\")
    L.append(r"\bottomrule\end{tabular}}")
    return "\n".join(L)


def snapshot_page(d: dict, checks: list[Check], labels: Mapping[str, str] | None = None) -> str:
    name = tex_escape(d.get("name") or "Portfolio")
    L = [r"\begin{center}{\sffamily\huge\bfseries\color{ink} Portfolio health snapshot}\\[2pt]"
         r"{\sffamily\small\color{muted} " + name + r" \textbullet{} data " + tex_escape(d.get("start") or "") + " to "
         + tex_escape(d.get("end") or "") + r" \textbullet{} " + f"{d.get('n_assets') or 0} holdings, {d.get('n_days') or 0:,} trading days" + r"}\end{center}"]
    counts = {s: sum(c.status == s for c in checks) for s in ("green", "amber", "red")}
    L.append(r"\begin{center}\sffamily\small " + rf"\stok\ {counts['green']} within range \quad \stwarn\ {counts['amber']} to watch \quad \stbad\ {counts['red']} outside range" + r"\end{center}\vspace{-2pt}")

    st = {c.name: c.status for c in checks}
    b = d.get("bt_best_row")
    tiles = [
        _tile("PORTFOLIO VALUE", money(d.get("value")), f"{d.get('n_assets') or 0} holdings, long-only"),
        _tile(f"1-DAY 99\\% VaR ({d.get('var_method')})", pct(d.get("var"), 2),
              (f"{money(d.get('var_usd'))}; 95\\% CI {pct(d.get('var_lo'), 2)}--{pct(d.get('var_hi'), 2)}" if d.get("var_lo") else money(d.get("var_usd"))),
              st.get("Daily VaR vs limit")),
        _tile("1-DAY 99\\% EXPECTED SHORTFALL", pct(d.get("es"), 2), f"mean loss beyond VaR; {money(d.get('es_usd'))}"),
        _tile("ANNUALISED VOLATILITY", pct(d.get("ann_vol"), 1), f"trailing 1y: {pct(d.get('vol_1y'), 1)}"),
        _tile("MARKET BETA", f"{d['beta']:.2f}" if d.get("beta") is not None else "n/a", f"factor $R^2$ {pct(d.get('r2'), 0)}", st.get("Market beta")),
        _tile("EFFECTIVE POSITIONS (RISK)", f"{d['effn_r']:.1f}" if d.get("effn_r") else "n/a", f"vs {d['effn_w']:.1f} by weight" if d.get("effn_w") else "", st.get("Diversification (effective positions)")),
        _tile("WORST HISTORICAL STRESS", signed_pct(d["worst_stress"][1]), tex_escape(d["worst_stress"][0] or ""), st.get("Worst historical stress")),
        _tile("RISK MODEL (BACKTEST)", (f"{float(b['breach_rate_%']):.2f}\\%" if b else "n/a"),
              (f"{tex_escape(b['method'])} breaches vs 1\\% expected" if b else ""), st.get("Risk model validity")),
    ]
    L.append(r"\begin{tcbraster}[raster columns=4,raster equal height,raster column skip=5pt,raster row skip=5pt]")
    L.extend(tiles)
    L.append(r"\end{tcbraster}")

    L.append(r"\subsection*{Health checklist}")
    col = lambda w: r">{\raggedright\arraybackslash}p{\dimexpr" + w + r"\linewidth-2\tabcolsep\relax}"  # noqa: E731
    L.append(r"{\small\rowcolors{2}{white}{rowalt}\begin{longtable}{@{}c" + col("0.21") + col("0.21") + col("0.19") + col("0.33") + r"@{}}\toprule")
    L.append(r" & \textbf{Check} & \textbf{Value} & \textbf{Rule} & \textbf{Reading}\\\midrule")
    for c in checks:
        lab = resolve_link(c.link, labels or {}) if c.link else ""
        nm = r"\hyperref[" + lab + "]{" + tex_escape(c.name) + "}" if lab else tex_escape(c.name)
        L.append(f"{_status_cmd(c.status)} & {nm} & {c.value} & \\footnotesize\\textcolor{{muted}}{{{c.rule}}} & \\footnotesize {c.reading}\\\\")
    L.append(r"\bottomrule\end{longtable}}")

    take = []
    if d.get("top_risk"):
        tk, wt, sh = d["top_risk"][0]
        lead = "Risk is concentrated: " if wt and sh > 1.2 * wt else "Largest risk contributor: "
        take.append(f"{lead}{tex_escape(tk)} is {pct(wt)} of capital and {pct(sh)} of VaR; the book behaves like {d.get('effn_r') or 0:.1f} independent positions.")
    nm, v = d.get("worst_stress", (None, None))
    if nm:
        take.append(f"Worst historical replay: {tex_escape(nm)} at {signed_pct(v)}.")
    if d.get("worst_hypo") and d["worst_hypo"][0]:
        take.append(f"Most damaging hypothetical: {tex_escape(d['worst_hypo'][0])} at {signed_pct(d['worst_hypo'][1])}.")
    if take:
        L.append(r"\begin{note}\textbf{Takeaways.} " + " ".join(take) + r"\end{note}")
    L.append(r"{\scriptsize\color{muted} Thresholds are defaults in \texttt{risklib.pdfreport.DEFAULT\_THRESHOLDS}; override them under \texttt{pdf: thresholds:} in the portfolio config. Colours flag where to look, not what to do.}")
    return "\n".join(L)


def summary_page(rep: Report, d: dict) -> str:
    L = [r"\clearpage\section*{Risk summary}"]
    rows = []
    for r in rep.rows("VaR / ES"):
        if _f(r.get("horizon")) == 1 and abs((_f(r.get("alpha")) or 0) - 0.99) < 1e-9:
            ci = f"{pct(_f(r['VaR_lo']), 2)}--{pct(_f(r['VaR_hi']), 2)}" if r.get("VaR_lo") else ""
            rows.append([inline(r["method"]), pct(_f(r["VaR"]), 2), ci, pct(_f(r["ES"]), 2), money(_f(r.get("VaR_$")))])
    left = [r"\subsection*{How much could be lost (1 day, 99\%)}", _small_table(["Method", "VaR", "95\\% CI", "ES", "VaR \\$"], rows, "@{}lrrrr@{}")]
    if d.get("var10") is not None:
        left.append(r"\\[2pt]{\footnotesize 10-day 99\% VaR (" + d.get("var_method", "") + "): " + pct(d["var10"], 2) + ".}")
    tr_rows = [[tex_escape(t), pct(w), pct(s)] for t, w, s in d.get("top_risk", [])]
    right = [r"\subsection*{Where the risk comes from}", _small_table(["Holding", "Weight", "Share of VaR"], tr_rows, "@{}lrr@{}")]
    if d.get("hedges"):
        right.append(r"\\[2pt]{\footnotesize Hedging the book today: " + ", ".join(f"{tex_escape(t)} ({signed_pct(s, 1)})" for t, s in d["hedges"]) + ".}")
    L.append(r"\begin{minipage}[t]{0.56\textwidth}" + "\n".join(left) + r"\end{minipage}\hfill\begin{minipage}[t]{0.40\textwidth}" + "\n".join(right) + r"\end{minipage}")

    ex_rows = [[tex_escape(f), num_cell(f"{v:.2f}")] for f, v in d.get("exposures", [])]
    sec_rows = [[tex_escape(s), pct(w), pct(r) if r is not None else ""] for s, w, r in d.get("sectors", [])[:6]]
    L.append(r"\par\medskip\begin{minipage}[t]{0.30\textwidth}\subsection*{Factor exposures}" + _small_table(["Factor", "Beta"], ex_rows, "@{}lr@{}")
             + (r"\\[2pt]{\footnotesize Stock-specific share of variance: " + pct(d.get("idio_pct")) + ".}" if d.get("idio_pct") is not None else "")
             + r"\end{minipage}\hfill\begin{minipage}[t]{0.64\textwidth}\subsection*{Sectors}" + _small_table(["Sector", "Weight", "Share of VaR"], sec_rows, "@{}lrr@{}") + r"\end{minipage}")

    st_rows = [[tex_escape(n), signed_pct(v)] for n, v in sorted(d.get("stress_hist", []), key=lambda x: x[1])[:8]]
    hy_rows = [[tex_escape(n), signed_pct(v)] for n, v in sorted(d.get("stress_hypo", []), key=lambda x: x[1])[:6]]
    L.append(r"\par\medskip\begin{minipage}[t]{0.44\textwidth}\subsection*{Historical replays (worst eight)}" + _small_table(["Scenario", "Return"], st_rows, "@{}lr@{}")
             + r"\end{minipage}\hfill\begin{minipage}[t]{0.52\textwidth}\subsection*{Hypothetical shocks (worst six)}" + _small_table(["Scenario", "Return"], hy_rows, r"@{}>{\raggedright\arraybackslash}p{0.72\linewidth}r@{}") + r"\end{minipage}")

    bt_rows = []
    for r in d.get("bt_rows", []):
        mark = r"\textbf{" + tex_escape(r["method"]) + "}" if r["method"] == d.get("bt_best") else tex_escape(r["method"])
        zone = r.get("traffic_light_full") or r.get("traffic_light", "")
        zc = {"green": r"\stok", "yellow": r"\stwarn", "red": r"\stbad"}.get(zone, "")
        bt_rows.append([mark, r["breaches"], f"{float(r['breach_rate_%']):.2f}\\%", f"{float(r['kupiec_p']):.3f}", f"{float(r['christoffersen_p']):.3f}",
                        f"{float(r['es_Z2']):.2f}", f"{float(r['es_p']):.3f}", zc + " " + tex_escape(zone)])
    if bt_rows:
        L.append(r"\subsection*{Can the VaR be trusted? (out-of-sample backtest, " + f"{int(d.get('bt_n') or 0):,}" + r" days, 1\% expected)}")
        L.append(_small_table(["Model", "Breaches", "Rate", "Kupiec $p$", "Indep.\\ $p$", "ES $Z_2$", "ES $p$", "Zone"], bt_rows, "@{}lrrrrrrl@{}"))
        L.append(r"\\[2pt]{\footnotesize $p>0.05$ = not rejected. Bold = model selected for the headline VaR.}")
    if d.get("wf_cur") and d.get("wf_rp"):
        c, rp = d["wf_cur"], d["wf_rp"]
        L.append(r"\subsection*{Allocation reference (walk-forward, out of sample)}")
        L.append(_small_table(["Portfolio", "Return", "Vol", "Sharpe", "Max DD"],
                              [["Current weights", pct(_f(c["ann_return"])), pct(_f(c["ann_vol"])), f"{float(c['sharpe']):.2f}", signed_pct(_f(c["max_drawdown"]))],
                               ["Risk parity (Ledoit-Wolf)", pct(_f(rp["ann_return"])), pct(_f(rp["ann_vol"])), f"{float(rp['sharpe']):.2f}", signed_pct(_f(rp["max_drawdown"]))]],
                              "@{}lrrrr@{}"))
    return "\n".join(L)


def render_table(b: Block) -> str:
    """A Markdown table as a longtable with content-proportional column widths."""
    ncol = len(b.header)
    numeric = [all(_is_num(r[j]) for r in b.rows if j < len(r)) for j in range(ncol)]
    lens = []
    for j in range(ncol):
        head = min(max((len(w) for w in re.split(r"[ _]", b.header[j])), default=1), 10)
        body = max((len(r[j]) for r in b.rows if j < len(r)), default=1)
        lens.append(min(max(head, body, 4), 42))
    tot = sum(lens)
    shrink = 2 * (ncol - 1) / ncol  # column separations shared across the columns
    cols = []
    for j in range(ncol):
        align = r"\raggedleft" if (numeric[j] and j > 0) else r"\raggedright"
        cols.append(r">{" + align + r"\arraybackslash}p{\dimexpr" + f"{0.998 * lens[j] / tot:.4f}" + r"\linewidth-" + f"{shrink:.3f}" + r"\tabcolsep\relax}")
    size = r"\scriptsize" if ncol > 8 else r"\footnotesize"
    sep = r"\setlength{\tabcolsep}{2.5pt}" if ncol > 8 else r"\setlength{\tabcolsep}{4pt}"
    head = " & ".join(r"\textbf{" + inline(h.replace("_", " ")) + "}" for h in b.header) + r"\\\midrule"
    L = ["{" + size + sep + r"\rowcolors{2}{white}{rowalt}\begin{longtable}{@{}" + "".join(cols) + r"@{}}\toprule", head, r"\endhead"]
    for r in b.rows:
        r = (r + [""] * ncol)[:ncol]
        L.append(" & ".join(num_cell(c) for c in r) + r"\\")
    L.append(r"\bottomrule\end{longtable}}")
    return "\n".join(L)


def detail_pages(rep: Report, base: Path, labels: Mapping[str, str] | None = None) -> str:
    labels = labels or heading_labels(rep)
    L = [r"\clearpage\section*{Detail}",
         r"{\small The complete report, section by section. Numbers are exactly those in \texttt{report.md}.}"]
    skip_first = True
    for b in rep.blocks:
        if b.kind == "heading":
            if b.level == 1:
                continue
            cmd = {2: r"\subsection*", 3: r"\subsubsection*"}.get(b.level, r"\paragraph*")
            L.append(cmd + "{" + inline(b.text) + r"}\label{" + labels[b.text] + "}")
        elif b.kind == "para":
            if skip_first and b.text.startswith("Generated"):
                skip_first = False
                continue
            L.append(r"{\small " + inline(b.text) + "}\n")
        elif b.kind == "quote":
            L.append(r"\begin{note}" + inline(b.text) + r"\end{note}")
        elif b.kind == "image":
            if (base / b.path).exists():
                L.append(r"\begin{center}\includegraphics[width=0.86\textwidth,height=0.36\textheight,keepaspectratio]{" + b.path.replace("\\", "/") + r"}\end{center}")
        elif b.kind == "table":
            L.append(render_table(b))
        elif b.kind == "rule":
            L.append(r"\medskip\hrule\medskip")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Build and compile
# ---------------------------------------------------------------------------
def build_tex(md_path: str | Path, thresholds: Mapping | None = None) -> tuple[str, dict, list[Check]]:
    """LaTeX source, snapshot numbers and checks for a report.md."""
    md_path = Path(md_path)
    rep = Report(md_path.read_text(encoding="utf-8"))
    d = extract(rep)
    checks = evaluate(d, thresholds)
    pre = PREAMBLE.replace("__NAME__", tex_escape(d.get("name") or "")).replace("__DATES__", tex_escape(f"{d.get('start')} to {d.get('end')}"))
    labels = heading_labels(rep)
    body = [snapshot_page(d, checks, labels), summary_page(rep, d), detail_pages(rep, md_path.parent, labels), r"\end{document}"]
    return pre + "\n".join(body), d, checks


def compile_tex(tex_path: Path) -> Path | None:
    """Compile with pdflatex (two passes) or latexmk. Returns the PDF path, or None without an engine.

    pdflatex is tried first because latexmk on MiKTeX needs a separate Perl install.
    """
    cwd = tex_path.parent
    if shutil.which("pdflatex"):
        cmds = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]] * 2
    elif shutil.which("latexmk"):
        cmds = [["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]]
    else:
        return None
    for c in cmds:
        r = subprocess.run(c, cwd=cwd, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            log = cwd / tex_path.with_suffix(".log").name
            tail = log.read_text(errors="replace").splitlines()[-25:] if log.exists() else r.stdout.splitlines()[-25:]
            raise RuntimeError("LaTeX failed:\n" + "\n".join(tail))
    for ext in (".aux", ".out", ".fls", ".fdb_latexmk", ".toc", ".log"):
        f = tex_path.with_suffix(ext)
        if f.exists():
            f.unlink()
    pdf = tex_path.with_suffix(".pdf")
    return pdf if pdf.exists() else None


def render_pdf(md_path: str | Path, thresholds: Mapping | None = None, compile: bool = True, out_name: str = "report.tex", verbose: bool = True) -> Path:
    """Write report.tex beside report.md and compile it if a TeX engine is available.

    Returns the PDF path when compiled, otherwise the .tex path.
    """
    md_path = Path(md_path)
    tex, _, checks = build_tex(md_path, thresholds)
    tex_path = md_path.parent / out_name
    tex_path.write_text(tex, encoding="utf-8")
    if not compile:
        return tex_path
    pdf = compile_tex(tex_path)
    if pdf is None:
        if verbose:
            print(f"wrote {tex_path} (no LaTeX engine on PATH: install MiKTeX or TeX Live, or compile on Overleaf with the charts/ folder)")
        return tex_path
    if verbose:
        n = {s: sum(c.status == s for c in checks) for s in ("green", "amber", "red")}
        print(f"wrote {pdf}  (health: {n['green']} green, {n['amber']} amber, {n['red']} red)")
    return pdf
