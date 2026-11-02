"""Command-line interface.

    risklib report portfolio.yaml [--out DIR] [--offline] [--pdf]
    risklib fetch  portfolio.yaml [--out DIR]
    risklib check  portfolio.yaml
    risklib pdf    reports/report.md [--config portfolio.yaml] [--no-compile]

``python -m risklib`` works the same way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="risklib", description="Portfolio risk report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    helps = {"report": "run the full report", "fetch": "download prices only", "check": "validate a config"}
    for name, text in helps.items():
        p = sub.add_parser(name, help=text)
        p.add_argument("config", help="portfolio YAML")
        p.add_argument("--out", default=None, help="output directory (default: output.dir in the config)")
        p.add_argument("--offline", action="store_true", help="use data/*.csv in the output directory instead of yfinance")
        p.add_argument("--quiet", action="store_true")
        if name == "report":
            p.add_argument("--pdf", action="store_true", help="also render report.pdf (same as output.pdf: true)")
    p = sub.add_parser("pdf", help="render an existing report.md as a PDF")
    p.add_argument("report_md", help="path to report.md (with its charts/ folder beside it)")
    p.add_argument("--config", default=None, help="portfolio YAML, read for pdf.thresholds only")
    p.add_argument("--no-compile", action="store_true", help="write report.tex without compiling")
    p.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    from .portfolio import load_config

    if args.cmd == "pdf":
        from .pdfreport import render_pdf

        th = (load_config(args.config).get("pdf") or {}).get("thresholds") if args.config else None
        out = render_pdf(args.report_md, thresholds=th, compile=not args.no_compile, verbose=not args.quiet)
        if args.no_compile and not args.quiet:
            print(f"wrote {out}")
        return 0

    from .report import fetch_data, resolve_config, run

    cfg = load_config(args.config)
    if args.offline:
        cfg.setdefault("data", {})["source"] = "csv"
    if args.cmd == "check":
        c = resolve_config(cfg)
        print(f"OK: {len(c['holdings'])} holdings, factors {list(c['factors'])}, start {c['start']}")
        return 0
    if args.cmd == "fetch":
        c = resolve_config(cfg)
        out = Path(args.out or c["output"]["dir"])
        prices, fprices = fetch_data(c, out)
        print(f"saved {prices.shape} prices and {fprices.shape} factor prices under {out / 'data'}")
        return 0
    if args.pdf:
        cfg.setdefault("output", {})["pdf"] = True
    run(cfg, out_dir=args.out, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
