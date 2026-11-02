# output

Stage 4 of the pipeline: results to files in `output.dir` (default `reports/`).

| File | Writes |
|---|---|
| `charts.py` | `charts/*.png`: conditional volatility, VaR backtest, component VaR, stress scenarios, walk-forward wealth |
| `markdown.py` | `report.md`: every table and chart, section by section |
| `metrics.py` | `metrics.json`: the headline numbers |
| `pdf.py` | `report.pdf`: a portfolio-health snapshot and the full report, via LaTeX (`--pdf`) |

Previous stage: [`models/`](../models) · Pipeline: [`pipeline.py`](../pipeline.py)
