# preprocessing

Stage 2 of the pipeline: raw prices to a clean, aligned panel and portfolio weights.

| File | Contents |
|---|---|
| `panel.py` | Price validation, capped forward-fill, proxy backfill for short histories, common-window alignment, log/simple returns, `build_panel` (the full cleaning sequence), `stress_panel` (long history for crisis replays) |
| `portfolio.py` | Holdings (shares, dollars or weights) to normalized weights, daily-rebalanced and buy-and-hold portfolio returns, the `Portfolio` object |

Previous stage: [`data/`](../data) · Next stage: [`models/`](../models)
