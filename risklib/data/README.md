# data

Stage 1 of the pipeline: where prices come from.

| File | Contents |
|---|---|
| `sources.py` | `download_prices` (Yahoo Finance via yfinance), `save_prices_csv` / `load_prices_csv`, `fetch_data` (holdings, proxies and factor ETFs for a config), `fetch_sectors` (sector labels, cached) |
| `synthetic.py` | Simulated factor markets with GARCH volatility for the tests and `examples/demo_offline.py`; the report pipeline never uses it |

Next stage: [`preprocessing/`](../preprocessing)
