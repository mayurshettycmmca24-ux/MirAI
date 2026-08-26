# MobiMart Inventory Allocation System

## Overview

MobiMart operates 25 stores and sells about 60 phone models under a ₹4 crore chain-wide weekly budget cap. This system generates 12 months of weekly store-model demand, allocates inventory using demand, lifecycle, margin, and store-fit signals, and identifies end-of-life exposure before it becomes stranded capital. A naive allocation is evaluated alongside the real engine as a transparent benchmark.

## Architecture

- `stores.py` - defines the 25 stores, city tiers, footfall, and sales-mix profiles.
- `catalog.py` - defines the 60-phone catalog, prices, margins, lifecycle assumptions, successor links, and rumoured/confirmed launch status.
- `sales_generator.py` - generates seeded 52-week store-model sales with lifecycle and festive effects, plus sanity checks.
- `allocation_engine.py` - ranks expected gross profit per rupee and allocates stock under the budget, store-model concentration, and stock constraints; includes the festive pre-commitment policy.
- `eol_risk.py` - scores at-risk positions and compares markdown, transfer, and hold actions, including launch-confidence treatment.
- `baseline.py` - provides the naive benchmark using each model's preceding four-week store sales share.
- `evaluate.py` - evaluates both strategies with carried warehouse stock and FIFO store inventory lots.
- `eol_dashboard.py` - generates the static interactive HTML dashboard from EOL and performance outputs.
- `rehearsal.py` - runs the exact live-defense successor and demand-drop scenario.
- `run_pipeline.py` - runs the complete generation, allocation, EOL, evaluation, and dashboard workflow.
- `output/` - contains generated sales, allocation, risk, evaluation, and dashboard artifacts.

## How to run it

From a clean clone with Python and `pandas`/`numpy` installed:

```powershell
python run_pipeline.py
```

This runs sales history generation, latest allocation, EOL risk assessment, carried-lot evaluation, and dashboard generation. The pipeline sets random seed `42`; catalog and risk components also use fixed local generators. Two post-fix reruns produced zero diff lines for both the performance file and scorecard summary, with matching SHA-256 hashes.

The focused demonstrations are:

```powershell
python allocation_engine.py
python rehearsal.py
```

All generated artifacts are written to `output/`. The dashboard can be opened directly; no server is required.

## Key results

Final corrected 51-week carried-inventory evaluation. The separate `output/trailing_4_week_performance.csv` remains a trailing-period view for the dashboard.

| Metric                 |  Real engine | Naive baseline |
| ---------------------- | -----------: | -------------: |
| Stockout rate          |        20.4% |          27.4% |
| Average weeks of cover |        3.907 |          4.962 |
| Dead-stock rate        |        27.5% |           2.5% |
| Markdown loss          | ₹235,169,293 |   ₹496,756,463 |
| Capital turns          |        5.773 |          0.279 |

## Where the engine wins / loses

- **Wins:** lower stockouts, lower average cover, lower modeled markdown loss, and higher capital turns than the naive baseline.
- **Loses:** dead stock is higher at 27.5% versus 2.5%. The engine buys stockout safety with more dead capital; the baseline avoids dead stock mainly by under-allocating.

## End-of-life risk handling

For each risky store-model position, the system compares a markdown charge, a better-fit-store transfer where available, and the modeled cost of holding stock until forced action. The evaluator's inventory lots are carried forward; residual units are charged a one-time 25% of acquisition-cost markdown at expiry.

Successor launch confidence changes the risk treatment: a confirmed successor uses the full unsold-probability estimate, while a rumoured successor discounts that probability by 25%. In the rehearsal for BLR-01/F03, a rumoured successor produces **HOLD at ₹69,258**, while the same case marked confirmed produces **MARKDOWN at ₹83,295**.

## Festive season handling

The generated demand uplift is **3.57x normal**, against the coded 3.5x festive multiplier. Known festive weeks receive a simplified **50% same-week budget uplift** as a distributor pre-commitment stand-in:

| Festive week | Unmet before | Unmet after |
| ------------ | -----------: | ----------: |
| 2025-09-29   |        72.6% |       28.6% |
| 2025-10-20   |        81.9% |       67.8% |

This is not a full 3-4 week purchase-order or distributor lead-time simulation. It is a documented policy approximation that increases the weekly budget from ₹4 crore to ₹6 crore for known festive weeks.

## Live-scenario rehearsal

Run:

```powershell
python rehearsal.py
```

The rehearsal injects a successor launching in 10 days, 42 units held across 9 stores for model F03, and a 40% trailing-sales drop at BLR-01. It prints all nine EOL flags and recommended actions, compares rumoured versus confirmed successor costs, and shows BLR-01/F03 allocation changing from 3 units in the normal case to 2 units after the sales drop.

## Dashboard

Open [output/eol_dashboard.html](output/eol_dashboard.html) directly in a browser. It shows at-risk positions, capital by store, trailing four-week recommendation performance, and the full corrected 51-week scorecard, including the ₹235,169,293 real-engine and ₹496,756,463 naive-baseline markdown totals.

## Known limitations

- Launch-confidence status is synthetic catalog metadata, not real market intelligence.
- Demand is simulated and seeded, not sourced from transactional sales systems.
- Festive pre-commitment is a same-week budget policy, not a true 3-4 week lead-time or purchase-order model.
- The evaluator uses a carried-lot inventory simulation with FIFO consumption. This is a deliberate modeling choice for reproducible evaluation; it does not represent a live inventory ledger, replenishment lead time, or transfer execution system.
- Capital turns use the evaluator's average inventory-value convention and are not audited financial turns.

## Repo structure

```text
.
├── allocation_engine.py
├── baseline.py
├── catalog.py
├── eol_dashboard.py
├── eol_risk.py
├── evaluate.py
├── rehearsal.py
├── run_pipeline.py
├── sales_generator.py
├── stores.py
└── output/
    ├── sales_history.csv
    ├── latest_allocation.csv
    ├── latest_unmet_demand.csv
    ├── eol_risk_output.csv
    ├── eol_risk_output.json
    ├── recommendation_log.csv
    ├── recommendation_scored_log.csv
    ├── trailing_4_week_performance.csv
    ├── current_inventory_position.csv
    └── eol_dashboard.html
```
