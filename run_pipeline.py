"""Regenerate the complete MobiMart review package."""

from pathlib import Path
import random

import numpy as np
import pandas as pd

from allocation_engine import allocate_weekly_stock
from catalog import generate_catalog
from eol_dashboard import build_dashboard
from eol_risk import assess_eol_risk
from evaluate import evaluate_over_history, print_scorecard
from sales_generator import generate_sales_history, print_sanity_checks
from stores import generate_stores


OUTPUT_DIR = Path("output")
BUDGET = 4_00_00_000
RANDOM_SEED = 42


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    stores = generate_stores()
    catalog = generate_catalog()
    history = generate_sales_history(stores, catalog)
    history.to_csv(OUTPUT_DIR / "sales_history.csv", index=False)
    print_sanity_checks(history, stores, catalog)

    stock = {model_id: 1_000 for model_id in catalog["model_id"]}
    dates = pd.to_datetime(history["week_start_date"]).sort_values().unique()
    first_allocation_week = pd.Timestamp(dates[-2])
    allocation, used, remaining, unmet = allocate_weekly_stock(
        history, catalog, stores, stock.copy(), budget=BUDGET,
        week_start_date=first_allocation_week,
    )
    allocation.to_csv(OUTPUT_DIR / "latest_allocation.csv", index=False)
    unmet.to_csv(OUTPUT_DIR / "latest_unmet_demand.csv", index=False)
    print(f"Allocation: Rs {used:,.0f} used; Rs {remaining:,.0f} remaining; {len(unmet)} unmet rows")

    risks = assess_eol_risk(
        allocation, catalog, history, stores, as_of_date=first_allocation_week
    )
    risks.to_csv(OUTPUT_DIR / "eol_risk_output.csv", index=False)
    risks.to_json(OUTPUT_DIR / "eol_risk_output.json", orient="records")

    log, scored_log, scorecard = evaluate_over_history(
        history, catalog, stores, stock, budget=BUDGET
    )
    log.to_csv(OUTPUT_DIR / "recommendation_log.csv", index=False)
    scored_log.to_csv(OUTPUT_DIR / "recommendation_scored_log.csv", index=False)
    trailing_weeks = pd.to_datetime(scored_log["week_start_date"]).nlargest(4).unique()
    scored_log[scored_log["week_start_date"].isin(trailing_weeks)].groupby("strategy").agg(
        predicted_gross_profit=("predicted_gross_profit", "sum"),
        realized_gross_profit=("realized_gross_profit", "sum"),
        markdown_loss=("markdown_loss", "sum"),
        stockouts=("stockout", "sum"),
    ).reset_index().to_csv(OUTPUT_DIR / "trailing_4_week_performance.csv", index=False)
    latest_inventory = log[
        (log["strategy"] == "real_engine")
        & (pd.to_datetime(log["week_start_date"]) == pd.Timestamp(log["week_start_date"].max()))
    ].merge(catalog[["model_id", "acquisition_cost"]], on="model_id", how="left")
    latest_inventory["capital_tied_up"] = (
        latest_inventory["allocated_units"] * latest_inventory["acquisition_cost"]
    )
    latest_inventory.to_csv(OUTPUT_DIR / "current_inventory_position.csv", index=False)
    build_dashboard(
        risks,
        inventory_df=latest_inventory,
        performance_df=pd.read_csv(OUTPUT_DIR / "trailing_4_week_performance.csv"),
        scorecard_df=scorecard,
        out_path=OUTPUT_DIR / "eol_dashboard.html",
    )
    print_scorecard(scorecard)
    print(f"Actual festive uplift: {history.loc[history.week_start_date.map(lambda d: pd.Timestamp(d).month == 9 and pd.Timestamp(d).day == 29 or pd.Timestamp(d).month == 10 and pd.Timestamp(d).day == 20), 'units_sold'].mean() / history.loc[~history.week_start_date.isin([pd.Timestamp('2025-09-29'), pd.Timestamp('2025-10-20')]), 'units_sold'].mean():.2f}x")


if __name__ == "__main__":
    main()
