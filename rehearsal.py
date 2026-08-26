"""Print the live-defense successor and demand-drop rehearsal."""

import pandas as pd

from allocation_engine import allocate_weekly_stock
from catalog import generate_catalog
from eol_risk import assess_eol_risk
from sales_generator import generate_sales_history
from stores import generate_stores


AS_OF = pd.Timestamp("2025-10-20")
MODEL_ID = "F03"
SUCCESSOR_ID = "F13"
STORE_IDS = ["BLR-01", "BLR-02", "BLR-03", "BLR-04", "BLR-05", "BLR-06", "BLR-07", "BLR-08", "BEL-01"]
HELD_UNITS = [5, 5, 5, 5, 5, 5, 4, 4, 4]


def main() -> None:
    stores = generate_stores()
    catalog = generate_catalog()
    history = generate_sales_history(stores, catalog)
    catalog.loc[catalog["model_id"].eq(SUCCESSOR_ID), "launch_date"] = AS_OF + pd.Timedelta(days=10)
    catalog.loc[catalog["model_id"].eq(SUCCESSOR_ID), "launch_date_status"] = "rumoured"

    dropped_history = history.copy()
    trailing = (
        dropped_history["store_id"].eq(STORE_IDS[0])
        & dropped_history["model_id"].eq(MODEL_ID)
        & dropped_history["week_start_date"].lt(AS_OF)
        & dropped_history["week_start_date"].ge(AS_OF - pd.Timedelta(weeks=6))
    )
    dropped_history.loc[trailing, "units_sold"] = (
        dropped_history.loc[trailing, "units_sold"] * 0.6
    ).astype(int)

    held = pd.DataFrame({"store_id": STORE_IDS, "model_id": MODEL_ID, "units_available": HELD_UNITS})
    risks = assess_eol_risk(held, catalog, dropped_history, stores, AS_OF)
    confirmed_catalog = catalog.copy()
    confirmed_catalog.loc[confirmed_catalog["model_id"].eq(SUCCESSOR_ID), "launch_date_status"] = "confirmed"
    confirmed_risks = assess_eol_risk(held, confirmed_catalog, dropped_history, stores, AS_OF)
    print("LIVE DEFENSE REHEARSAL: F03 with successor F13 launching in 10 days")
    print("Held inventory: 42 units across 9 stores; BLR-01 trailing six-week sales reduced by 40%.")
    print("\nEOL flags and actions:")
    print(risks[["store_id", "model_id", "units_at_risk", "successor_launch_status", "risk_triggers", "markdown_cost", "transfer_cost", "hold_total_cost", "recommended_action"]].to_string(index=False))
    comparison = risks[["store_id", "recommended_action", "recommended_cost"]].merge(
        confirmed_risks[["store_id", "recommended_action", "recommended_cost"]],
        on="store_id", suffixes=("_rumoured", "_confirmed"),
    )
    print("\nRumoured versus confirmed successor costs:")
    print(comparison.to_string(index=False))

    normal, _, _, _ = allocate_weekly_stock(
        history, catalog, stores, {MODEL_ID: 1000}, week_start_date=AS_OF
    )
    dropped, _, _, _ = allocate_weekly_stock(
        dropped_history, catalog, stores, {MODEL_ID: 1000}, week_start_date=AS_OF
    )
    normal_row = normal[normal["store_id"].eq(STORE_IDS[0]) & normal["model_id"].eq(MODEL_ID)]
    dropped_row = dropped[dropped["store_id"].eq(STORE_IDS[0]) & dropped["model_id"].eq(MODEL_ID)]
    print("\nAllocation change for BLR-01 / F03:")
    print(pd.DataFrame([
        {"case": "normal", "expected_demand": normal_row.expected_demand.iloc[0], "requested_units": normal_row.requested_units.iloc[0], "allocated_units": normal_row.allocated_units.iloc[0]},
        {"case": "40% trailing-sales drop", "expected_demand": dropped_row.expected_demand.iloc[0], "requested_units": dropped_row.requested_units.iloc[0], "allocated_units": dropped_row.allocated_units.iloc[0]},
    ]).to_string(index=False))


if __name__ == "__main__":
    main()
