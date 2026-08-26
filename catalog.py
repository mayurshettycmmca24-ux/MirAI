"""Phone catalog used by the MobiMart demand simulation."""



from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhoneModel:
    """A sellable model and the lifecycle assumptions used for forecasting."""

    model_id: str
    model_name: str
    brand: str
    price_tier: str  # Price band determines affordability and store-model fit.
    selling_price: int  # Customer-facing price used to calculate turnover and gross profit.
    margin_pct: float  # Tier-specific gross margin varies with brand and commercial terms.
    acquisition_cost: int  # Distributor cost is selling price less MobiMart's gross margin.
    launch_date: str  # Launch timing positions the model on its demand lifecycle curve.
    expected_lifecycle_weeks: int  # Shorter flagship hype cycles reflect faster replacement behavior.
    successor_model_id: Optional[str]  # A family replacement triggers predecessor cannibalization.
    launch_date_status: str  # Whether this model's launch date is confirmed or rumoured.


def generate_catalog() -> pd.DataFrame:
    """Return 60 phones spanning MobiMart's three commercial price bands.

    Launches run from February 2025 through August 2026, giving the 12-month
    window both new launches and mature/near-end-of-life models. Successors are
    linked within the same brand family and deliberately overlap the sales
    window so the cannibalization effect can be observed.
    """
    rows = [
        # Budget models: longer lifecycles because buyers replace less frequently.
        ("B01", "Nokia 110 Power", "Nokia", "budget", 6999, "2025-02-10", 40, "B11"),
        ("B02", "Samsung Galaxy M05", "Samsung", "budget", 8999, "2025-03-03", 38, "B12"),
        ("B03", "Redmi A5", "Xiaomi", "budget", 7999, "2025-03-24", 40, "B13"),
        ("B04", "Realme Note 60", "Realme", "budget", 9499, "2025-04-14", 38, "B14"),
        ("B05", "Moto G15 Play", "Motorola", "budget", 10999, "2025-05-05", 36, "B15"),
        ("B06", "Lava Blaze Core", "Lava", "budget", 8499, "2025-05-26", 40, "B16"),
        ("B07", "POCO C75", "Xiaomi", "budget", 11999, "2025-06-16", 36, None),
        ("B08", "Samsung Galaxy F06", "Samsung", "budget", 10499, "2025-07-07", 38, None),
        ("B09", "Realme C75x", "Realme", "budget", 12999, "2025-08-04", 36, None),
        ("B10", "Nokia 3210 4G", "Nokia", "budget", 7499, "2025-09-01", 42, None),
        ("B11", "Nokia 125 Power", "Nokia", "budget", 7999, "2025-10-06", 40, None),
        ("B12", "Samsung Galaxy M06", "Samsung", "budget", 9999, "2025-11-03", 38, None),
        ("B13", "Redmi A6", "Xiaomi", "budget", 8999, "2026-01-12", 40, None),
        ("B14", "Realme Note 70", "Realme", "budget", 10499, "2026-02-16", 38, None),
        ("B15", "Moto G16 Play", "Motorola", "budget", 11499, "2026-03-23", 36, None),
        ("B16", "Lava Blaze Core 2", "Lava", "budget", 9499, "2026-04-20", 40, None),
        ("B17", "POCO C85", "Xiaomi", "budget", 12499, "2026-05-18", 36, None),
        ("B18", "Samsung Galaxy F07", "Samsung", "budget", 11499, "2026-06-15", 38, None),
        ("B19", "Realme C76x", "Realme", "budget", 13999, "2026-07-13", 36, None),
        ("B20", "Nokia 3310 Max", "Nokia", "budget", 8999, "2026-08-03", 42, None),
        # Mid-range models: balanced replacement cycles and broadest demand.
        ("M01", "Samsung Galaxy A26", "Samsung", "mid", 22999, "2025-02-17", 32, "M11"),
        ("M02", "Redmi Note 14", "Xiaomi", "mid", 18999, "2025-03-10", 34, "M12"),
        ("M03", "Realme 14 Pro", "Realme", "mid", 27999, "2025-04-07", 30, "M13"),
        ("M04", "OnePlus Nord CE5", "OnePlus", "mid", 24999, "2025-05-12", 30, "M14"),
        ("M05", "Vivo V50e", "Vivo", "mid", 29999, "2025-06-02", 30, "M15"),
        ("M06", "Motorola Edge 60 Fusion", "Motorola", "mid", 22999, "2025-07-14", 32, "M16"),
        ("M07", "POCO X7 Pro", "Xiaomi", "mid", 26999, "2025-08-11", 30, None),
        ("M08", "Nothing Phone (3a)", "Nothing", "mid", 27999, "2025-09-08", 30, None),
        ("M09", "Samsung Galaxy A36", "Samsung", "mid", 32999, "2025-10-13", 32, None),
        ("M10", "OnePlus Nord 5", "OnePlus", "mid", 29999, "2025-11-10", 30, None),
        ("M11", "Samsung Galaxy A27", "Samsung", "mid", 23999, "2025-12-08", 32, None),
        ("M12", "Redmi Note 15", "Xiaomi", "mid", 19999, "2026-01-19", 34, None),
        ("M13", "Realme 15 Pro", "Realme", "mid", 28999, "2026-02-23", 30, None),
        ("M14", "OnePlus Nord CE6", "OnePlus", "mid", 25999, "2026-03-30", 30, None),
        ("M15", "Vivo V51e", "Vivo", "mid", 30999, "2026-04-27", 30, None),
        ("M16", "Motorola Edge 70 Fusion", "Motorola", "mid", 23999, "2026-05-25", 32, None),
        ("M17", "POCO X8 Pro", "Xiaomi", "mid", 28999, "2026-06-22", 30, None),
        ("M18", "Nothing Phone (4a)", "Nothing", "mid", 29999, "2026-07-20", 30, None),
        ("M19", "Samsung Galaxy A37", "Samsung", "mid", 33999, "2026-08-03", 32, None),
        ("M20", "OnePlus Nord 6", "OnePlus", "mid", 31999, "2026-08-17", 30, None),
        # Flagships: shorter hype cycles and higher value per unit.
        ("F01", "Samsung Galaxy S25", "Samsung", "flagship", 74999, "2025-02-24", 26, "F11"),
        ("F02", "iPhone 16", "Apple", "flagship", 79900, "2025-03-17", 28, "F12"),
        ("F03", "OnePlus 13", "OnePlus", "flagship", 69999, "2025-04-21", 24, "F13"),
        ("F04", "Vivo X200", "Vivo", "flagship", 65999, "2025-05-19", 24, "F14"),
        ("F05", "Pixel 9 Pro", "Google", "flagship", 109999, "2025-06-23", 26, "F15"),
        ("F06", "Xiaomi 15", "Xiaomi", "flagship", 64999, "2025-07-21", 24, "F16"),
        ("F07", "iPhone 16 Pro", "Apple", "flagship", 119900, "2025-08-18", 28, None),
        ("F08", "Samsung Galaxy Z Flip7", "Samsung", "flagship", 99999, "2025-09-15", 24, None),
        ("F09", "OnePlus Open 2", "OnePlus", "flagship", 119999, "2025-10-20", 22, None),
        ("F10", "iPhone 17", "Apple", "flagship", 84900, "2025-11-17", 28, None),
        ("F11", "Samsung Galaxy S26", "Samsung", "flagship", 79999, "2026-01-26", 26, None),
        ("F12", "iPhone 17 Pro", "Apple", "flagship", 124900, "2026-02-16", 28, None),
        ("F13", "OnePlus 14", "OnePlus", "flagship", 72999, "2026-03-16", 24, None),
        ("F14", "Vivo X210", "Vivo", "flagship", 69999, "2026-04-13", 24, None),
        ("F15", "Pixel 10 Pro", "Google", "flagship", 114999, "2026-05-11", 26, None),
        ("F16", "Xiaomi 16", "Xiaomi", "flagship", 67999, "2026-06-08", 24, None),
        ("F17", "iPhone 17 Pro Max", "Apple", "flagship", 139900, "2026-06-29", 28, None),
        ("F18", "Samsung Galaxy Z Fold8", "Samsung", "flagship", 149999, "2026-07-20", 24, None),
        ("F19", "OnePlus Open 3", "OnePlus", "flagship", 124999, "2026-08-03", 22, None),
        ("F20", "Vivo X210 Pro", "Vivo", "flagship", 79999, "2026-08-17", 24, None),
    ]
    columns = [
        "model_id", "model_name", "brand", "price_tier", "selling_price", "launch_date",
        "expected_lifecycle_weeks", "successor_model_id", "launch_date_status",
    ]
    catalog = pd.DataFrame(
        [row + ("confirmed" if row[0] in {"B01", "B11", "M01", "M11", "F01", "F11"} else "rumoured",)
         for row in rows],
        columns=columns,
    )
    margin_ranges = {
        "budget": (0.05, 0.08),
        "mid": (0.10, 0.15),
        "flagship": (0.15, 0.22),
    }
    rng = np.random.default_rng(2025)
    catalog["margin_pct"] = [
        round(rng.uniform(*margin_ranges[price_tier]), 4)
        for price_tier in catalog["price_tier"]
    ]
    catalog["acquisition_cost"] = np.rint(
        catalog["selling_price"] * (1 - catalog["margin_pct"])
    ).astype(int)
    return catalog


if __name__ == "__main__":
    catalog = generate_catalog()
    print(catalog.to_string(index=False))
    print("\nCatalog size:", len(catalog))
    print("\nModels by price tier:\n", catalog["price_tier"].value_counts().to_string())