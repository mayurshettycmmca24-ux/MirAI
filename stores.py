"""Store profiling data for the MobiMart inventory allocation model.

The profiles are intentionally deterministic. In an allocation system, a store's
commercial profile should be explainable to a business owner and reproducible
between runs; randomness belongs in the later demand simulation, not here.
"""

from dataclasses import asdict, dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class StoreProfile:
    """Commercial characteristics used to shape demand by store.

    Each input profile is explicit, while ``catchment_income_level`` and
    ``sales_mix_profile`` are derived below so the assumptions remain visible.
    """

    store_id: str
    store_name: str
    city: str  # Geographic market used to compare metro and tier-2/3 demand.
    city_tier: str  # Metro versus tier-2/3 affects purchasing power and traffic.
    locality: str  # Specific neighborhood explains differences within Bangalore.
    catchment_income_level: str  # Local purchasing power is a better price proxy than city alone.
    footfall_index: int  # Relative weekly traffic proxy; 100 represents a busy reference store.
    sales_mix_profile: str  # Price mix derived from income and traffic, not sampled randomly.


def _income_level(city_tier: str, locality: str, footfall_index: int) -> str:
    """Derive catchment income from market tier and known locality signals.

    Bangalore's premium neighborhoods are treated as high-income catchments;
    otherwise traffic helps distinguish a stronger mid-market catchment from a
    lower-income one. Tier-2/3 markets default to mid unless their traffic is
    distinctly lower, reflecting a broader but more price-sensitive audience.
    """
    premium_localities = {"Koramangala", "Indiranagar", "Whitefield", "Jayanagar"}
    if city_tier == "tier-1" and locality in premium_localities:
        return "high"
    if footfall_index >= 105:
        return "mid"
    return "low"


def _sales_mix(city_tier: str, income_level: str, footfall_index: int) -> str:
    """Translate catchment economics and traffic into the expected price mix."""
    if income_level == "high" and footfall_index >= 105:
        return "flagship-heavy"
    if income_level == "low" or footfall_index < 80:
        return "budget-heavy"
    return "mid-range"


def generate_stores() -> pd.DataFrame:
    """Return the 25 MobiMart stores as a pandas DataFrame.

    The eight Bangalore locations have distinct neighborhood traffic patterns.
    The remaining seventeen stores cover representative Karnataka tier-2/3
    markets. Explicit traffic values are a planning index, not claimed census
    measurements; they provide stable relative weights for demand generation.
    """
    raw_stores = [
        ("BLR-01", "MobiMart Jayanagar", "Bangalore", "Jayanagar", 122),
        ("BLR-02", "MobiMart Koramangala", "Bangalore", "Koramangala", 138),
        ("BLR-03", "MobiMart Indiranagar", "Bangalore", "Indiranagar", 130),
        ("BLR-04", "MobiMart Whitefield", "Bangalore", "Whitefield", 128),
        ("BLR-05", "MobiMart Malleshwaram", "Bangalore", "Malleshwaram", 104),
        ("BLR-06", "MobiMart Rajajinagar", "Bangalore", "Rajajinagar", 98),
        ("BLR-07", "MobiMart Yelahanka", "Bangalore", "Yelahanka", 91),
        ("BLR-08", "MobiMart Electronic City", "Bangalore", "Electronic City", 112),
        ("MYS-01", "MobiMart Devaraja Mohalla", "Mysore", "Devaraja Mohalla", 96),
        ("MYS-02", "MobiMart Kuvempunagar", "Mysore", "Kuvempunagar", 82),
        ("HUB-01", "MobiMart Vidya Nagar", "Hubli", "Vidya Nagar", 101),
        ("HUB-02", "MobiMart Gokul Road", "Hubli", "Gokul Road", 88),
        ("TUM-01", "MobiMart MG Road", "Tumkur", "MG Road", 86),
        ("TUM-02", "MobiMart SS Puram", "Tumkur", "SS Puram", 74),
        ("DAV-01", "MobiMart PJ Extension", "Davangere", "PJ Extension", 93),
        ("DAV-02", "MobiMart MCC B Block", "Davangere", "MCC B Block", 79),
        ("BEL-01", "MobiMart Cantonment", "Belagavi", "Cantonment", 99),
        ("BEL-02", "MobiMart Tilakwadi", "Belagavi", "Tilakwadi", 84),
        ("MAN-01", "MobiMart Court Road", "Mangalore", "Court Road", 116),
        ("MAN-02", "MobiMart Kottara", "Mangalore", "Kottara", 89),
        ("SHI-01", "MobiMart Vidya Nagar", "Shivamogga", "Vidya Nagar", 87),
        ("KAL-01", "MobiMart KCD Road", "Kalaburagi", "KCD Road", 81),
        ("HOS-01", "MobiMart Hosur Road", "Hassan", "Hosur Road", 76),
        ("UDU-01", "MobiMart Udupi Main Road", "Udupi", "Main Road", 90),
        ("KOL-01", "MobiMart Station Road", "Kolar", "Station Road", 68),
    ]

    profiles: List[StoreProfile] = []
    for store_id, store_name, city, locality, footfall_index in raw_stores:
        city_tier = "tier-1" if city == "Bangalore" else "tier-2/3"
        income_level = _income_level(city_tier, locality, footfall_index)
        profiles.append(
            StoreProfile(
                store_id=store_id,
                store_name=store_name,
                city=city,
                city_tier=city_tier,
                locality=locality,
                catchment_income_level=income_level,
                footfall_index=footfall_index,
                sales_mix_profile=_sales_mix(city_tier, income_level, footfall_index),
            )
        )

    return pd.DataFrame(asdict(profile) for profile in profiles)


if __name__ == "__main__":
    stores = generate_stores()
    print(stores.to_string(index=False))
    print("\nStore count:", len(stores))
    print("\nStores by tier:\n", stores["city_tier"].value_counts().to_string())
    print("\nSales mix by tier:\n", stores.groupby(["city_tier", "sales_mix_profile"]).size().to_string())