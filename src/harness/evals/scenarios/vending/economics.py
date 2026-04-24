"""Customer purchase simulation using price elasticity of demand.

Each simulated day, for every stocked slot in the vending machine, we:
1. Look up (or generate) the product's economics (base price, elasticity, base sales).
2. Compute a demand adjustment from the current price.
3. Apply day-of-week and variety multipliers plus random noise.
4. Cap sales at available inventory.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime

from .state import GameState, ProductEconomics

logger = logging.getLogger(__name__)

DAY_OF_WEEK_MULTIPLIER = {
    0: 0.9,  # Monday
    1: 0.95,  # Tuesday
    2: 1.0,  # Wednesday
    3: 1.0,  # Thursday
    4: 1.15,  # Friday
    5: 1.3,  # Saturday
    6: 1.2,  # Sunday
}


@dataclass
class SlotSaleResult:
    slot_row: int
    slot_col: int
    product_id: str
    product_name: str
    units_sold: int
    revenue: float
    price: float


@dataclass
class DailySalesSummary:
    day: int
    date: datetime
    slot_results: list[SlotSaleResult]
    total_units: int
    total_revenue: float


def _variety_multiplier(unique_products: int) -> float:
    """Reward 4-8 unique products, penalise extremes (min 0.5x)."""
    if unique_products == 0:
        return 0.0
    if unique_products <= 3:
        return 0.6 + 0.1 * unique_products
    if unique_products <= 8:
        return 1.0 + 0.02 * (unique_products - 4)
    return max(0.5, 1.1 - 0.05 * (unique_products - 8))


def simulate_daily_sales(state: GameState, sim_date: datetime) -> DailySalesSummary:
    """Run the daily customer-purchase simulation and mutate *state* in place."""
    dow = sim_date.weekday()
    dow_mult = DAY_OF_WEEK_MULTIPLIER.get(dow, 1.0)

    unique = len(state.machine.unique_products())
    variety_mult = _variety_multiplier(unique)

    slot_results: list[SlotSaleResult] = []
    total_units = 0
    total_revenue = 0.0

    for slot in state.machine.flat_slots():
        if slot.is_empty:
            continue

        econ = state.product_economics.get(slot.product_id)
        if econ is None:
            continue

        price_ratio = (slot.price - econ.base_price) / econ.base_price if econ.base_price else 0
        elasticity_factor = max(0.0, 1.0 - econ.price_elasticity * price_ratio)

        base = econ.base_daily_sales * elasticity_factor * dow_mult * variety_mult
        noise = random.gauss(1.0, 0.15)
        raw = base * noise
        units = max(0, min(math.floor(raw), slot.quantity))

        if units > 0:
            revenue = units * slot.price
            slot.sell(units)
            state.uncollected_cash += revenue
            total_units += units
            total_revenue += revenue
            slot_results.append(
                SlotSaleResult(
                    slot_row=slot.row,
                    slot_col=slot.col,
                    product_id=slot.product_id or "",
                    product_name=slot.product_name or "",
                    units_sold=units,
                    revenue=revenue,
                    price=slot.price,
                )
            )

    state.total_units_sold += total_units
    return DailySalesSummary(
        day=state.current_day,
        date=sim_date,
        slot_results=slot_results,
        total_units=total_units,
        total_revenue=total_revenue,
    )


def generate_product_economics_batch(
    product_names: list[str],
    product_ids: list[str],
    product: object | None = None,
) -> list[ProductEconomics]:
    """Use an LLM to generate realistic economics for a batch of product names.

    Returns one ``ProductEconomics`` per product.  Falls back to reasonable
    hard-coded defaults when the LLM call fails.
    """
    if not product_names:
        return []

    try:
        return _llm_generate(product_names, product_ids, product)
    except Exception:
        logger.exception("LLM economics generation failed – using defaults")
        return _fallback_economics(product_names, product_ids)


def _llm_generate(
    names: list[str], ids: list[str], product: object | None
) -> list[ProductEconomics]:
    # Bedrock's version branched on Claude vs OpenAI through its in-tree
    # `core.llm` abstraction. In the harness everything goes through
    # `harness.core.llm.complete` (OpenRouter OpenAI-compat).
    from harness.core.llm import complete

    system = (
        "You are an economics assistant. For each vending machine product below, "
        "return a JSON array of objects with keys: name, base_price (typical retail "
        "price in USD), base_daily_sales (expected units/day at base_price, float "
        "between 0.5 and 5), price_elasticity (float 0.5-3, higher = more price "
        "sensitive). Use realistic values for US vending machines."
    )
    user = "Products:\n" + "\n".join(f"- {n}" for n in names)

    response = complete(
        model="openai/gpt-4o-mini",
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = response.text or ""

    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array in LLM response")

    items = json.loads(text[start:end])
    results: list[ProductEconomics] = []
    for item, pid in zip(items, ids):
        results.append(
            ProductEconomics(
                product_id=pid,
                name=item.get("name", ""),
                base_price=float(item.get("base_price", 2.0)),
                base_daily_sales=float(item.get("base_daily_sales", 1.5)),
                price_elasticity=float(item.get("price_elasticity", 1.5)),
            )
        )
    return results


def _fallback_economics(names: list[str], ids: list[str]) -> list[ProductEconomics]:
    defaults = {
        "small": (1.50, 2.0, 1.5),
        "large": (2.50, 1.2, 1.2),
    }
    results = []
    for name, pid in zip(names, ids):
        base_price, base_sales, elasticity = defaults.get("small", (1.50, 2.0, 1.5))
        results.append(
            ProductEconomics(
                product_id=pid,
                name=name,
                base_price=base_price,
                base_daily_sales=base_sales,
                price_elasticity=elasticity,
            )
        )
    return results
