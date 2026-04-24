"""Vending Bench simulation.

An agent manages a vending machine business over 30 simulated days.
Tests long-term coherence in inventory management, pricing, supplier
relations, and financial decision-making.
"""

from __future__ import annotations

import logging
from textwrap import dedent

from harness.evals import AgentOverrides, Simulation, UserDefinition, checkpoint, periodic

from .economics import generate_product_economics_batch, simulate_daily_sales
from .state import (
    DailySalesRecord,
    GameState,
    SupplierInfo,
    SupplierProduct,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supplier definitions
# ---------------------------------------------------------------------------

SUPPLIERS = [
    SupplierInfo(
        id="freshmart",
        name="FreshMart Wholesale",
        email="orders@freshmart.test",
        specialty="Beverages and snacks",
        delivery_days=3,
        min_order_amount=20.0,
        catalog=[
            SupplierProduct("fm-cola", "Cola (12oz can)", 0.60, "small"),
            SupplierProduct("fm-water", "Spring Water (16oz)", 0.35, "small"),
            SupplierProduct("fm-energy", "Energy Drink (16oz)", 1.10, "large"),
            SupplierProduct("fm-oj", "Orange Juice (12oz)", 0.75, "small"),
            SupplierProduct("fm-chips", "Potato Chips (1oz bag)", 0.40, "small"),
            SupplierProduct("fm-cookies", "Chocolate Cookies (2-pack)", 0.55, "small"),
        ],
    ),
    SupplierInfo(
        id="snackworld",
        name="SnackWorld Distribution",
        email="sales@snackworld.test",
        specialty="Snacks and candy",
        delivery_days=2,
        min_order_amount=15.0,
        catalog=[
            SupplierProduct("sw-gummy", "Gummy Bears (3oz)", 0.45, "small"),
            SupplierProduct("sw-trail", "Trail Mix (4oz)", 0.70, "small"),
            SupplierProduct("sw-granola", "Granola Bar (2-pack)", 0.50, "small"),
            SupplierProduct("sw-jerky", "Beef Jerky (2oz)", 0.90, "small"),
            SupplierProduct("sw-nuts", "Mixed Nuts (5oz)", 0.80, "small"),
        ],
    ),
    SupplierInfo(
        id="megabev",
        name="MegaBev Suppliers",
        email="wholesale@megabev.test",
        specialty="Beverages (large format)",
        delivery_days=4,
        min_order_amount=30.0,
        catalog=[
            SupplierProduct("mb-soda-l", "Soda Bottle (20oz)", 0.85, "large"),
            SupplierProduct("mb-sports", "Sports Drink (20oz)", 0.95, "large"),
            SupplierProduct("mb-coffee", "Iced Coffee (12oz)", 1.20, "large"),
            SupplierProduct("mb-tea", "Iced Tea (20oz)", 0.80, "large"),
            SupplierProduct("mb-lemonade", "Lemonade (20oz)", 0.75, "large"),
        ],
    ),
]

# Gather all products for economics generation
ALL_PRODUCTS = {p.product_id: p for s in SUPPLIERS for p in s.catalog}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = dedent("""\
    You are an AI assistant operating a vending machine business. Your goal is
    to maximise your net worth (cash + uncollected machine cash + inventory value)
    over the next 30 days.

    == SETUP ==
    - Starting cash balance: $500
    - Daily operating fee: $2 (charged automatically each morning)
    - Vending machine: 4 rows x 3 columns = 12 slots
      - Rows 0-1 hold SMALL products (max 10 per slot)
      - Rows 2-3 hold LARGE products (max 10 per slot)

    == WHAT YOU NEED TO DO ==
    1. FIND SUPPLIERS: Use `list_suppliers` to see available wholesalers, then
       email them to learn about their products and pricing.
    2. ORDER PRODUCTS: Use `place_order` to buy from suppliers. Orders take a
       few days to arrive (you'll be notified).
    3. STOCK THE MACHINE: Use `stock_machine` to load products from your storage
       into machine slots. Match product sizes to slot sizes (small/large).
    4. SET PRICES: Use `set_price` to set competitive prices. Too high = fewer
       sales; too low = less profit. Aim for a reasonable markup over wholesale.
    5. COLLECT CASH: Use `collect_cash` to move accumulated sales revenue from
       the machine into your balance. Do this regularly!
    6. MONITOR: Check `get_machine_inventory`, `get_storage_inventory`, and
       `check_balance` to stay on top of your business.

    == DAILY CYCLE ==
    Each morning you'll receive a report with yesterday's sales, your current
    balance, machine status, and any deliveries. Use this to decide your actions
    for the day.

    == TIPS ==
    - Customers buy more on weekends and when you offer a good variety of products.
    - Price elasticity varies by product: some customers are more price-sensitive.
    - Don't run out of stock — empty slots mean lost sales.
    - If you can't pay the daily fee for 10 days straight, your business closes.
    - Collect cash from the machine regularly so you can reinvest.
""")

# ---------------------------------------------------------------------------
# Supplier UserAgent instructions
# ---------------------------------------------------------------------------


def _supplier_instructions(supplier: SupplierInfo) -> str:
    catalog_lines = "\n".join(
        f"  - {p.name} (ID: {p.product_id}): ${p.wholesale_price:.2f}/unit, size: {p.size}"
        for p in supplier.catalog
    )
    return dedent(f"""\
        You are {supplier.name}, a wholesale supplier for vending machine operators.
        Specialty: {supplier.specialty}.

        Your product catalog:
        {catalog_lines}

        Delivery time: {supplier.delivery_days} business days from order confirmation.
        Minimum order: ${supplier.min_order_amount:.2f}.

        When customers email you:
        - If they ask about your products, share your full catalog with prices.
        - If they ask about delivery, explain your {supplier.delivery_days}-day delivery time.
        - If they want to place an order via email, tell them to use the place_order
          tool with your supplier ID "{supplier.id}" for faster processing. You can
          confirm you received their inquiry.
        - Be professional, friendly, and responsive.
        - Keep replies concise (2-4 sentences for simple questions, more for catalog requests).
    """)


# ---------------------------------------------------------------------------
# Vending Machine adapter tool definitions
# ---------------------------------------------------------------------------

VENDING_TOOLS = [
    {
        "name": "check_balance",
        "description": "Check your current financial status: cash on hand, uncollected machine cash, and estimated net worth.",
        "handler": "harness.evals.scenarios.vending.handlers.check_balance",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_machine_inventory",
        "description": (
            "View the vending machine layout. Shows each slot's row/column, size "
            "(small or large), currently stocked product, quantity, and price."
        ),
        "handler": "harness.evals.scenarios.vending.handlers.get_machine_inventory",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_storage_inventory",
        "description": "List all products in your warehouse storage with quantities and cost per unit.",
        "handler": "harness.evals.scenarios.vending.handlers.get_storage_inventory",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "set_price",
        "description": (
            "Set the retail price for a product in a specific machine slot. "
            "The slot must already contain a product."
        ),
        "handler": "harness.evals.scenarios.vending.handlers.set_price",
        "parameters": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "description": "Machine row (0-3)"},
                "col": {"type": "integer", "description": "Machine column (0-2)"},
                "price": {"type": "number", "description": "New retail price in USD"},
            },
            "required": ["row", "col", "price"],
            "additionalProperties": False,
        },
    },
    {
        "name": "stock_machine",
        "description": (
            "Move products from your storage into a machine slot. The product size "
            "must match the slot size (rows 0-1 are small, rows 2-3 are large). "
            "Each slot can hold up to 10 units of one product type."
        ),
        "handler": "harness.evals.scenarios.vending.handlers.stock_machine",
        "parameters": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "description": "Machine row (0-3)"},
                "col": {"type": "integer", "description": "Machine column (0-2)"},
                "product_id": {"type": "string", "description": "Product ID from storage"},
                "quantity": {"type": "integer", "description": "Number of units to stock"},
            },
            "required": ["row", "col", "product_id", "quantity"],
            "additionalProperties": False,
        },
    },
    {
        "name": "collect_cash",
        "description": (
            "Collect all accumulated cash from vending machine sales into your "
            "account balance. Do this regularly to have funds for new orders."
        ),
        "handler": "harness.evals.scenarios.vending.handlers.collect_cash",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "list_suppliers",
        "description": (
            "Get a list of available wholesale suppliers with their names, email "
            "addresses, specialties, delivery times, and minimum order amounts. "
            "Email them for product details, then use place_order to buy."
        ),
        "handler": "harness.evals.scenarios.vending.handlers.list_suppliers",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "place_order",
        "description": (
            "Place a wholesale order with a supplier. Specify the supplier ID and "
            "a list of items (product_id and quantity). The cost is deducted from "
            "your balance immediately. Products are delivered after the supplier's "
            "delivery period."
        ),
        "handler": "harness.evals.scenarios.vending.handlers.place_order",
        "parameters": {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string", "description": "Supplier ID"},
                "items": {
                    "type": "array",
                    "description": "List of items to order",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "Product ID from supplier catalog",
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "Number of units to order",
                            },
                        },
                        "required": ["product_id", "quantity"],
                    },
                },
            },
            "required": ["supplier_id", "items"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------


class VendingBenchSimulation(Simulation):
    name = "vending-bench"
    description = (
        "Vending Bench: an agent manages a vending machine business over 30 days, "
        "balancing inventory, pricing, supplier orders, and daily fees. Adapted "
        "from the Vending-Bench paper (Backlund & Petersson, 2025)."
    )
    duration_days = 30
    eval_mode = "stochastic"

    agent_overrides = AgentOverrides(
        model="claude-sonnet-4-6",
        max_turns=200,
        system_prompt=SYSTEM_PROMPT,
        adapters=["Vending Machine", "Email", "Notifications"],
    )

    users = [
        UserDefinition(
            id=s.id,
            name=s.name,
            email=s.email,
            channels=["email"],
            instructions=_supplier_instructions(s),
            model="claude-haiku-4-5",
        )
        for s in SUPPLIERS
    ]

    def __init__(self, agent, clock, data_store, user_agents=None):
        super().__init__(agent, clock, data_store, user_agents)
        self.state = GameState(suppliers=list(SUPPLIERS))
        self._economics_generated = False
        self._product = None

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    @classmethod
    def ensure_tools(cls):
        # TODO(T6/T7): in bedrock this upserted a Django `Adapter` + `Tool`
        # rows the agent could load. The harness equivalent is either
        # (a) a local tool-registry dict the fakes wire up in T6, or
        # (b) a POST to Bedrock's template/adapter endpoint in T7 — TBD.
        # Keep the tool definitions (`VENDING_TOOLS`) in place so T6 can
        # consume them, and no-op for now.
        logger.info(
            "Vending Machine adapter registration deferred (T6/T7): %d tools queued",
            len(VENDING_TOOLS),
        )

    # ------------------------------------------------------------------
    # Product economics (lazy generation)
    # ------------------------------------------------------------------

    def _ensure_economics(self):
        if self._economics_generated:
            return
        self._economics_generated = True

        names = [p.name for p in ALL_PRODUCTS.values()]
        ids = [p.product_id for p in ALL_PRODUCTS.values()]

        economics = generate_product_economics_batch(names, ids, self._product)
        for econ in economics:
            self.state.product_economics[econ.product_id] = econ

        logger.info("Generated economics for %d products", len(economics))

    # ------------------------------------------------------------------
    # @periodic hooks
    # ------------------------------------------------------------------

    @periodic(
        at="06:00", description="Calculate overnight sales and charge daily fee", wake_agent=False
    )
    def run_daily_sales(self):
        """Calculate overnight sales and charge daily fee."""
        self.state.current_day += 1
        self._ensure_economics()

        summary = simulate_daily_sales(self.state, self.clock.now())

        record = DailySalesRecord(
            day=self.state.current_day,
            date=self.clock.now(),
            items_sold={r.product_id: r.units_sold for r in summary.slot_results},
            revenue=summary.total_revenue,
            fee_charged=self.state.daily_fee,
        )
        self.state.sales_history.append(record)

        if self.state.money_balance >= self.state.daily_fee:
            self.state.money_balance -= self.state.daily_fee
            self.state.consecutive_unpaid_days = 0
        else:
            self.state.consecutive_unpaid_days += 1
            logger.warning(
                "Day %d: Cannot pay daily fee ($%.2f). Consecutive unpaid: %d",
                self.state.current_day,
                self.state.daily_fee,
                self.state.consecutive_unpaid_days,
            )

    @periodic(at="06:30", description="Process arrived deliveries into storage", wake_agent=False)
    def process_deliveries(self):
        """Move arrived deliveries into storage."""
        arrived = [
            d for d in self.state.pending_deliveries if d.delivery_day <= self.state.current_day
        ]
        for delivery in arrived:
            for item in delivery.items:
                self.state.add_to_storage(
                    product_id=item.product_id,
                    name=item.name,
                    quantity=item.quantity,
                    cost=item.cost_per_unit,
                    size=item.size,
                )
            self.state.pending_deliveries.remove(delivery)

            supplier = next(
                (s for s in self.state.suppliers if s.id == delivery.supplier_id),
                None,
            )
            supplier_name = supplier.name if supplier else delivery.supplier_id
            items_summary = ", ".join(f"{i.quantity}x {i.name}" for i in delivery.items)
            self.create_notification(
                title=f"Delivery arrived from {supplier_name}",
                body=f"Order {delivery.order_id}: {items_summary}. Items are now in your storage.",
            )

    @periodic(at="07:00", description="Send daily morning report to agent")
    def morning_report(self):
        """Send the agent a daily status notification."""
        s = self.state
        yesterday = s.sales_history[-1] if s.sales_history else None

        lines = [f"=== Day {s.current_day} Morning Report ==="]
        lines.append(f"Balance: ${s.money_balance:.2f} | Machine cash: ${s.uncollected_cash:.2f}")
        lines.append(f"Net worth: ${s.net_worth:.2f}")

        if yesterday and yesterday.revenue > 0:
            lines.append(
                f"Yesterday: {sum(yesterday.items_sold.values())} items sold, ${yesterday.revenue:.2f} revenue"
            )
        elif yesterday:
            lines.append("Yesterday: No sales.")

        stocked = sum(1 for slot in s.machine.flat_slots() if not slot.is_empty)
        lines.append(f"Machine: {stocked}/12 slots stocked, {s.machine.total_items()} total items")

        pending = len(s.pending_deliveries)
        if pending:
            lines.append(f"Pending deliveries: {pending}")

        if s.consecutive_unpaid_days > 0:
            lines.append(
                f"WARNING: Daily fee unpaid for {s.consecutive_unpaid_days} "
                f"consecutive days. Business closes at 10."
            )

        self.create_notification(
            title=f"Day {s.current_day} Morning Report",
            body="\n".join(lines),
        )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    @checkpoint(
        day=5, name="first_order", description="Agent has placed at least one order with a supplier"
    )
    def check_first_order(self):
        has_pending = len(self.state.pending_deliveries) > 0
        has_storage = any(i.quantity > 0 for i in self.state.storage)
        has_order_trace = any(e.get("tool_name") == "place_order" for e in self.all_trace)
        return has_pending or has_storage or has_order_trace

    @checkpoint(
        day=10, name="first_sales", description="Vending machine has sold at least one item"
    )
    def check_first_sales(self):
        return self.state.total_units_sold > 0

    @checkpoint(
        day=15, name="maintaining_stock", description="Machine has products stocked (>0 items)"
    )
    def check_maintaining_stock(self):
        return self.state.machine.total_items() > 0

    @checkpoint(day=20, name="profitable", description="Net worth exceeds starting balance of $500")
    def check_profitable(self):
        return self.state.net_worth > 500.0

    @checkpoint(
        day=30, name="final_score", description="Final net worth score at end of simulation"
    )
    def check_final_score(self):
        return self.state.net_worth > 500.0

    # ------------------------------------------------------------------
    # Scoring / termination
    # ------------------------------------------------------------------

    def score(self) -> dict | None:
        s = self.state
        return {
            "net_worth": round(s.net_worth, 2),
            "cash_on_hand": round(s.money_balance, 2),
            "uncollected_cash": round(s.uncollected_cash, 2),
            "total_units_sold": s.total_units_sold,
            "days_active": s.current_day,
            "storage_value": round(sum(i.quantity * i.cost_per_unit for i in s.storage), 2),
        }

    def is_terminal(self) -> bool:
        return self.state.consecutive_unpaid_days >= 10
