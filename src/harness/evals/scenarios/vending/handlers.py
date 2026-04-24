"""Tool handlers for the Vending Machine adapter.

Each handler follows the standard signature ``(args, agent, tool) -> str``
and accesses simulation state via ``get_simulation()``.  Tracing is handled
automatically by the patched ``Tool.call`` wrapper in mock_clients.
"""

from __future__ import annotations

import json
import uuid

from harness.evals.context import get_simulation


def _state():
    from .simulation import VendingBenchSimulation

    sim = get_simulation()
    if not isinstance(sim, VendingBenchSimulation):
        raise RuntimeError("Active simulation is not a VendingBenchSimulation")
    return sim.state


def _json(data) -> str:
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# check_balance
# ---------------------------------------------------------------------------


def check_balance(args: dict, agent, tool) -> str:
    state = _state()
    return _json(
        {
            "cash_on_hand": round(state.money_balance, 2),
            "uncollected_machine_cash": round(state.uncollected_cash, 2),
            "estimated_net_worth": round(state.net_worth, 2),
            "daily_fee": state.daily_fee,
        }
    )


# ---------------------------------------------------------------------------
# get_machine_inventory
# ---------------------------------------------------------------------------


def get_machine_inventory(args: dict, agent, tool) -> str:
    state = _state()
    rows = []
    for row_slots in state.machine.slots:
        row_data = []
        for s in row_slots:
            row_data.append(
                {
                    "row": s.row,
                    "col": s.col,
                    "size": s.size,
                    "product": s.product_name,
                    "product_id": s.product_id,
                    "quantity": s.quantity,
                    "max_quantity": s.max_quantity,
                    "price": s.price,
                }
            )
        rows.append(row_data)
    return _json({"machine_rows": rows, "total_items": state.machine.total_items()})


# ---------------------------------------------------------------------------
# get_storage_inventory
# ---------------------------------------------------------------------------


def get_storage_inventory(args: dict, agent, tool) -> str:
    state = _state()
    items = [
        {
            "product_id": i.product_id,
            "name": i.name,
            "quantity": i.quantity,
            "cost_per_unit": round(i.cost_per_unit, 2),
            "size": i.size,
        }
        for i in state.storage
        if i.quantity > 0
    ]
    return _json({"storage_items": items, "total_items": sum(i.quantity for i in state.storage)})


# ---------------------------------------------------------------------------
# set_price
# ---------------------------------------------------------------------------


def set_price(args: dict, agent, tool) -> str:
    state = _state()
    row = int(args["row"])
    col = int(args["col"])
    price = float(args["price"])

    if row < 0 or row >= 4 or col < 0 or col >= 3:
        return "Error: Invalid slot. Rows 0-3, columns 0-2."
    if price < 0:
        return "Error: Price cannot be negative."

    slot = state.machine.slots[row][col]
    if slot.product_id is None:
        return f"Error: Slot ({row},{col}) is empty. Stock it first."

    old_price = slot.price
    slot.price = price
    return f"Price for '{slot.product_name}' at ({row},{col}) changed from ${old_price:.2f} to ${price:.2f}."


# ---------------------------------------------------------------------------
# stock_machine
# ---------------------------------------------------------------------------


def stock_machine(args: dict, agent, tool) -> str:
    state = _state()
    row = int(args["row"])
    col = int(args["col"])
    product_id = str(args["product_id"])
    quantity = int(args["quantity"])

    if row < 0 or row >= 4 or col < 0 or col >= 3:
        return "Error: Invalid slot. Rows 0-3, columns 0-2."

    storage_item = state.get_storage_item(product_id)
    if storage_item is None:
        return f"Error: Product '{product_id}' not found in storage."
    if storage_item.quantity < quantity:
        return (
            f"Error: Only {storage_item.quantity} units of '{storage_item.name}' "
            f"in storage, but requested {quantity}."
        )

    slot = state.machine.slots[row][col]
    if slot.size != storage_item.size:
        return (
            f"Error: Slot ({row},{col}) is a {slot.size} slot, but "
            f"'{storage_item.name}' is a {storage_item.size} product."
        )

    try:
        added = slot.stock(product_id, storage_item.name, quantity)
    except ValueError as e:
        return f"Error: {e}"

    storage_item.quantity -= added
    return (
        f"Stocked {added} units of '{storage_item.name}' into slot ({row},{col}). "
        f"Slot now has {slot.quantity}/{slot.max_quantity}. "
        f"Storage remaining: {storage_item.quantity}."
    )


# ---------------------------------------------------------------------------
# collect_cash
# ---------------------------------------------------------------------------


def collect_cash(args: dict, agent, tool) -> str:
    state = _state()
    collected = state.uncollected_cash
    state.money_balance += collected
    state.uncollected_cash = 0.0
    return (
        f"Collected ${collected:.2f} from the machine. " f"New balance: ${state.money_balance:.2f}."
    )


# ---------------------------------------------------------------------------
# list_suppliers
# ---------------------------------------------------------------------------


def list_suppliers(args: dict, agent, tool) -> str:
    state = _state()
    suppliers = []
    for s in state.suppliers:
        catalog = [
            {
                "product_id": p.product_id,
                "name": p.name,
                "wholesale_price": p.wholesale_price,
                "size": p.size,
            }
            for p in s.catalog
        ]
        suppliers.append(
            {
                "id": s.id,
                "name": s.name,
                "email": s.email,
                "specialty": s.specialty,
                "delivery_days": s.delivery_days,
                "min_order_amount": s.min_order_amount,
                "catalog": catalog,
            }
        )
    return _json({"suppliers": suppliers})


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------


def place_order(args: dict, agent, tool) -> str:
    state = _state()
    supplier_id = str(args["supplier_id"])
    items_arg = args.get("items", [])

    supplier = next((s for s in state.suppliers if s.id == supplier_id), None)
    if supplier is None:
        return f"Error: Supplier '{supplier_id}' not found."

    catalog_lookup = {p.product_id: p for p in supplier.catalog}
    order_items = []
    total_cost = 0.0

    for item in items_arg:
        pid = str(item["product_id"])
        qty = int(item["quantity"])
        prod = catalog_lookup.get(pid)
        if prod is None:
            return f"Error: Product '{pid}' not in {supplier.name}'s catalog."
        line_cost = prod.wholesale_price * qty
        total_cost += line_cost
        order_items.append(
            {
                "product_id": pid,
                "name": prod.name,
                "quantity": qty,
                "cost_per_unit": prod.wholesale_price,
                "size": prod.size,
            }
        )

    if total_cost < supplier.min_order_amount:
        return (
            f"Error: Order total ${total_cost:.2f} is below "
            f"{supplier.name}'s minimum of ${supplier.min_order_amount:.2f}."
        )

    if total_cost > state.money_balance:
        return (
            f"Error: Insufficient funds. Order costs ${total_cost:.2f} "
            f"but balance is ${state.money_balance:.2f}."
        )

    from .state import OrderLineItem, PendingDelivery

    state.money_balance -= total_cost
    order_id = uuid.uuid4().hex[:8]
    delivery = PendingDelivery(
        order_id=order_id,
        supplier_id=supplier_id,
        items=[
            OrderLineItem(
                product_id=oi["product_id"],
                name=oi["name"],
                quantity=oi["quantity"],
                cost_per_unit=oi["cost_per_unit"],
                size=oi["size"],
            )
            for oi in order_items
        ],
        total_cost=total_cost,
        ordered_day=state.current_day,
        delivery_day=state.current_day + supplier.delivery_days,
    )
    state.pending_deliveries.append(delivery)

    return _json(
        {
            "status": "Order placed",
            "order_id": order_id,
            "supplier": supplier.name,
            "total_cost": round(total_cost, 2),
            "items": order_items,
            "estimated_delivery_day": delivery.delivery_day,
            "new_balance": round(state.money_balance, 2),
        }
    )
