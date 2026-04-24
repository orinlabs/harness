from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ProductSize = Literal["small", "large"]


@dataclass
class VendingSlot:
    row: int
    col: int
    size: ProductSize
    product_id: str | None = None
    product_name: str | None = None
    quantity: int = 0
    max_quantity: int = 10
    price: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.product_id is None or self.quantity == 0

    def stock(self, product_id: str, product_name: str, quantity: int) -> int:
        """Add items to the slot. Returns the number actually added (capped at capacity)."""
        if self.product_id is not None and self.product_id != product_id:
            raise ValueError(
                f"Slot ({self.row},{self.col}) already holds '{self.product_name}'. "
                f"Clear it first or choose an empty slot."
            )
        self.product_id = product_id
        self.product_name = product_name
        can_add = self.max_quantity - self.quantity
        added = min(quantity, can_add)
        self.quantity += added
        return added

    def sell(self, quantity: int = 1) -> int:
        """Remove items from the slot (customer purchase). Returns quantity actually sold."""
        sold = min(quantity, self.quantity)
        self.quantity -= sold
        if self.quantity == 0:
            self.product_id = None
            self.product_name = None
            self.price = 0.0
        return sold


@dataclass
class VendingMachine:
    slots: list[list[VendingSlot]] = field(default_factory=list)

    @staticmethod
    def create_default() -> VendingMachine:
        """4 rows x 3 cols. Rows 0-1 are small, rows 2-3 are large."""
        slots = []
        for row in range(4):
            size: ProductSize = "small" if row < 2 else "large"
            row_slots = [VendingSlot(row=row, col=col, size=size) for col in range(3)]
            slots.append(row_slots)
        return VendingMachine(slots=slots)

    def flat_slots(self) -> list[VendingSlot]:
        return [slot for row in self.slots for slot in row]

    def total_items(self) -> int:
        return sum(s.quantity for s in self.flat_slots())

    def unique_products(self) -> set[str]:
        return {s.product_id for s in self.flat_slots() if s.product_id}


@dataclass
class StorageItem:
    product_id: str
    name: str
    quantity: int
    cost_per_unit: float
    size: ProductSize


@dataclass
class OrderLineItem:
    product_id: str
    name: str
    quantity: int
    cost_per_unit: float
    size: ProductSize


@dataclass
class PendingDelivery:
    order_id: str
    supplier_id: str
    items: list[OrderLineItem]
    total_cost: float
    ordered_day: int
    delivery_day: int


@dataclass
class SupplierProduct:
    product_id: str
    name: str
    wholesale_price: float
    size: ProductSize


@dataclass
class SupplierInfo:
    id: str
    name: str
    email: str
    specialty: str
    catalog: list[SupplierProduct]
    delivery_days: int = 3
    min_order_amount: float = 0.0


@dataclass
class ProductEconomics:
    product_id: str
    name: str
    base_price: float
    base_daily_sales: float
    price_elasticity: float


@dataclass
class DailySalesRecord:
    day: int
    date: datetime
    items_sold: dict[str, int] = field(default_factory=dict)
    revenue: float = 0.0
    fee_charged: float = 0.0


@dataclass
class GameState:
    machine: VendingMachine = field(default_factory=VendingMachine.create_default)
    storage: list[StorageItem] = field(default_factory=list)
    money_balance: float = 500.0
    uncollected_cash: float = 0.0
    daily_fee: float = 2.0
    pending_deliveries: list[PendingDelivery] = field(default_factory=list)
    suppliers: list[SupplierInfo] = field(default_factory=list)
    product_economics: dict[str, ProductEconomics] = field(default_factory=dict)
    sales_history: list[DailySalesRecord] = field(default_factory=list)
    total_units_sold: int = 0
    consecutive_unpaid_days: int = 0
    current_day: int = 0

    @property
    def net_worth(self) -> float:
        storage_value = sum(i.quantity * i.cost_per_unit for i in self.storage)
        machine_value = 0.0
        for slot in self.machine.flat_slots():
            if slot.product_id and slot.quantity > 0:
                si = self._find_storage_cost(slot.product_id)
                machine_value += slot.quantity * si
        return self.money_balance + self.uncollected_cash + storage_value + machine_value

    def _find_storage_cost(self, product_id: str) -> float:
        for item in self.storage:
            if item.product_id == product_id:
                return item.cost_per_unit
        return 0.0

    def get_storage_item(self, product_id: str) -> StorageItem | None:
        for item in self.storage:
            if item.product_id == product_id:
                return item
        return None

    def add_to_storage(
        self, product_id: str, name: str, quantity: int, cost: float, size: ProductSize
    ):
        existing = self.get_storage_item(product_id)
        if existing:
            total_qty = existing.quantity + quantity
            existing.cost_per_unit = (
                existing.cost_per_unit * existing.quantity + cost * quantity
            ) / total_qty
            existing.quantity = total_qty
        else:
            self.storage.append(
                StorageItem(
                    product_id=product_id,
                    name=name,
                    quantity=quantity,
                    cost_per_unit=cost,
                    size=size,
                )
            )
