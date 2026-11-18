"""Limit order book with price-time priority (CLAUDE.md Section 6).

Pure data structure: no randomness, no tax logic. Prices are integer ticks.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

BUY = 1
SELL = -1


@dataclass(frozen=True, slots=True)
class Fill:
    t: int
    price_ticks: int
    qty: int
    buyer_id: int
    seller_id: int
    aggressor_side: int  # +1 if incoming order was a buy, -1 if a sell


@dataclass(slots=True)
class Order:
    order_id: int
    agent_id: int
    side: int
    price_ticks: int
    qty_remaining: int
    t_submitted: int
    expiry_t: int
    seq: int


class OrderBook:
    """Continuous double auction with price-time priority and lazy deletion.

    Parameters
    ----------
    tick : float
        Minimum price increment in currency (kept for reference; the book works in ticks).
    """

    def __init__(self, tick: float) -> None:
        if tick <= 0:
            raise ValueError("tick must be > 0")
        self.tick = tick
        self.orders: dict[int, Order] = {}
        self._bids: list[tuple[int, int, int]] = []  # (-price, seq, order_id)
        self._asks: list[tuple[int, int, int]] = []  # (price, seq, order_id)
        self._expiry: deque[tuple[int, int]] = deque()  # (expiry_t, order_id)
        self._next_id = 0
        self._seq = 0
        self._last_expiry = -(2**62)

    # ------------------------------------------------------------------ queries
    def __len__(self) -> int:
        return len(self.orders)

    def is_live(self, order_id: int) -> bool:
        """True if ``order_id`` is resting in the book."""
        return order_id in self.orders

    def _clean_top(self, heap: list[tuple[int, int, int]]) -> tuple[int, int, int] | None:
        orders = self.orders
        while heap and heap[0][2] not in orders:
            heapq.heappop(heap)
        return heap[0] if heap else None

    def best_bid(self) -> int | None:
        top = self._clean_top(self._bids)
        return -top[0] if top is not None else None

    def best_ask(self) -> int | None:
        top = self._clean_top(self._asks)
        return top[0] if top is not None else None

    def depth_within(self, mid_ticks: float, pct: float) -> tuple[int, int]:
        """Bid qty with price >= mid*(1-pct); ask qty with price <= mid*(1+pct).

        Boundary prices count as inside.
        """
        lo = mid_ticks * (1.0 - pct)
        hi = mid_ticks * (1.0 + pct)
        bid_q = 0
        ask_q = 0
        for o in self.orders.values():
            if o.side == BUY:
                if o.price_ticks >= lo:
                    bid_q += o.qty_remaining
            elif o.price_ticks <= hi:
                ask_q += o.qty_remaining
        return bid_q, ask_q

    def levels(self, n: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Aggregated ``(price_ticks, qty)`` levels: the ``n`` best bids and ``n`` best asks.

        Bids are sorted high to low and asks low to high (best first). Read-only.
        """
        bid: dict[int, int] = {}
        ask: dict[int, int] = {}
        for o in self.orders.values():
            side = bid if o.side == BUY else ask
            side[o.price_ticks] = side.get(o.price_ticks, 0) + o.qty_remaining
        bids = sorted(bid.items(), key=lambda x: -x[0])[:n]
        asks = sorted(ask.items())[:n]
        return bids, asks

    # ------------------------------------------------------------------ actions
    def submit(
        self, agent_id: int, side: int, price_ticks: int, qty: int, t: int, expiry_t: int
    ) -> tuple[int | None, list[Fill]]:
        """Match, then rest any remainder. Returns (resting order_id or None, fills)."""
        if side not in (BUY, SELL):
            raise ValueError(f"side must be +1 or -1, got {side}")
        if qty <= 0 or price_ticks < 1:
            raise ValueError("qty must be > 0 and price_ticks >= 1")
        fills: list[Fill] = []
        orders = self.orders
        opp = self._asks if side == BUY else self._bids
        while qty > 0:
            top = self._clean_top(opp)
            if top is None:
                break
            rest_price = top[0] if side == BUY else -top[0]
            if (side == BUY and rest_price > price_ticks) or (
                side == SELL and rest_price < price_ticks
            ):
                break
            resting = orders[top[2]]
            q = min(qty, resting.qty_remaining)
            if side == BUY:
                buyer, seller = agent_id, resting.agent_id
            else:
                buyer, seller = resting.agent_id, agent_id
            assert buyer != seller, "self-trade"
            fills.append(Fill(t, rest_price, q, buyer, seller, side))
            qty -= q
            resting.qty_remaining -= q
            if resting.qty_remaining == 0:
                heapq.heappop(opp)
                del orders[resting.order_id]
        if qty == 0:
            return None, fills
        return self._rest(agent_id, side, price_ticks, qty, t, expiry_t), fills

    def _rest(
        self, agent_id: int, side: int, price_ticks: int, qty: int, t: int, expiry_t: int
    ) -> int:
        oid = self._next_id
        self._next_id += 1
        seq = self._seq
        self._seq += 1
        self.orders[oid] = Order(oid, agent_id, side, price_ticks, qty, t, expiry_t, seq)
        if side == BUY:
            heapq.heappush(self._bids, (-price_ticks, seq, oid))
        else:
            heapq.heappush(self._asks, (price_ticks, seq, oid))
        if expiry_t < self._last_expiry:
            # The deque relies on non-decreasing expiry times (constant order_life).
            raise ValueError("expiry times must be non-decreasing in submission order")
        self._last_expiry = expiry_t
        self._expiry.append((expiry_t, oid))
        self._maybe_compact()
        return oid

    def cancel(self, order_id: int) -> bool:
        """Remove a live order. Returns False if it is not live."""
        if self.orders.pop(order_id, None) is None:
            return False
        self._maybe_compact()
        return True

    def expire_orders(self, t: int) -> list[Order]:
        """Remove and return all live orders with ``expiry_t <= t``."""
        removed: list[Order] = []
        q = self._expiry
        orders = self.orders
        while q and q[0][0] <= t:
            _, oid = q.popleft()
            o = orders.pop(oid, None)
            if o is not None:
                removed.append(o)
        if removed:
            self._maybe_compact()
        return removed

    def expire(self, t: int) -> int:
        """Remove all orders with ``expiry_t <= t``; returns the number removed."""
        return len(self.expire_orders(t))

    def _maybe_compact(self) -> None:
        """Rebuild heaps when stale (lazily deleted) entries dominate."""
        live = len(self.orders)
        if len(self._bids) + len(self._asks) > 4 * live + 256:
            orders = self.orders
            self._bids = [e for e in self._bids if e[2] in orders]
            self._asks = [e for e in self._asks if e[2] in orders]
            heapq.heapify(self._bids)
            heapq.heapify(self._asks)
