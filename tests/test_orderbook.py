from hypothesis import given, settings
from hypothesis import strategies as st

from marketabm.orderbook import BUY, SELL, OrderBook


def make() -> OrderBook:
    return OrderBook(tick=0.01)


def test_empty_book() -> None:
    b = make()
    assert b.best_bid() is None and b.best_ask() is None and len(b) == 0
    oid, fills = b.submit(1, BUY, 100, 1, 0, 10)
    assert oid is not None and fills == [] and b.best_bid() == 100
    oid, fills = b.submit(2, SELL, 105, 1, 0, 10)
    assert fills == [] and b.best_ask() == 105


def test_buy_below_ask_rests() -> None:
    b = make()
    b.submit(1, SELL, 105, 1, 0, 10)
    oid, fills = b.submit(2, BUY, 104, 1, 0, 10)
    assert fills == [] and b.best_bid() == 104 and b.is_live(oid)


def test_marketable_buy_fills_at_resting_price() -> None:
    b = make()
    b.submit(1, SELL, 105, 1, 0, 10)
    oid, fills = b.submit(2, BUY, 110, 1, 1, 10)
    assert oid is None
    assert len(fills) == 1
    f = fills[0]
    assert (f.price_ticks, f.buyer_id, f.seller_id, f.aggressor_side) == (105, 2, 1, BUY)
    assert b.best_ask() is None


def test_marketable_sell_fills_at_bid() -> None:
    b = make()
    b.submit(1, BUY, 100, 1, 0, 10)
    _, fills = b.submit(2, SELL, 90, 1, 0, 10)
    assert fills[0].price_ticks == 100 and fills[0].aggressor_side == SELL


def test_price_priority() -> None:
    b = make()
    b.submit(1, SELL, 107, 1, 0, 10)
    b.submit(2, SELL, 105, 1, 0, 10)
    _, fills = b.submit(3, BUY, 110, 1, 0, 10)
    assert fills[0].seller_id == 2 and fills[0].price_ticks == 105


def test_time_priority() -> None:
    b = make()
    b.submit(1, SELL, 105, 1, 0, 10)
    b.submit(2, SELL, 105, 1, 0, 10)
    _, fills = b.submit(3, BUY, 105, 1, 0, 10)
    assert fills[0].seller_id == 1


def test_crossing_several_levels() -> None:
    b = make()
    b.submit(1, SELL, 101, 1, 0, 10)
    b.submit(2, SELL, 102, 2, 0, 10)
    b.submit(3, SELL, 104, 1, 0, 10)
    oid, fills = b.submit(9, BUY, 103, 5, 0, 10)
    assert [(f.price_ticks, f.qty, f.seller_id) for f in fills] == [(101, 1, 1), (102, 2, 2)]
    assert oid is not None and b.orders[oid].qty_remaining == 2
    assert b.best_bid() == 103 and b.best_ask() == 104


def test_partial_fill_of_resting_order() -> None:
    b = make()
    rid, _ = b.submit(1, SELL, 100, 3, 0, 10)
    oid, fills = b.submit(2, BUY, 100, 1, 0, 10)
    assert oid is None and fills[0].qty == 1
    assert b.orders[rid].qty_remaining == 2


def test_cancel() -> None:
    b = make()
    oid, _ = b.submit(1, SELL, 100, 1, 0, 10)
    assert b.cancel(oid) is True
    assert b.cancel(oid) is False
    _, fills = b.submit(2, BUY, 200, 1, 0, 10)
    assert fills == []
    assert b.best_ask() is None


def test_expire() -> None:
    b = make()
    o1, _ = b.submit(1, BUY, 100, 1, 0, 5)
    o2, _ = b.submit(2, BUY, 99, 1, 1, 6)
    o3, _ = b.submit(3, SELL, 110, 1, 2, 7)
    assert b.expire(4) == 0
    assert b.expire(6) == 2
    assert not b.is_live(o1) and not b.is_live(o2) and b.is_live(o3)
    assert b.best_bid() is None


def test_depth_within_boundaries() -> None:
    b = make()
    b.submit(1, BUY, 99, 2, 0, 10)  # exactly mid*(1-0.01)
    b.submit(2, BUY, 98, 5, 0, 10)  # outside
    b.submit(3, SELL, 101, 3, 0, 10)  # exactly mid*(1+0.01)
    b.submit(4, SELL, 102, 7, 0, 10)  # outside
    assert b.depth_within(100.0, 0.01) == (2, 3)


# ---------------------------------------------------------------- property tests
order_st = st.tuples(
    st.sampled_from(["submit", "submit", "submit", "cancel", "expire"]),
    st.integers(0, 9),  # agent
    st.sampled_from([BUY, SELL]),
    st.integers(90, 110),  # price
    st.integers(1, 3),  # qty
)


@settings(max_examples=200, deadline=None)
@given(st.lists(order_st, min_size=1, max_size=120))
def test_book_invariants(ops: list) -> None:
    b = make()
    limits: dict[int, int] = {}
    submitted = filled = cancelled = expired = 0
    agent_live: dict[int, int] = {}
    t = 0
    for op, agent, side, price, qty in ops:
        t += 1
        if op == "expire":
            for o in b.expire_orders(t - 15):
                expired += o.qty_remaining
            continue
        old = agent_live.pop(agent, None)
        if op == "cancel":
            if old is not None and b.is_live(old):
                cancelled += b.orders[old].qty_remaining
                assert b.cancel(old)
            continue
        if old is not None and b.is_live(old):
            cancelled += b.orders[old].qty_remaining
            b.cancel(old)
        submitted += qty
        incoming_limit = price
        oid, fills = b.submit(agent, side, price, qty, t, t + 20)
        for f in fills:
            filled += 2 * f.qty  # both sides' quantity leaves the system
            resting_agent = f.seller_id if side == BUY else f.buyer_id
            lo, hi = sorted((incoming_limit, limits[resting_agent]))
            assert lo <= f.price_ticks <= hi
        if oid is not None:
            agent_live[agent] = oid
            limits[agent] = price
        bb, ba = b.best_bid(), b.best_ask()
        if bb is not None and ba is not None:
            assert bb < ba
        resting = sum(o.qty_remaining for o in b.orders.values())
        assert submitted == filled + resting + cancelled + expired


def test_levels_aggregates_by_price() -> None:
    book = OrderBook(0.01)
    book.submit(0, 1, 100, 2, 0, 10)
    book.submit(1, 1, 100, 3, 0, 10)
    book.submit(2, 1, 99, 1, 0, 10)
    book.submit(3, -1, 102, 4, 0, 10)
    bids, asks = book.levels(5)
    assert bids == [(100, 5), (99, 1)]
    assert asks == [(102, 4)]
    assert book.levels(1)[0] == [(100, 5)]
