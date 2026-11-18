import math

import numpy as np
import pytest

import marketabm.market as mk
from marketabm.config import ModelConfig, with_overrides
from marketabm.market import Ledger, run
from marketabm.orderbook import BUY, SELL, OrderBook


def test_zero_tax_no_revenue_no_tax_skips(small_stable: ModelConfig) -> None:
    r = run(small_stable, 7)
    assert r.tax_revenue_cum[-1] == 0.0
    assert r.skipped_tax_by_type.sum() == 0


def test_revenue_matches_fills_and_money_conserved(
    small_stable: ModelConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = with_overrides(small_stable, {"tau": 0.006})
    seen: list[float] = []
    orig = mk.Ledger.apply_fill

    def spy(self: Ledger, buyer: int, seller: int, price: float, qty: int) -> None:
        seen.append(price * qty)
        orig(self, buyer, seller, price, qty)

    monkeypatch.setattr(mk.Ledger, "apply_fill", spy)
    r = run(cfg, 7, debug=True)  # debug checks money/units conservation every step
    assert r.tax_revenue_cum[-1] == pytest.approx(math.fsum(0.006 * v for v in seen), abs=1e-9)
    assert r.skipped_tax_by_type.sum() > 0
    assert abs(r.cash.sum() + r.tax_revenue_cum[-1]) < 1e-6
    assert r.pos.sum() == 0


def test_only_seller_pays() -> None:
    tau = 0.006
    book = OrderBook(0.01)
    led = Ledger(2, tau)
    book.submit(0, SELL, 10000, 1, 0, 10)  # agent 0 rests a sell at 100.00
    _, fills = book.submit(1, BUY, 10050, 1, 1, 10)  # agent 1 buys, aggressor
    for f in fills:
        led.apply_fill(f.buyer_id, f.seller_id, f.price_ticks * 0.01, f.qty)
    assert led.cash[1] == pytest.approx(-100.0, abs=1e-12)
    assert led.cash[0] == pytest.approx(100.0 * (1 - tau), abs=1e-12)
    assert led.tax_revenue == pytest.approx(0.6, abs=1e-12)
    assert led.pos == [-1, 1]
    # Seller as aggressor pays too.
    book.submit(0, BUY, 9900, 1, 2, 10)
    _, fills = book.submit(1, SELL, 9800, 1, 3, 10)
    for f in fills:
        led.apply_fill(f.buyer_id, f.seller_id, f.price_ticks * 0.01, f.qty)
    assert led.cash[1] == pytest.approx(-100.0 + 99.0 * (1 - tau), abs=1e-12)
    assert led.cash[0] == pytest.approx(100.0 * (1 - tau) - 99.0, abs=1e-12)
    assert sum(led.cash) + led.tax_revenue == pytest.approx(0.0, abs=1e-12)
    assert sum(led.pos) == 0


def test_multi_unit_fill_tax() -> None:
    led = Ledger(2, 0.001)
    led.apply_fill(buyer=0, seller=1, price=50.0, qty=4)
    assert led.cash == pytest.approx([-200.0, 200.0 * 0.999])
    assert led.tax_revenue == pytest.approx(0.2)
    assert led.pos == [4, -4]


def test_skipped_agent_trades_without_tax(small_stable: ModelConfig) -> None:
    # Same seed: tax-induced skips at tau > 0 become trades at tau = 0.
    r0 = run(small_stable, 3)
    r1 = run(with_overrides(small_stable, {"tau": 0.01}), 3)
    assert r1.skipped_tax_by_type.sum() > 0
    assert r0.skipped_tax_by_type.sum() == 0
    assert np.array_equal(r0.active_by_type, r1.active_by_type)
    assert r1.volume.sum() < r0.volume.sum()
