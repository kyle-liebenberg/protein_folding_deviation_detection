import pytest

from src.schedules import learning_rate

PEAK, TOTAL = 0.1, 1000


def test_constant_schedule_never_changes():
    assert {learning_rate(s, PEAK, TOTAL) for s in [0, 500, 999]} == {PEAK}


def test_linear_warmup_rises_to_peak_then_holds():
    lr = lambda s: learning_rate(s, PEAK, TOTAL, warmup_steps=100)
    assert lr(0) == pytest.approx(PEAK / 100)     # first step: small but not zero
    assert lr(49) == pytest.approx(PEAK / 2)      # halfway through warmup
    assert lr(99) == pytest.approx(PEAK)          # last warmup step reaches the peak
    assert lr(100) == lr(999) == PEAK             # constant afterwards


def test_cosine_decays_from_peak_to_zero():
    lr = lambda s: learning_rate(s, PEAK, TOTAL, decay="cosine")
    assert lr(0) == pytest.approx(PEAK)
    assert lr(500) == pytest.approx(PEAK / 2)     # halfway: cos(pi/2) = 0 -> half the peak
    assert lr(1000) == pytest.approx(0.0, abs=1e-12)


def test_warmup_then_cosine():
    lr = lambda s: learning_rate(s, PEAK, TOTAL, warmup_steps=100, decay="cosine")
    assert lr(99) == pytest.approx(PEAK)
    assert lr(100) == pytest.approx(PEAK)          # decay starts after warmup
    assert lr(550) == pytest.approx(PEAK / 2)      # halfway through the 900 decay steps
