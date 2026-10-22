import numpy as np

from cwfc.models import (
    ENSEMBLE_P_MA12_AR6_S48,
    MOVING_AVERAGE_12,
    PERSISTENCE,
    SEASONAL_NAIVE_48,
    available_models,
    forecast_ar,
    forecast_ensemble,
    forecast_moving_average,
    forecast_persistence,
    forecast_seasonal_naive,
    forecast_seasonal_random_walk,
)


def test_persistence_is_not_capped_at_100():
    # Active VM counts can exceed 100; the model layer must not apply the CPU cap.
    history = np.array([9000.0, 9500.0, 10000.0])
    forecast = forecast_persistence(history, 24)
    assert np.all(forecast == 10000.0)


def test_moving_average_uses_the_requested_window():
    history = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    forecast = forecast_moving_average(history, 4, window=3)
    assert np.allclose(forecast, np.mean([3.0, 4.0, 100.0]))


def test_seasonal_naive_repeats_the_last_season():
    history = np.arange(1.0, 13.0)
    forecast = forecast_seasonal_naive(history, 6, season=4)
    np.testing.assert_allclose(forecast, [9.0, 10.0, 11.0, 12.0, 9.0, 10.0])


def test_seasonal_random_walk_is_anchored_at_the_last_value():
    history = np.arange(1.0, 13.0)
    forecast = forecast_seasonal_random_walk(history, 6, season=4, weight=0.5)
    assert forecast[0] == history[-1]
    assert np.all(np.isfinite(forecast))


def test_ar_forecast_is_finite_and_has_the_requested_shape():
    rng = np.random.default_rng(0)
    history = 10.0 + np.cumsum(rng.normal(0.0, 0.05, 80))
    forecast = forecast_ar(history, 24, order=6)
    assert forecast.shape == (24,)
    assert np.all(np.isfinite(forecast))


def test_ensemble_skips_components_without_enough_history():
    short_history = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
    forecast = forecast_ensemble(
        short_history,
        6,
        ((PERSISTENCE, 1.0), (SEASONAL_NAIVE_48, 1.0)),
    )
    assert forecast.shape == (6,)
    assert np.allclose(forecast, 10.4)


def test_available_models_respects_minimum_history():
    names = {spec.name for spec in available_models(10)}
    assert "persistence" in names
    assert "moving_average_12" not in names
    assert "seasonal_naive_48" not in names
    assert "ar_6" in names
    assert "ensemble_p_ma12_ar6_s48" in names


def test_ensemble_definition_contains_the_expected_components():
    assert ENSEMBLE_P_MA12_AR6_S48.name == "ensemble_p_ma12_ar6_s48"
    assert MOVING_AVERAGE_12.min_history == 12
