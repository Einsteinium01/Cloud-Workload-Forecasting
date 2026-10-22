import numpy as np

from cwfc.features import build_supervised, lag_feature_vector, make_future_features


def test_features_do_not_depend_on_future_values():
    values = np.arange(30, dtype=float) + 10.0
    index = 20
    before = lag_feature_vector(values, index)
    mutated = values.copy()
    mutated[index + 1 :] = -999.0
    after = lag_feature_vector(mutated, index)
    np.testing.assert_allclose(before, after)


def test_supervised_rows_use_only_past_values():
    values = np.arange(40, dtype=float) + 5.0
    features, targets, min_history = build_supervised(
        values, lags=(1, 2), windows=(3,)
    )
    assert features.shape == (37, 4)
    assert targets.shape == (37,)
    assert min_history == 3
    # The first supervised row predicts values[3] using values[0:3].
    np.testing.assert_allclose(
        features[0], [values[2], values[1], np.mean(values[0:3]), np.std(values[0:3])]
    )
    assert targets[0] == values[3]


def test_future_feature_vector_is_the_vector_at_the_forecast_origin():
    values = np.sin(np.linspace(0, 3, 40)) + 10.0
    future = make_future_features(values, lags=(1, 3), windows=(4,))
    expected = lag_feature_vector(
        values, len(values) - 1, lags=(1, 3), windows=(4,)
    )
    np.testing.assert_allclose(future, expected)
