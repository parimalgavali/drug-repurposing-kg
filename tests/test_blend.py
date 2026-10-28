import numpy as np
import polars as pl

from ml.blend import mix, pick_alpha, zscore

DRUGS = np.array([10, 11, 12, 13])


def test_zscore_rows_and_constant_rows():
    z = zscore(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z[0].std(), 1)
    assert (z[1] == 0).all()


def test_mix_endpoints_follow_each_model():
    a = lambda chunk: np.tile([4.0, 3.0, 2.0, 1.0], (len(chunk), 1))  # noqa: E731
    b = lambda chunk: np.tile([1.0, 2.0, 3.0, 400.0], (len(chunk), 1))  # noqa: E731
    chunk = np.array([1])
    assert mix(a, b, 1.0)(chunk).argmax() == 0
    assert mix(a, b, 0.0)(chunk).argmax() == 3


def test_pick_alpha_prefers_model_that_is_right_on_validation():
    right = lambda chunk: np.tile([0.0, 0.0, 9.0, 0.0], (len(chunk), 1))  # noqa: E731
    wrong = lambda chunk: np.tile([9.0, 0.0, 0.0, 0.0], (len(chunk), 1))  # noqa: E731
    valid = pl.DataFrame({"drug": [12, 12], "disease": [1, 2]})
    known = pl.DataFrame({"drug": [11], "disease": [1]})
    alpha, curve = pick_alpha(right, wrong, DRUGS, valid, known)
    assert curve[alpha] == 1.0
    assert alpha > 0.5
    alpha, _ = pick_alpha(wrong, right, DRUGS, valid, known)
    assert alpha < 0.5
