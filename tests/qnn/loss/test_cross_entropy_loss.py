import numpy as np
import pytest

from squlearn.qnn.loss import CrossEntropyLoss


class TestCrossEntropyLoss:
    def test_value_requires_ground_truth(self):
        loss = CrossEntropyLoss()
        with pytest.raises(AttributeError):
            loss.value({"f": np.array([0.1, 0.9])})

    def test_value_binary_simple(self):
        loss = CrossEntropyLoss()
        f = np.array([0.8, 0.2])
        y = np.array([1, 0])
        out = loss.value({"f": f}, ground_truth=y)
        # expected: probabilities stacked -> [[0.8,0.2],[0.2,0.8]]
        # both samples have log(0.8) for true class -> mean = log(0.8)
        expected = -np.log(0.8)
        assert np.allclose(out, expected)

    def test_value_multioutput_simple(self):
        loss = CrossEntropyLoss()
        f = np.array([[0.7, 0.3], [0.2, 0.8]])
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = loss.value({"f": f}, ground_truth=y)
        # expected: -mean( sum(y * log(f), axis=1) )
        expected = -np.mean(np.sum(y * np.log(f), axis=1))
        assert np.allclose(out, expected)

    def test_value_with_weights(self):
        loss = CrossEntropyLoss()
        f = np.array([0.8, 0.2])
        y = np.array([1, 0])
        weights = np.array([2.0, 1.0])
        out = loss.value({"f": f}, ground_truth=y, weights=weights)
        # manual compute: stacked probs [[0.8,0.2],[0.2,0.8]]
        sums = np.array([2.0 * np.log(0.8), 1.0 * np.log(0.8)])
        expected = -np.mean(sums)
        assert np.allclose(out, expected)

    def test_value_eps_clipping(self):
        eps = 1e-3
        loss = CrossEntropyLoss(eps=eps)
        f = np.array([0.0, 1.0])
        y = np.array([0, 1])
        out = loss.value({"f": f}, ground_truth=y)
        # after clipping: p -> [eps, 1-eps], stacked -> sample sums -> log(1-eps)
        expected = -np.log(1.0 - eps)
        assert np.allclose(out, expected)

    def test_gradient_requires_ground_truth(self):
        loss = CrossEntropyLoss()
        with pytest.raises(AttributeError):
            loss.gradient({"f": np.array([0.5])})
