import unittest

import numpy as np

from src.batching.strategies import (
    build_batch_order,
    build_curriculum_order,
    build_grouped_order,
    build_random_order,
)


class NumpyInnerProductIndex:
    """
    Small in-memory stand-in for FAISS IndexFlatIP.

    It gives the strategy tests deterministic nearest-neighbor behavior
    without requiring a real embedding model or FAISS index build.
    """

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def search(self, query: np.ndarray, k: int):
        scores = query @ self.embeddings.T
        ids = np.argsort(-scores, axis=1)[:, :k]
        sorted_scores = np.take_along_axis(scores, ids, axis=1)
        return sorted_scores, ids


class TestBatchingStrategies(unittest.TestCase):
    def setUp(self):
        self.n_examples = 12
        self.batch_size = 3
        self.n_batches = 8
        self.top_k = 5
        self.seed = 13

        # Self-similarity is maximal, matching the behavior expected by
        # the grouped strategy when querying a normalized FAISS index.
        self.embeddings = np.eye(
            self.n_examples,
            dtype=np.float32,
        )

        self.index = NumpyInnerProductIndex(self.embeddings)

    def test_random_order_has_expected_length_and_valid_indices(self):
        order = build_random_order(
            n_examples=self.n_examples,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            seed=self.seed,
        )

        self.assertEqual(
            len(order),
            self.n_batches * self.batch_size,
        )

        self.assertTrue(
            all(0 <= idx < self.n_examples for idx in order)
        )

        # Random sampling is without replacement inside each physical batch.
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            self.assertEqual(len(batch), len(set(batch)))

    def test_random_order_is_deterministic_for_same_seed(self):
        first = build_random_order(
            self.n_examples,
            self.batch_size,
            self.n_batches,
            self.seed,
        )

        second = build_random_order(
            self.n_examples,
            self.batch_size,
            self.n_batches,
            self.seed,
        )

        self.assertEqual(first, second)

    def test_random_order_changes_with_seed(self):
        first = build_random_order(
            self.n_examples,
            self.batch_size,
            self.n_batches,
            self.seed,
        )

        second = build_random_order(
            self.n_examples,
            self.batch_size,
            self.n_batches,
            self.seed + 1,
        )

        self.assertNotEqual(first, second)

    def test_grouped_order_has_expected_length_and_valid_indices(self):
        order = build_grouped_order(
            embeddings=self.embeddings,
            index=self.index,
            n_examples=self.n_examples,
            batch_size=self.batch_size,
            top_k=self.top_k,
            n_batches=self.n_batches,
            seed=self.seed,
        )

        self.assertEqual(
            len(order),
            self.n_batches * self.batch_size,
        )

        self.assertTrue(
            all(0 <= idx < self.n_examples for idx in order)
        )

    def test_grouped_order_is_deterministic_for_same_seed(self):
        first = build_grouped_order(
            self.embeddings,
            self.index,
            self.n_examples,
            self.batch_size,
            self.top_k,
            self.n_batches,
            self.seed,
        )

        second = build_grouped_order(
            self.embeddings,
            self.index,
            self.n_examples,
            self.batch_size,
            self.top_k,
            self.n_batches,
            self.seed,
        )

        self.assertEqual(first, second)

    def test_grouped_to_random_matches_two_phase_construction(self):
        half = self.n_batches // 2

        expected = (
            build_grouped_order(
                self.embeddings,
                self.index,
                self.n_examples,
                self.batch_size,
                self.top_k,
                half,
                self.seed,
            )
            + build_random_order(
                self.n_examples,
                self.batch_size,
                self.n_batches - half,
                self.seed + 1000,
            )
        )

        actual = build_curriculum_order(
            strategy="grouped_to_random",
            embeddings=self.embeddings,
            index=self.index,
            n_examples=self.n_examples,
            batch_size=self.batch_size,
            top_k=self.top_k,
            n_batches=self.n_batches,
            seed=self.seed,
        )

        self.assertEqual(actual, expected)

    def test_random_to_grouped_matches_two_phase_construction(self):
        half = self.n_batches // 2

        expected = (
            build_random_order(
                self.n_examples,
                self.batch_size,
                half,
                self.seed,
            )
            + build_grouped_order(
                self.embeddings,
                self.index,
                self.n_examples,
                self.batch_size,
                self.top_k,
                self.n_batches - half,
                self.seed + 1000,
            )
        )

        actual = build_curriculum_order(
            strategy="random_to_grouped",
            embeddings=self.embeddings,
            index=self.index,
            n_examples=self.n_examples,
            batch_size=self.batch_size,
            top_k=self.top_k,
            n_batches=self.n_batches,
            seed=self.seed,
        )

        self.assertEqual(actual, expected)

    def test_dispatcher_supports_all_group1_strategies(self):
        strategies = [
            "random",
            "grouped",
            "grouped_to_random",
            "random_to_grouped",
        ]

        for strategy in strategies:
            with self.subTest(strategy=strategy):
                order = build_batch_order(
                    strategy=strategy,
                    embeddings=self.embeddings,
                    index=self.index,
                    n_examples=self.n_examples,
                    batch_size=self.batch_size,
                    top_k=self.top_k,
                    n_batches=self.n_batches,
                    seed=self.seed,
                )

                self.assertEqual(
                    len(order),
                    self.n_batches * self.batch_size,
                )

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            build_batch_order(
                strategy="unsupported",
                embeddings=self.embeddings,
                index=self.index,
                n_examples=self.n_examples,
                batch_size=self.batch_size,
                top_k=self.top_k,
                n_batches=self.n_batches,
                seed=self.seed,
            )


if __name__ == "__main__":
    unittest.main()