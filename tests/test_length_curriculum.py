import unittest

from src.batching.length_curriculum import (
    build_length_order,
    compute_input_lengths,
)


class WhitespaceTokenizer:
    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class LengthCurriculumTests(unittest.TestCase):
    def test_compute_input_lengths_uses_formatted_input_by_default(self):
        dataset = [
            {"instruction": "short", "input_text": "one two three"},
            {"instruction": "two words", "input_text": "one two"},
        ]

        self.assertEqual(
            compute_input_lengths(dataset, WhitespaceTokenizer()),
            [3, 2],
        )

    def test_compute_input_lengths_can_use_instruction_only(self):
        dataset = [
            {"instruction": "short", "input_text": "one two three"},
            {"instruction": "two words", "input_text": "one two"},
        ]

        self.assertEqual(
            compute_input_lengths(
                dataset,
                WhitespaceTokenizer(),
                instruction_only=True,
            ),
            [1, 2],
        )

    def test_easy_to_hard_cycles(self):
        self.assertEqual(
            build_length_order(
                [30, 10, 20],
                "easy_to_hard",
                2,
                2,
            ),
            [1, 2, 0, 1],
        )

    def test_hard_to_easy_cycles(self):
        self.assertEqual(
            build_length_order(
                [30, 10, 20],
                "hard_to_easy",
                2,
                2,
            ),
            [0, 2, 1, 0],
        )

    def test_hard_to_easy_reverses_equal_length_examples(self):
        self.assertEqual(
            build_length_order(
                [10, 10, 20],
                "hard_to_easy",
                1,
                3,
            ),
            [2, 1, 0],
        )

    def test_exact_length(self):
        order = build_length_order(
            [3, 1, 2, 4],
            "easy_to_hard",
            5,
            3,
        )

        self.assertEqual(len(order), 15)

    def test_zero_batches_returns_empty_order(self):
        self.assertEqual(
            build_length_order([], "easy_to_hard", 0, 2),
            [],
        )

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            build_length_order([1, 2, 3], "random", 1, 2)

    def test_empty_lengths(self):
        with self.assertRaises(ValueError):
            build_length_order([], "easy_to_hard", 1, 2)

    def test_negative_batches(self):
        with self.assertRaises(ValueError):
            build_length_order([1], "easy_to_hard", -1, 2)

    def test_non_positive_batch_size(self):
        with self.assertRaises(ValueError):
            build_length_order([1], "easy_to_hard", 1, 0)


if __name__ == "__main__":
    unittest.main()
