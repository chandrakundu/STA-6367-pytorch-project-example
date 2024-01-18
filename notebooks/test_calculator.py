# File: notebooks/test_calculator.py

import unittest
from calculator import add


class TestCalculator(unittest.TestCase):
    def test_add_positive_numbers(self):
        result = add(2, 3)
        self.assertEqual(result, 5, "Adding 2 and 3 should equal 5")

    def test_add_negative_numbers(self):
        result = add(-2, -3)
        self.assertEqual(result, -5, "Adding -2 and -3 should equal -5")


if __name__ == "__main__":
    unittest.main()
    # python -m unittest test_calculator.py
