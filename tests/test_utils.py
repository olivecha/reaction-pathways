import os
import sys
import numpy as np
import unittest
from reaction_pathways.utils import format_value


class TestUtilities(unittest.TestCase):

    def test_formatter_d0(self):
        fstr = format_value(0.5, 1.0)
        self.assertEqual(fstr, "50%")

    def test_formatter_d1(self):
        fstr = format_value(0.095, 1.0)
        self.assertEqual(fstr, "9.5%")

    def test_formatter_d2(self):
        fstr = format_value(0.0051, 1.0)
        self.assertEqual(fstr, "0.51%")

    def test_formatter_e(self):
        fstr = format_value(0.00012, 1.0)
        self.assertEqual(fstr, "1.2e-02%")

    def test_formatter_error(self):
        with self.assertRaises(ValueError):
            format_value(2.0, 1.0)
