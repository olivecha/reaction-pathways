import os
import numpy as np
import unittest

# from rpa_module import ReactionPathway

class TestResidual(unittest.TestCase):

    def test_conservation(self):
        RPA = np.eye(10)
        for line in RPA:
            self.assertAlmostEqual(np.sum(line), 1.)
