import os
import numpy as np
import unittest

# from rpa_module import ReactionPathway

class TestResidual(unittest.TestCase):

    def test_conservation(self):
        RPA = np.ones((3, 3))
        self.assertAlmostEqual(RPA[0, 0], 1.)
