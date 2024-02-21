import os
import sys
sys.path.append('..')
import numpy as np
import unittest
from rpa_module import format_value

#import rpa_.......

# from rpa_module import ReactionPathway

class TestResidual(unittest.TestCase):

    def test_conservation(self):
        RPA = np.eye(10)
        for line in RPA:
            self.assertAlmostEqual(np.sum(line), 1.)

    def test_create_rpa(self):
        #gas = Solution(mech.yaml)
        
        self.assertEqual(1, 1)

    def test_formatter(self):
        format_value(0.1)
