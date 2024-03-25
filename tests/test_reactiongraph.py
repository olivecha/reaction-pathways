import os
import unittest
import pickle
import cantera as ct
from reaction_pathways import ReactionGraph

CT_MECHS = os.path.join(os.path.dirname(__file__), 
                        "test_assets", 
                        "mechs")
GRAPH_FILES = os.path.join(os.path.dirname(__file__), 
                           "test_assets", 
                           "reactiongraph")

class TestReactionGraph(unittest.TestCase):
    # Create 1D flame data
    gas = ct.Solution(os.path.join(CT_MECHS, "h2o2.yaml"))
    gas.TP = 300, ct.one_atm
    gas.set_equivalence_ratio(1.0, {'H2':1.0}, {'O2':0.21, 'N2':0.79})
    flame = ct.FreeFlame(gas, width=0.1)
    flame.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
    flame.solve(loglevel=0, auto=True)

    def test_reaction_graph_0D(self):
        graph = ReactionGraph(rdata=self.gas,
                              element='H')

    def test_reaction_graph_1D(self):
        graph = ReactionGraph(rdata=self.flame,
                              element='H')

    def test_formatter_d2(self):
        gas = pickle.load(open(os.path.join(GRAPH_FILES, 'user_gas.pkl'), 'rb'))
        rdata = pickle.load(open(os.path.join(GRAPH_FILES, 'user_rdata.pkl'), 'rb'))
        graph = ReactionGraph(rdata=rdata, 
                              sol=gas, 
                              element='N') 
