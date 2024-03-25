import os
import sys
import unittest
import pickle
import numpy as np
import cantera as ct
import reaction_pathways.element_transfert as transfert

CT_MECHS = os.path.join(os.path.dirname(__file__), "test_assets", "mechs")
REF_FILES = os.path.join(os.path.dirname(__file__), "test_assets", "transfert")


class TestUtilities(unittest.TestCase):

    def test_transfert_matrix_H_SanDiego(self):
        # Cantera mech
        mech = os.path.join(CT_MECHS, "NH3_SanDiego.yaml")
        gas = ct.Solution(mech)
        # Reference matrix
        pfile = open(os.path.join(REF_FILES, "tmat_sdiego_H.pkl"), "rb")
        tmat_ref = pickle.load(pfile)
        # Compute the matrix
        tmat_test = transfert.e_transfer_matrix(gas, 'H')
        # Same len
        self.assertEqual(len(tmat_test), len(tmat_ref))
        # Same combs
        for cref, ctest in zip(tmat_ref, tmat_test):
            self.assertListEqual(list(cref), list(ctest))

    def test_transfert_matrix_H_h2o2(self):
        # Cantera mech
        mech = os.path.join(CT_MECHS, "h2o2.yaml")
        gas = ct.Solution(mech)
        # Reference matrix
        pfile = open(os.path.join(REF_FILES, "tmat_h2o2_H.pkl"), "rb")
        tmat_ref = pickle.load(pfile)
        # Compute the matrix
        tmat_test = transfert.e_transfer_matrix(gas, 'H')
        # Same len
        self.assertEqual(len(tmat_test), len(tmat_ref))
        # Same combs
        for cref, ctest in zip(tmat_ref, tmat_test):
            self.assertListEqual(list(cref), list(ctest))
            
    def test_transfert_matrix_O_SanDiego(self):
        # Cantera mech
        mech = os.path.join(CT_MECHS, "NH3_SanDiego.yaml")
        gas = ct.Solution(mech)
        # Reference matrix
        pfile = open(os.path.join(REF_FILES, "tmat_sdiego_O.pkl"), "rb")
        tmat_ref = pickle.load(pfile)
        # Compute the matrix
        tmat_test = transfert.e_transfer_matrix(gas, 'O')
        # Same len
        self.assertEqual(len(tmat_test), len(tmat_ref))
        # Same combs
        for cref, ctest in zip(tmat_ref, tmat_test):
            self.assertListEqual(list(cref), list(ctest))

    def test_transfert_matrix_N_SanDiego(self):
        # Cantera mech
        mech = os.path.join(CT_MECHS, "NH3_SanDiego.yaml")
        gas = ct.Solution(mech)
        # Reference matrix
        pfile = open(os.path.join(REF_FILES, "tmat_sdiego_N.pkl"), "rb")
        tmat_ref = pickle.load(pfile)
        # Compute the matrix
        tmat_test = transfert.e_transfer_matrix(gas, 'N')
        # Same len
        self.assertEqual(len(tmat_test), len(tmat_ref))
        # Same combs
        for cref, ctest in zip(tmat_ref, tmat_test):
            self.assertListEqual(list(cref), list(ctest))

    def test_transfert_matrix_N_Jiang(self):
        # Cantera mech
        mech = os.path.join(CT_MECHS, "NH3_Jiang.yaml")
        gas = ct.Solution(mech)
        # Reference matrix
        pfile = open(os.path.join(REF_FILES, "tmat_jiang_N.pkl"), "rb")
        tmat_ref = pickle.load(pfile)
        # Compute the matrix
        tmat_test = transfert.e_transfer_matrix(gas, 'N')
        # Same len
        self.assertEqual(len(tmat_test), len(tmat_ref))
        # Same combs
        for cref, ctest in zip(tmat_ref, tmat_test):
            self.assertListEqual(list(cref), list(ctest))

    def test_transfert_matrix_nofail_on_C(self):
        mech = os.path.join(CT_MECHS, "gri30.yaml")
        gas = ct.Solution(mech)

        try:
            _ = transfert.e_transfer_matrix(gas, 'C')
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        
