import unittest
from moa_v3 import *
class Tests(unittest.TestCase):
    def test_sequential_and_lift(self):
        d=DNF((10,10),4)
        seq=gamma_sequential(d)
        self.assertEqual(seq.address((3,2)),32)
        m=[MachineArray('CPU',tuple(range(10)),400,60),MachineArray('GPU',tuple(range(10)),400,80)]
        lifted=lift_with_explicit_chunks(seq,m,[40,60])
        self.assertTrue(lifted.verify())
        self.assertEqual(sum(p.stop-p.start for p in lifted.partitions),d.tau)
    def test_rejections(self):
        d=DNF((100,),8); seq=gamma_sequential(d)
        m=[MachineArray('CPU',tuple(range(10)),400,100)]
        with self.assertRaises(ValueError): lift_with_explicit_chunks(seq,m,[99])
        with self.assertRaises(ValueError): lift_with_explicit_chunks(seq,m,[100])
    def test_scalar_and_empty(self):
        self.assertEqual(gamma_sequential(DNF((),8)).address(()),0)
        self.assertEqual(DNF((0,3),8).tau,0)
if __name__=='__main__': unittest.main()
