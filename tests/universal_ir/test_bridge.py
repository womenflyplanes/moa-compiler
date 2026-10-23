import unittest
from moa_universal_ir_v2 import Array,Psi,psi_to_dnf
from moa_v3 import MachineArray
from moa_partition_planner import MeasuredResource
from moa_v2_v3_bridge import plan_selection

def resource(name, seconds, cap=100, dtype_bytes=4):
    fields=(4,32,49152,seconds,cap,cap*dtype_bytes,100,'scalar',('explicit',),0)
    return MeasuredResource(MachineArray(name,fields,cap*dtype_bytes,cap),100,seconds)

class BridgeTests(unittest.TestCase):
    def test_cpu_gpu_selection(self):
        dnf=psi_to_dnf(Psi(Array('X',(2,100)),(1,)))
        before=repr(dnf)
        r=plan_selection(dnf,4,(resource('CPU',2),resource('GPU',1)))
        self.assertEqual(r.chunks,(33,67))
        self.assertEqual(r.source_offsets,tuple(range(100,200)))
        self.assertEqual(r.resource_offsets()[0][1],tuple(range(100,133)))
        self.assertEqual(repr(dnf),before)
    def test_cpu_only(self):
        dnf=psi_to_dnf(Psi(Array('X',(7,)),()))
        r=plan_selection(dnf,4,(resource('CPU',1),))
        self.assertEqual(r.chunks,(7,))
        self.assertTrue(r.verify())
    def test_gpu_only(self):
        dnf=psi_to_dnf(Psi(Array('X',(3,4)),(2,)))
        r=plan_selection(dnf,4,(resource('GPU',1),))
        self.assertEqual(r.source_offsets,(8,9,10,11))
    def test_scalar(self):
        dnf=psi_to_dnf(Psi(Array('S',()),()))
        r=plan_selection(dnf,4,(resource('CPU',1),))
        self.assertEqual(r.chunks,(1,))
    def test_empty(self):
        dnf=psi_to_dnf(Psi(Array('X',(0,)),()))
        r=plan_selection(dnf,4,(resource('CPU',1),resource('GPU',1)))
        self.assertEqual(r.chunks,(0,0))
    def test_insufficient_capacity(self):
        dnf=psi_to_dnf(Psi(Array('X',(20,)),()))
        with self.assertRaises(ValueError):
            plan_selection(dnf,4,(resource('CPU',1,4),resource('GPU',1,4)))

if __name__=='__main__': unittest.main()
