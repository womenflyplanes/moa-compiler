import unittest
from moa_v3 import DNF, MachineArray, gamma_sequential
from moa_partition_planner import MeasuredResource, plan_lift

def resource(name, seconds, cap=100, minchunk=1, extra=()):
    fields=(4,32,49152,seconds,cap,cap*4,100,'scalar',('explicit',),0)+tuple(extra)
    return MeasuredResource(MachineArray(name,fields,cap*4,cap),100,seconds,minchunk)

class PlannerTests(unittest.TestCase):
    def test_heterogeneous(self):
        seq=gamma_sequential(DNF((100,),4))
        plan=plan_lift(seq,(resource('cpu',2),resource('gpu',1)))
        self.assertEqual(sum(plan.chunks),100)
        self.assertEqual(plan.chunks,(33,67))
        self.assertTrue(plan.lifted.verify())
    def test_capacity_and_idle(self):
        plan=plan_lift(gamma_sequential(DNF((20,),4)),(resource('a',1,10),resource('b',1,20)))
        self.assertEqual(sum(plan.chunks),20)
        self.assertLessEqual(plan.chunks[0],10)
    def test_min_viable_and_infeasible(self):
        seq=gamma_sequential(DNF((7,),4))
        with self.assertRaises(ValueError):
            plan_lift(seq,(resource('a',1,4,5),resource('b',1,4,5)))
    def test_extensible_and_scalar(self):
        r=resource('cpu',1,10,extra=('mpi_future',))
        p=plan_lift(gamma_sequential(DNF((),4)),(r,))
        self.assertEqual(p.chunks,(1,))
    def test_reservation(self):
        r=resource('cpu',1,100)
        r=MeasuredResource(r.machine,100,1,reserved_bytes=390)
        with self.assertRaises(ValueError):
            plan_lift(gamma_sequential(DNF((20,),4)),(r,))

if __name__=='__main__': unittest.main()
