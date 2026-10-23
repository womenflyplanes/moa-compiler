import unittest
from moa_prediction_validation import validate

def r(shape,seconds,role,machine='CPU',workload='sobel',dtype='4'):
    return dict(machine=machine,workload=workload,shape=shape,dtype_bytes=dtype,seconds=str(seconds),role=role)

class PredictionValidationTests(unittest.TestCase):
    def test_exact_linear(self):
        out=validate([r('10x10',1,'calibration'),r('20x10',2,'validation')])
        self.assertEqual(out[0]['epsilon_seconds'],0)
    def test_nonzero_residual(self):
        out=validate([r('10',1,'calibration'),r('20',3,'validation')])
        self.assertAlmostEqual(out[0]['relative_error_percent'],100/3)
    def test_multiple_machine_groups(self):
        out=validate([r('10',1,'calibration'),r('20',2,'validation'),r('10',0.5,'calibration','GPU'),r('20',1,'validation','GPU')])
        self.assertEqual(len(out),2)
    def test_no_reference_rejected(self):
        with self.assertRaises(ValueError): validate([r('20',2,'validation')])
    def test_duplicate_reference_rejected(self):
        with self.assertRaises(ValueError): validate([r('10',1,'calibration'),r('20',2,'calibration')])
    def test_invalid_measurement_rejected(self):
        with self.assertRaises(ValueError): validate([r('10',1,'calibration'),r('20',-2,'validation')])
    def test_zero_extent_rejected(self):
        with self.assertRaises(ValueError): validate([r('0x10',1,'calibration')])
if __name__=='__main__': unittest.main()
