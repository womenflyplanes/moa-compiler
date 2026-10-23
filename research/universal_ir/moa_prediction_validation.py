"""v3.4 additive held-out performance validation; no hardware timings invented.

Input CSV columns: machine,workload,shape,dtype_bytes,seconds,role
shape is e.g. 128x256; role is calibration or validation.
Each (machine,workload,dtype_bytes) requires exactly one calibration size.
Predicts compute-only linear throughput; reports residual, not claimed optimality.
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

REQUIRED = ('machine','workload','shape','dtype_bytes','seconds','role')

def parse_shape(value):
    dims = tuple(int(v) for v in value.split('x'))
    if not dims or any(v < 0 for v in dims):
        raise ValueError('shape must be nonempty, nonnegative dimensions joined by x')
    return dims

def validate(rows):
    groups = defaultdict(list)
    for row in rows:
        if not all(key in row for key in REQUIRED):
            raise ValueError('Missing CSV columns')
        shape = parse_shape(row['shape'])
        size = math.prod(shape)
        seconds = float(row['seconds'])
        dtype = int(row['dtype_bytes'])
        if size <= 0 or dtype <= 0 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Measured timings require positive sizes, bytes and seconds')
        if row['role'] not in ('calibration','validation'):
            raise ValueError('role must be calibration or validation')
        groups[(row['machine'],row['workload'],dtype)].append((row,shape,size,seconds))
    result = []
    for key, entries in groups.items():
        refs = [e for e in entries if e[0]['role'] == 'calibration']
        if len(refs) != 1:
            raise ValueError(f'Exactly one calibration record required per group: {key}')
        ref = refs[0]
        for row, shape, n, seconds in entries:
            if row['role'] != 'validation':
                continue
            predicted = n * ref[3] / ref[2]
            result.append(dict(machine=key[0],workload=key[1],shape=row['shape'],
                               dtype_bytes=key[2],elements=n,reference_shape=ref[0]['shape'],
                               predicted_seconds=predicted,measured_seconds=seconds,
                               epsilon_seconds=seconds-predicted,
                               relative_error_percent=100*(seconds-predicted)/seconds,
                               absolute_percentage_error=100*abs(seconds-predicted)/seconds))
    return result

def read_csv(path):
    with open(path,newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path,records):
    if not records:
        raise ValueError('No held-out validation records found')
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements_csv')
    parser.add_argument('predictions_csv')
    args=parser.parse_args()
    results=validate(read_csv(args.measurements_csv))
    write_csv(args.predictions_csv,results)
    print(f'Held-out records: {len(results)}')
    print(f'Mean absolute percentage error: {statistics.mean(r["absolute_percentage_error"] for r in results):.3f}%')
    print('Predictions are linear compute-only baselines; not hardware validation until measured CSV is supplied.')
if __name__=='__main__':
    main()
