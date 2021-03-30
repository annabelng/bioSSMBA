import argparse
import os

def threshold_labels(args):
    with open(args.output, 'w') as ofile:
        with open(args.input, 'r') as ifile:
            for line in ifile:
                vals = [float(v) for v in line.strip().split()]
                if vals[1] > args.threshold:
                    ofile.write('1\n')
                else:
                    ofile.write('0\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input file')
    parser.add_argument('-t', '--threshold', type=float, help='positive label threshold')
    parser.add_argument('-o', '--output', type=str, help='output file')
    args = parser.parse_args()
    threshold_labels(args)

