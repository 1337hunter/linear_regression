#!/usr/bin/env python3
import sys
import re
import torch

def ReadWeights():
    try:
        with open("./weights.csv", 'r') as file:
            content = file.read().strip().split(',')
            return [float(content[0]), float(content[1])]
    except OSError as error:
        return [0, 0]

def GetMeanStd():
    with open("data.csv", 'r') as file:
        i = 0;
        X = []
        Y = []
        file.readline()
        for line in file:
            row = line.strip().split(',')
            X += [float(row[0])]
            Y += [float(row[1])]
        torchX = torch.FloatTensor(X)
        torchY = torch.FloatTensor(Y)
        return torchX.mean(), torchX.std(unbiased=True), \
               torchY.mean(), torchY.std(unbiased=True), \
               torchX.min(), torchX.max(), \
               torchY.min(), torchY.max()

def NormalizeNormal(arg, mean, std):
    return (arg - mean) / std

def NormalizeLinear(arg, minimum, maximum):
    return (arg - minimum) / (maximum - minimum)

def UnnormalizeNormal(arg, mean, std):
    return arg * std + mean

def UnnormalizeLinear(arg, minimum, maximum):
    return arg * (maximum - minimum) + minimum

def estimatePrice(km, weights):
    return weights[0] + weights[1] * km

def main():
    weights = ReadWeights()
    meanX, stdX, meanY, stdY, minX, maxX, minY, maxY = GetMeanStd();
    #arg = NormalizeLinear(float(sys.argv[1]), minX, maxX)
    arg = NormalizeNormal(float(sys.argv[1]), meanX, stdX)
    rez = estimatePrice(arg, weights)
    #rez = UnnormalizeLinear(rez, minY, maxY)
    rez = UnnormalizeNormal(rez, meanY, stdY)

    print(int(rez) if rez >= 0 else 0)


def CheckArguments():
    if len(sys.argv) != 2:
        print("Usage: 'python3 predict.py <km>'")
        exit()
    if re.match(r'^-?\d+(?:\.\d+)$', sys.argv[1]) is None and not sys.argv[1].isdigit():
        print('<km> must be a positive digit.')
        exit()


if __name__ == '__main__':
    CheckArguments()
    main()
