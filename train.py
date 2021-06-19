#!/usr/bin/env python3
import os
import sys

class   LinearRegression:
    def __init__(self, data_path, weights_path):
        self.data_path = data_path
        self.weights_path = weights_path
        self.weights = []
        self.train_dataset = []

    def h(self, x):
        return self.weights[0] + self.weights[1] * x

    def write_weights(self):
        try:
            with open(self.weights_path, 'w+') as file:
                for x in self.weights:
                    file.write(str(x) + '\n')
        except OSError as error:
            print(error)
            exit()
    def read_train_data(self):
        try:
            with open(self.data_path, 'r') as file:
                file.readline();
                line = file.readline().strip()
                while line:
                    self.train_dataset += [line.split(',')]
                    line = file.readline().strip()
        except OSError as error:
            print(error)
            exit()
    def train(self):
        self.weights = [0, 0]



def main():
    model = LinearRegression(sys.argv[1], sys.argv[2])
    model.read_train_data()
    model.train()

def CheckArguments():
    if len(sys.argv) != 3:
        print("Usage: 'python3 train.py <path to train dataset> <path to file with weights>'")
        exit()


if __name__ == '__main__':
    CheckArguments()
    main()
