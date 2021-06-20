#!/usr/bin/env python3
import os
import sys
import random
import torch
import re

class   LinearRegression:
    def __init__(self, data_path, weights_path, lr=0.01, epoch=700):
        #there should be a validation dataset, but ok
        self.data_path = data_path
        self.epoch = epoch
        self.weights_path = weights_path
        self.weights = [random.randint(-10, 10), random.randint(-10, 10)]
        self.lr = lr
        self.X_train = torch.FloatTensor()
        self.y_train = torch.FloatTensor()
        self.X_val = torch.FloatTensor()
        self.y_val = torch.FloatTensor()

    def hypothesis(self, X_train):
        return self.weights[0] + self.weights[1] * X_train

    def write_weights(self):
        try:
            with open(self.weights_path, 'w+') as file:
                for x in self.weights:
                    file.write(str(x) + '\n')
        except OSError as error:
            print(error)
            exit()
    
    def normalize_dataset_normal(self):
        self.X_mean = float(self.X_train.mean())
        self.X_std = float(self.X_train.std(unbiased=True))
        self.X_train = (self.X_train - self.X_mean) / self.X_std
        self.y_mean = float(self.y_train.mean())
        self.y_std = float(self.y_train.std())
        self.y_train = (self.y_train - self.y_mean) / self.y_std

    def normalize_dataset_linear(self):
        self.X_min = float(self.X_train.min())
        self.X_max = float(self.X_train.max())
        self.X_train = (self.X_train - self.X_min) / (self.X_max - self.X_min)
        self.y_min = float(self.y_train.min())
        self.y_max = float(self.y_train.max())
        self.y_train = (self.y_train - self.y_min) / (self.y_max - self.y_min)

    def read_train_data(self):
        try:
            with open(self.data_path, 'r') as file:
                file.readline();
                line = file.readline().strip()
                train_dataset = []
                while line:
                    train_dataset += [line.split(',')]
                    line = file.readline().strip()
            self.X_train = torch.FloatTensor([float(x[0]) for x in train_dataset])
            self.y_train = torch.FloatTensor([float(x[1]) for x in train_dataset])
            self.normalize_dataset_normal()
            #self.normalize_dataset_linear()
            if len(train_dataset) == 0:
                print("No data in train dataset")
                exit()
        except OSError as error:
            print(error)
            exit()
    
    def save_weights(self):
        try:
            with open(self.weights_path, "w+") as file:
                file.write(str(self.weights[0]) + "," + str(self.weights[1]) + "\n")
                print(self.weights[0], self.weights[1])
        except OSError as error:
            print(error)
            exit()

    def build_up(self):
        self.read_train_data()
        self.train()

    def train(self):
        i = 0
        downgrade_lr = 1
        coefficient = self.lr
        while i < self.epoch:
            error = self.hypothesis(self.X_train) - self.y_train
            if (error * error).sqrt().mean() < 0.1:
                break
            print('RMSE:', float((error * error).sqrt().mean()))
            temp0 = self.weights[0] - coefficient * float(error.mean())
            temp1 = self.weights[1] - coefficient * float((self.X_train * error).mean())
            self.weights[0] = temp0
            self.weights[1] = temp1
            if i != 0 and i % 100 == 0:
                coefficient = self.lr / (1.2 ** downgrade_lr)
                downgrade_lr += 1
            i += 1



def main():
    model = LinearRegression("data.csv", "weights.csv", lr=0.01, epoch=700)
    model.build_up()
    model.save_weights()

if __name__ == '__main__':
    main()
