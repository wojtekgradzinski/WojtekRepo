#train & validate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
np.random.seed(42)

epochs = 1000
learning_rate = 0.01

def train(train_features, train_targets, test_features, test_targets, epochs, learning_rate):
    weights = np.random.normal(size=train_features.shape[1])
    bias = 0
    errors = []
    test_errors = []
    for epoch in range(epochs):
        for x, y in zip(train_features, train_targets):
            output = output_formula(x, weights, bias)
            weights, bias = update_weights(x, y, weights, bias, learning_rate)
        # at the end of one epoch
        out = output_formula(train_features, weights, bias)
        loss = np.mean(error_formula(train_targets, out))
        errors.append(loss)
        out_test = output_formula(test_features, weights, bias)
        loss_test = np.mean(error_formula(test_targets, out_test))
        test_errors.append(loss_test)
        
        if epoch % 10 == 0:
            print("Epoch:", epoch)
            print("Train loss", loss)
            predictions = out > 0.5
            accuracy = np.mean(predictions == train_targets)
            print("Train Accuracy", accuracy)
            print("Test loss", loss_test)
            predictions = out_test > 0.5
            accuracy = np.mean(predictions == test_targets)
            print("Test Accuracy", accuracy)
        plt.plot(errors)
        plt.plot(test_errors)
        