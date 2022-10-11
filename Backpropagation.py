# Backpropagation algorithm

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    # Derivative of activation function
    def sigmoid_derivative(x):
        return x * (1 - x)


    # Input dataset
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    # Output dataset
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # Seed the random number generator
    np.random.seed(1)

    # Initialize weights randomly with mean 0
    w0 = 2 * np.random.random((3, 4)) - 1
    w1 = 2 * np.random.random((4, 1)) - 1

    # Store the weights
    weights = [w0, w1]

    # Store the errors
    errors = []

    # Store the accuracy
    accuracy = []

    # Store the number of epochs
    epochs = []


    # Training loop
    for i in range(1000):

            # Feed forward through layers 0, 1, and 2
            l0 = X
            l1 = sigmoid(np.dot(l0, w0))
            l2 = sigmoid(np.dot(l1, w1))

            # Calculate error
            l2_error = y - l2

            # Calculate accuracy
            acc = 1 - np.mean(np.abs(l2_error))
            accuracy.append(acc)

            # Calculate error
            errors.append(np.mean(np.abs(l2_error)))

            # Calculate number of epochs
            epochs.append(i)

            # Calculate the gradient
            l2_delta = l2_error * sigmoid_derivative(l2)
            l1_error = l2_delta.dot(w1.T)
            l1_delta = l1_error * sigmoid_derivative(l1)

            # Update the weights
            w1 += l1.T.dot(l2_delta)
            w0 += l0.T.dot(l1_delta)

            # Store the weights
            weights = [w0, w1]

            # Print the error
            if (i % 1000) == 0:
                print("Error: " + str(np.mean(np.abs(l2_error))))
                print("Accuracy: " + str(acc))
                print("Epoch: " + str(i))
                print("")

    # Plot the error
    plt.plot(epochs, errors)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

    # Plot the accuracy
    plt.plot(epochs, accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
