import numpy as np
import matplotlib.pyplot as plt
from check_grad import check_grad
from utils import *
from logistic import *

def run_logistic_regression():
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.2,
                    'weight_regularization': 0.5,
                    'num_iterations': 300
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.ones((M+1, 1)) / M+1

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)


    iteration = []
    train_ce = []
    valid_ce = []
    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        iteration.append(t+1)
        train_ce.append(cross_entropy_train)
        valid_ce.append(cross_entropy_valid)

        # print some stats
        print("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}".format(
                   t+1, float(f / N), float(cross_entropy_train), float(frac_correct_train*100),
                   float(cross_entropy_valid), float(frac_correct_valid*100)))



    plt.plot(iteration, train_ce, label='training data')
    plt.plot(iteration, valid_ce, label='validation data')
    plt.xlabel('iteration number')
    plt.ylabel('cross entropy')
    plt.savefig('./graphs/2.2.png')

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)

if __name__ == '__main__':
    run_logistic_regression()
