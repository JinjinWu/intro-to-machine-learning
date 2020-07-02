import matplotlib.pyplot as plt
from utils import load_train, load_valid
from run_knn import run_knn

train_inputs, train_targets = load_train()
valid_inputs, valid_targets = load_valid()

k = [1, 3, 5, 7, 9]
rates = []

for i in k:
    predictions = run_knn(i, train_inputs, train_targets, valid_inputs)
    num_correct = 0
    for j in range(len(predictions)):
        if predictions[j] == valid_targets[j]:
            num_correct += 1
    rates.append(num_correct/len(predictions))

plt.plot(k, rates)
plt.xlabel('k')
plt.ylabel('misclassification rate')
plt.savefig('./graphs/2.1.png')
