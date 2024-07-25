import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

if __name__ == "__main__":
    number = 10000
    # generate random size in centimeters for women
    women_sizes = np.random.normal(130, 10, number)
    # generate random weight in kilograms for women
    women_weights = np.random.normal(40, 10, number)
    # generate random size in centimeters for men
    men_sizes = np.random.normal(190, 10, number)
    # generate random weight in kilograms for men
    men_weights = np.random.normal(90, 10, number)


    # create a shuffle dataset with the sizes and weight of both women and men
    dataset = []
    for i in range(len(women_sizes)):
        size = women_sizes[i]
        weight = women_weights[i]
        dataset.append((size, weight, 0))
    for i in range(len(men_sizes)):
        size = men_sizes[i]
        weight = men_weights[i]
        dataset.append((size, weight, 1))

    # shuffle the dataset and split in training in testing set
    np.random.shuffle(dataset)
    train_set = dataset[:number]
    test_set = dataset[number:]

    # create the model which is a perceptron with 2 inputs and 1 output
    model = Perceptron()
    # train the model
    print("Training...")
    model.fit([x[:2] for x in train_set], [x[2] for x in train_set])
    # test the model
    print("Testing...")
    print(model.score([x[:2] for x in test_set], [x[2] for x in test_set]))

    # print weights and bias
    print(model.coef_) # weights
    print(model.intercept_) # bias


    # plot the decision boundary
    display = DecisionBoundaryDisplay.from_estimator(model, X=np.array([x[:2] for x in dataset]), xlabel="Size", ylabel="Weight")
    plt.plot()
    plt.savefig("decision_boundary.png")




