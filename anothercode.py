import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

if __name__ == "__main__":
    number = 1000000
    # generate random size in centimeters for women
    women_sizes = np.random.normal(-150, 100, number)
    # generate random size in centimeters for men
    men_sizes = np.random.normal(185, 100, number)


    # create a shuffle dataset with the sizes of both women and men
    dataset = []
    for i in range(len(women_sizes)):
        size = women_sizes[i]
        dataset.append((size, 0))
    for i in range(len(men_sizes)):
        size = men_sizes[i]
        dataset.append((size, 1))

    # shuffle the dataset and split in training in testing set
    np.random.shuffle(dataset)
    train_set = dataset[:number]
    test_set = dataset[number:]

    # create the model which is a perceptron with 2 inputs and 1 output
    model = Perceptron()
    # train the model
    print("Training...")
    model.fit([x[:1] for x in train_set], [x[1] for x in train_set])
    # test the model
    print("Testing...")
    print(model.score([x[:1] for x in test_set], [x[1] for x in test_set]))

    # print weights and bias
    print(model.coef_) # weights
    print(model.intercept_) # bias


    # plot a small subset of the dataset
    dataset = dataset[:100]
    # sort the dataset by size
    dataset = sorted(dataset, key=lambda x: x[0])
    # print each element
    for x in dataset:
        # print the size, the prediction, the score and the target
        size = x[0]
        prediction = model.predict(np.array(x[0]).reshape(1, -1))
        score = model.decision_function(np.array(x[0]).reshape(1, -1))
        target = x[1]
        # print(size, prediction, score, target)

    # plot the decision line
    x = np.linspace(-150, 200, 100)
    y = model.decision_function(x.reshape(-1, 1))
    plt.plot(x, y)
    
    # scatter plot the dataset
    for x in dataset:
        size = x[0]
        target = x[1]
        if target == 0:
            plt.scatter(size, 0, color="red", s=20)
        else:
            plt.scatter(size, 0, color="blue", s=20)

    plt.savefig("decision_boundary2.png")



