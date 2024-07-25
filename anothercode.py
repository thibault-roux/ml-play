import numpy as np
from sklearn.linear_model import Perceptron

if __name__ == "__main__":
    number = 1000000
    # generate random size in centimeters for women
    women_sizes = np.random.normal(160, 10, number)
    # generate random size in centimeters for men
    men_sizes = np.random.normal(180, 10, number)


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

    # # print weights and bias
    # print(model.coef_) # weights
    # print(model.intercept_) # bias


    # plot a small subset of the dataset
    dataset = dataset[:10]
    # sort the dataset by size
    dataset = sorted(dataset, key=lambda x: x[0])
    # print each element
    for x in dataset:
        # print the size, the prediction, the score and the target
        print(x[0], model.predict(np.array(x[0]).reshape(1, -1)), model.decision_function(np.array(x[0]).reshape(1, -1)), x[1])





