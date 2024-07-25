import numpy as np
import sklearn

if __name__ == "__main__":
    number = 100
    # generate random size in centimeters for women
    women_sizes = np.random.normal(150, 10, number)
    # generate random weight in kilograms for women
    women_weights = np.random.normal(50, 10, number)
    # generate random size in centimeters for men
    men_sizes = np.random.normal(180, 10, number)
    # generate random weight in kilograms for men
    men_weights = np.random.normal(80, 10, number)


    # create a shuffle dataset with the sizes of both women and men
    dataset = []
    for size in women_sizes:
        dataset.append((size, 0))
    for size in men_sizes:
        dataset.append((size, 1))
    print(dataset)

    # shuffle the dataset and split in training in testing set
    np.random.shuffle(dataset)
    train_set = dataset[:number]
    test_set = dataset[number:]

    # create the model

