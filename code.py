import numpy as np
import sklearn

if __name__ == "__main__":
    # generate random size in centimeters for women
    women_sizes = np.random.randint(140, 160, 100)
    # generate random size in centimeters for men
    men_sizes = np.random.randint(170, 200, 100)

    # create a shuffle dataset with the sizes of both women and men
    dataset = []
    for size in women_sizes:
        dataset.append((size, 0))
    for size in men_sizes:
        dataset.append((size, 1))
    print(dataset)