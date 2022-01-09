from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

def load_data():
    '''
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data splitted in ratio 70:15:15.
    Each data is returned as a tuple with two elements. The first
    contains images and the second contains number which are represented
    on image.
    '''
    digits = datasets.load_digits()

    tuple_data = [data for data in zip(digits.data, digits.target)]

    train, other = train_test_split(tuple_data, train_size=0.7)
    validation, test = train_test_split(other, test_size=0.5)

    train_data = [(np.reshape(data[0], (64, 1)), vectorized_result(data[1])) for data in train]
    validation_data = [(np.reshape(data[0], (64, 1)), data[1]) for data in validation]
    test_data = [(np.reshape(data[0], (64, 1)), data[1]) for data in test]
    
    return (train_data, validation_data, test_data)

def vectorized_result(result_digit):
    '''
    Return a 10-dimensional unit vector with a 1 where result_digit is
    and zeroes elsewhere.
    '''
    vector = np.zeros((10, 1))
    vector[result_digit] = 1.0
    return vector
