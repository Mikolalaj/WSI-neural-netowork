

import mnist_loader
from network import Network

training_data, validation_data, test_data = mnist_loader.load_data()

net = Network([64, 25, 10])
net.stochastic_gradient_descent(
    training_data=training_data,
    epochs=40,
    batch_size=15,
    learn_rate=0.1,
    lambda_=5.0,
    test_data=test_data
)
