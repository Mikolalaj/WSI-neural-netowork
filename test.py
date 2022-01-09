

from data_loader import load_data
from network import Network

image_size=64 # 8x8

training_data, validation_data, test_data = load_data(image_size)

net = Network([image_size, 25, 10])
net.stochastic_gradient_descent(
    training_data=training_data,
    test_data=test_data,
    epochs=40,
    batch_size=15,
    learn_rate=0.1,
    lambda_=5.0
)
