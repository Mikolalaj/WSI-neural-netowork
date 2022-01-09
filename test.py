
import mnist_loader
from network import Network, CrossEntropyCost

training_data, validation_data, test_data = mnist_loader.load_data()

net = Network([64, 30, 10], cost=CrossEntropyCost)
net.SGD(
    training_data=training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.1,
    lambda_=5.0,
    evaluation_data=test_data
)
