
import mnist_loader
from network2 import Network, CrossEntropyCost

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = Network([784, 30, 10], cost=CrossEntropyCost)
net.large_weight_initializer()
net.SGD(
    training_data=training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.1,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True
)
