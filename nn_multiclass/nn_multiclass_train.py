import numpy as np
import torch
from torch import nn, optim
from tqdm import trange

from nn_multiclass.nn_multiclass import Net

LEARNING_RATE = 0.001
MOMENTUM = 0.9
MAX_ITERATIONS = 500
INPUT_SIZE = 2
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1

# Initialize the network
net = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# Define the loss criterion and the training algorithm
criterion = nn.BCEWithLogitsLoss()  # Be careful, use binary cross entropy for binary, CrossEntropy for Multi-class
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


def train_neural_network_pytorch(net, inputs, labels, optimizer, criterion, iterations=10000):
    """
    Function for training the PyTorch network.

    :param net: the neural network object
    :param inputs: numpy array of training data values
    :param labels: numpy array of training data labels
    :param optimizer: PyTorch optimizer instance
    :param criterion: PyTorch loss function
    :param iterations: number of training steps
    :return: final_loss - The final loss
    """
    net.train()  # Before training, set the network to training mode
    total_loss = 0  # Initialize total loss
    for iter in trange(iterations):  # loop over the dataset multiple times

        # It is a common practice to track the losses during training
        # Feel free to do so if you want

        # Get the inputs; data is a list of [inputs, labels]
        # Convert to tensors if data is in the form of numpy arrays
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs.astype(np.float32))

        if not torch.is_tensor(labels):
            labels = torch.from_numpy(labels.astype(np.float32))


        # Follow these steps:
        # 1. Reset gradients: Zero the parameter gradients (Check the link for optimizers in the text cell
        #                     above to find the correct function)
        # 2. Forward: Pass `inputs` through the network. This can be done calling
        #             the `forward` function of `net` explicitly but there is an
        #             easier way that is more commonly used
        # 3. Compute the loss: Use `criterion` and pass it the `outputs` and `labels`
        #                      Check the link in the text cell above for details
        # 4. Backward: Call the `backward` function in `loss`
        # 5. Update parameters: This is done using the optimizer's `step` function.
        #                       Check the link provided for details.

        ### BEGIN SOLUTION

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        ### END SOLUTION