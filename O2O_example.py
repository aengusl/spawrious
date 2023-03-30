import torch
import torchvision
import torchvision.transforms as transforms

from torchvision import models
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


from spawrious import Spawrious020_easy, download_spawrious

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

path_to_dataset = ""

batch_size = 64

# downloads the dataset
download_spawrious(path_to_dataset)

datasets = Sparious020_easy(path_to_dataset)

trainloaders = zip(
    torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2) for trainset in datasets.datasets[1:]
)

testloader = torch.utils.data.DataLoader(datasets.datasets[0], batch_size=batch_size,shuffle=False, num_workers=2)

classes = ('bulldog', 'corgi', 'dachshund', 'labrador')


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(testloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


net = models.resnet18(pretrained=True)
num_ftrs = 512
net.fc = torch.nn.Linear(num_ftrs, num_classes)

net.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

## train

for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, batches in enumerate(trainloader, 0):
        for data in batches:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
    print(f'[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}')
    running_loss = 0.0

print('Finished Training')


dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
