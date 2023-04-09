import argparse

import torch
import torch.optim as optim
from torch import nn
from torchvision import models
from tqdm import tqdm

from spawrious import get_torch_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 on the Spawrious O2O-easy dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="o2o_easy",
        help="name of the dataset",
        choices=[
            "o2o_easy",
            "o2o_medium",
            "o2o_hard",
            "m2m_easy",
            "m2m_medium",
            "m2m_hard",
        ],
    )

    parser.add_argument(
        "--data_dir", type=str, default="./data/", help="path to the dataset directory"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=2, help="number of workers for data loading"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs")
    return parser.parse_args()


def train(model, train_loader, optimizer, criterion, num_epochs, device):
    for epoch in tqdm(
        range(num_epochs), desc="Epochs"
    ):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}]: Loss: {running_loss / len(train_loader):.3f}")


def evaluate(model, test_loader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct // total} %")


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = get_torch_dataset(dataset_name=args.dataset, root_dir=args.data_dir)
    train_sets = datasets.datasets[1:]
    trainset = torch.utils.data.ConcatDataset(train_sets)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.datasets[0],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 4)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(model, train_loader, optimizer, criterion, args.num_epochs, device)
    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
