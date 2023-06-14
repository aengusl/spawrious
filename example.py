import argparse

import torch
import torch.optim as optim
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from tqdm.auto import tqdm

from spawrious.torch import get_spawrious_dataset


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
        "--val_split",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--data_dir", type=str, default="./data/", help="path to the dataset directory"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=2, help="number of workers for data loading"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--num_epochs", type=int, default=3, help="number of epochs")
    return parser.parse_args()


def train(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Module,
    num_epochs: int,
    device: torch.device,
) -> None:

    for epoch in tqdm(range(num_epochs), desc="Training. Epochs", leave=False):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch {epoch + 1}: Training Loss: {running_loss / len(train_loader):.3f}"
        )
        print("Evaluating on validation set...")
        evaluate(model, val_loader, device)


def evaluate(model: Module, loader: DataLoader, device: torch.device) -> None:
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Acc: {100 * correct / total:.3f}%")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spawrious = get_spawrious_dataset(dataset_name=args.dataset, root_dir=args.data_dir)
    train_set = spawrious.get_train_dataset()
    test_set = spawrious.get_test_dataset()
    val_size = int(len(train_set) * args.val_split)
    train_set, val_set = torch.utils.data.random_split(
        train_set, [len(train_set) - val_size, val_size]
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 4)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        args.num_epochs,
        device,
    )
    print("Finished training, now evaluating on test set.")
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
