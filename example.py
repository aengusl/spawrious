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
import timm
from spawrious.torch import get_spawrious_dataset
import wandb

# # MODEL_NAME = "vit_so400m_patch14_siglip_384"
# # MODEL_NAME = 'swin_base_patch4_window7_224.ms_in22k_ft_in1k'
# MODEL_NAME = 'deit3_base_patch16_224.fb_in22k_ft_in1k'
from spawrious.torch import MODEL_NAME
from spawrious.torch import set_model_name

set_model_name('deit3_base_patch16_224.fb_in22k_ft_in1k')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 on the Spawrious O2O-easy dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="m2m_hard",
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
    parser.add_argument("--model", type=str, default="siglip", help="model name")
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
        for inputs, labels, _ in tqdm(train_loader):  # third item is the location label
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
        val_acc = evaluate(model, val_loader, device)
        wandb.log(
            {"train_loss": running_loss / len(train_loader), "val_acc": val_acc},
            step=epoch,
        )


def evaluate(model: Module, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in tqdm(
            loader, desc="Evaluating", leave=False
        ):  # third item is the location label
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Acc: {acc:.3f}%")
    return acc


class ClassifierOnTop(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            # "vit_so400m_patch14_siglip_384",
            MODEL_NAME,
            pretrained=True,
            num_classes=0,
        ).eval()
        self.linear = nn.Linear(1152, num_classes)
        if MODEL_NAME == 'swin_base_patch4_window7_224.ms_in22k_ft_in1k':
            self.linear = nn.Linear(1024, num_classes)
        elif MODEL_NAME == 'deit3_base_patch16_224.fb_in22k_ft_in1k':
            self.linear = nn.Linear(768, num_classes)
        elif MODEL_NAME == 'beit_base_patch16_224.in22k_ft_in22k_in1k':
            self.linear = nn.Linear(768, num_classes)
        elif MODEL_NAME == 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k':
            self.linear = nn.Linear(768, num_classes)
        elif MODEL_NAME == 'levit_128s.fb_dist_in1k':
            self.linear = nn.Linear(384, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        return self.linear(x)


def get_model(args: argparse.Namespace) -> Module:
    if args.model == "siglip":
        model = ClassifierOnTop(num_classes=4)
    else:
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 4)
    return model


def main() -> None:
    args = parse_args()
    experiment_name = f"{args.dataset}_{MODEL_NAME.split('_')[0]}-e={args.num_epochs}-lr={args.lr}"
    wandb.init(project="spawrious", name=experiment_name, config=args)
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

    model = get_model(args)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
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
    torch.save(model.state_dict(), f"{experiment_name}.pt")
    test_acc = evaluate(model, test_loader, device)
    wandb.log({"final_test_acc": test_acc}, step=args.num_epochs)


if __name__ == "__main__":
    main()
