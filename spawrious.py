import os
import tarfile
import urllib
import urllib.request
from typing import Any, Tuple

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def _extract_dataset_from_tar(
    tar_file_name: str, data_dir: str, remove_tar_after_extracting: bool = True
) -> None:
    tar_file_dst = os.path.join(data_dir, tar_file_name)
    print("Extracting dataset...")
    tar = tarfile.open(tar_file_dst, "r:gz")
    tar.extractall(os.path.dirname(tar_file_dst))
    tar.close()
    print("Dataset extracted. Delete tar file.")
    if remove_tar_after_extracting:
        os.remove(tar_file_dst)


def _download_dataset_if_not_available(
    dataset_name: str, data_dir: str, remove_tar_after_extracting: bool = True
) -> None:
    """
    datasets.txt file, which is present in the data_dir, is used to check if the dataset is already extracted. If the dataset is already extracted, then the tar file is not downloaded again.
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset_name = dataset_name.lower()
    if dataset_name.split("_")[0] == "m2m":
        dataset_name = "m2m"
    url_dict = {
        "entire_dataset": "https://www.dropbox.com/s/383xmavhd3x0w2k/spawrious224_with_domain_adaptation.gz?dl=1",
        "o2o_easy": "https://www.dropbox.com/s/kwhiv60ihxe3owy/spawrious__o2o_easy.tar.gz?dl=1",
        "o2o_medium": "https://www.dropbox.com/s/x03gkhdwar5kht4/spawrious224__o2o_medium.tar.gz?dl=1",
        "o2o_hard": "https://www.dropbox.com/s/p1ry121m2gjj158/spawrious__o2o_hard.tar.gz?dl=1",
        "m2m": "https://www.dropbox.com/s/5usem63nfub266y/spawrious__m2m.tar.gz?dl=1",
    }
    tar_file_name = f"spawrious__{dataset_name}.tar.gz"
    tar_file_dst = os.path.join(data_dir, tar_file_name)
    url = url_dict[dataset_name]
    # Check if the tar file is already downloaded and present in the data_dir
    if os.path.exists(tar_file_dst):
        print("Dataset already downloaded.")
        # Check if the datasets.txt file is present, and if the dataset is already extracted
        if os.path.exists(os.path.join(data_dir, "datasets.txt")):
            with open(os.path.join(data_dir, "datasets.txt"), "r") as f:
                # lines = set(f.readlines())
                lines = [line.strip() for line in f]
                if (dataset_name in lines) or ("entire_dataset" in lines):
                    print("... and extracted.")
                else:
                    print("Dataset not extracted. Extracting...")
                    _extract_dataset_from_tar(
                        tar_file_name, data_dir, remove_tar_after_extracting
                    )

                    # Write the dataset name to the datasets.txt file to mark extraction
                    with open(os.path.join(data_dir, "datasets.txt"), "a") as f:
                        f.write("\n" + dataset_name)
        # If the datasets.txt file is not present, then extract the dataset
        else:
            print("Dataset not extracted. Extracting...")
            _extract_dataset_from_tar(
                tar_file_name, data_dir, remove_tar_after_extracting
            )
            # Write the dataset name to the datasets.txt file to mark extraction
            with open(os.path.join(data_dir, "datasets.txt"), "a") as f:
                f.write("\n" + dataset_name)
    # Check if the dataset is already extracted by inspecting the datasets.txt file
    else:
        download = True
        # Check if the datasets.txt file is present, and if the dataset is already extracted
        if os.path.exists(os.path.join(data_dir, "datasets.txt")):
            with open(os.path.join(data_dir, "datasets.txt"), "r") as f:
                # lines = set(f.readlines())
                lines = [line.strip() for line in f]
                if (dataset_name in lines) or ("entire_dataset" in lines):
                    print("Dataset already downloaded and extracted.")
                    download = False
        # Download if the dataset is not already extracted
        if download:
            print("Dataset not found. Downloading...")
            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 1024
            # Track progress of download
            progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(tar_file_dst, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    progress_bar.update(len(buffer))
            progress_bar.close()
            print("Dataset downloaded. Extracting...")
            _extract_dataset_from_tar(
                tar_file_name, data_dir, remove_tar_after_extracting
            )
            # Write the dataset name to the datasets.txt file to mark extraction
            with open(os.path.join(data_dir, "datasets.txt"), "a") as f:
                f.write("\n" + dataset_name)


class CustomImageFolder(Dataset):
    """
    A class that takes one folder at a time and loads a set number of images in a folder and assigns them a specific class
    """

    def __init__(self, folder_path, class_index, limit=None, transform=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.image_paths = [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.class_index, dtype=torch.long)
        return img, label


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


def build_combination(benchmark_type, group, test, filler=None):
    total = 3168
    combinations = {}
    if "m2m" in benchmark_type:
        counts = [total, total]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[1], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[0], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[3], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[2], counts[1])],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[1]],
            ("dachshund",): [test[1], test[0]],
            ("labrador",): [test[2], test[3]],
            ("corgi",): [test[3], test[2]],
        }
    else:
        counts = [int(0.97 * total), int(0.87 * total)]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[0], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[1], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[2], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[3], counts[1])],
            ("bulldog", "dachshund", "labrador", "corgi"): [
                (filler, total - counts[0]),
                (filler, total - counts[1]),
            ],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[0]],
            ("dachshund",): [test[1], test[1]],
            ("labrador",): [test[2], test[2]],
            ("corgi",): [test[3], test[3]],
        }
    return combinations


def get_combinations(benchmark_type: str) -> Tuple[dict, dict]:
    combinations = {
        "o2o_easy": (
            ["desert", "jungle", "dirt", "snow"],
            ["dirt", "snow", "desert", "jungle"],
            "beach",
        ),
        "o2o_medium": (
            ["mountain", "beach", "dirt", "jungle"],
            ["jungle", "dirt", "beach", "snow"],
            "desert",
        ),
        "o2o_hard": (
            ["jungle", "mountain", "snow", "desert"],
            ["mountain", "snow", "desert", "jungle"],
            "beach",
        ),
        "m2m_hard": (
            ["dirt", "jungle", "snow", "beach"],
            ["snow", "beach", "dirt", "jungle"],
            None,
        ),
        "m2m_easy": (
            ["desert", "mountain", "dirt", "jungle"],
            ["dirt", "jungle", "mountain", "desert"],
            None,
        ),
        "m2m_medium": (
            ["beach", "snow", "mountain", "desert"],
            ["desert", "mountain", "beach", "snow"],
            None,
        ),
    }
    if benchmark_type not in combinations:
        raise ValueError("Invalid benchmark type")
    group, test, filler = combinations[benchmark_type]
    return build_combination(benchmark_type, group, test, filler)


class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]
    input_shape = (3, 224, 224)
    num_classes = 4
    class_list = ["bulldog", "corgi", "dachshund", "labrador"]

    def __init__(self, benchmark, root_dir, augment=True):
        combinations = get_combinations(benchmark.lower())
        self.type1 = benchmark.lower().startswith("o2o")
        train_datasets, test_datasets = self._prepare_data_lists(
            combinations["train_combinations"],
            combinations["test_combinations"],
            root_dir,
            augment,
        )
        self.datasets = [ConcatDataset(test_datasets)] + train_datasets

    def get_train_dataset(self):
        return torch.utils.data.ConcatDataset(self.datasets[1:])

    def get_test_dataset(self):
        return self.datasets[0]

    # Prepares the train and test data lists by applying the necessary transformations.
    def _prepare_data_lists(
        self, train_combinations, test_combinations, root_dir, augment
    ):
        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.input_shape[1], self.input_shape[2])),
                transforms.transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if augment:
            train_transforms = transforms.Compose(
                [
                    transforms.Resize((self.input_shape[1], self.input_shape[2])),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            train_transforms = test_transforms

        train_data_list = self._create_data_list(
            train_combinations, root_dir, train_transforms
        )
        test_data_list = self._create_data_list(
            test_combinations, root_dir, test_transforms
        )

        return train_data_list, test_data_list

    # Creates a list of datasets based on the given combinations and transformations.
    def _create_data_list(self, combinations, root_dir, transforms):
        data_list = []
        if isinstance(combinations, dict):

            # Build class groups for a given set of combinations, root directory, and transformations.
            for_each_class_group = []
            cg_index = 0
            for classes, comb_list in combinations.items():
                for_each_class_group.append([])
                for ind, location_limit in enumerate(comb_list):
                    if isinstance(location_limit, tuple):
                        location, limit = location_limit
                    else:
                        location, limit = location_limit, None
                    cg_data_list = []
                    for cls in classes:
                        path = os.path.join(
                            root_dir,
                            "spawrious224",
                            f"{0 if not self.type1 else ind}/{location}/{cls}",
                        )
                        data = CustomImageFolder(
                            folder_path=path,
                            class_index=self.class_list.index(cls),
                            limit=limit,
                            transform=transforms,
                        )
                        cg_data_list.append(data)

                    for_each_class_group[cg_index].append(ConcatDataset(cg_data_list))
                cg_index += 1

            for group in range(len(for_each_class_group[0])):
                data_list.append(
                    ConcatDataset(
                        [
                            for_each_class_group[k][group]
                            for k in range(len(for_each_class_group))
                        ]
                    )
                )
        else:
            for location in combinations:
                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(root=path, transform=transforms)
                data_list.append(data)

        return data_list


def get_spawrious_dataset(dataset_name: str, root_dir: str):
    """
    Returns the dataset as a torch dataset, and downloads it if it is not already available.
    """
    assert dataset_name.lower() in {
        "o2o_easy",
        "o2o_medium",
        "o2o_hard",
        "m2m_easy",
        "m2m_medium",
        "m2m_hard",
        "m2m",
        "entire_dataset",
    }, f"Invalid dataset type: {dataset_name}"
    _download_dataset_if_not_available(dataset_name, root_dir)
    return SpawriousBenchmark(dataset_name, root_dir, augment=True)
