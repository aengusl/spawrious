import os
import tarfile
import urllib
import urllib.request
from typing import Any, Tuple
from tqdm import tqdm

import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory

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
    data_dir = data_dir.split("/spawrious224/")[0] # in case people pass in the wrong root_dir
    os.makedirs(data_dir, exist_ok=True)
    dataset_name = dataset_name.lower()
    if dataset_name.split("_")[0] == "m2m":
        dataset_name = "entire_dataset"
    url_dict = {
        "entire_dataset": "https://www.dropbox.com/s/1gkwqut1p735ccn/spawrious224__entire_dataset.tar.gz?dl=1",
        "o2o_easy": "https://www.dropbox.com/s/kwhiv60ihxe3owy/spawrious224__o2o_easy.tar.gz?dl=1",
        "o2o_medium": "https://www.dropbox.com/s/x03gkhdwar5kht4/spawrious224__o2o_medium.tar.gz?dl=1",
        "o2o_hard": "https://www.dropbox.com/s/p1ry121m2gjj158/spawrious224__o2o_hard.tar.gz?dl=1",
        # "m2m": "https://www.dropbox.com/s/5usem63nfub266y/spawrious__m2m.tar.gz?dl=1",
    }
    tar_file_name = f"spawrious224__{dataset_name}.tar.gz"
    tar_file_dst = os.path.join(data_dir, tar_file_name)
    url = url_dict[dataset_name]

    # check if the dataset is already extracted
    if _check_images_availability(data_dir, dataset_name):
        print("Dataset already downloaded and extracted.")
        return
    # check if the tar file is already downloaded
    else:
        if os.path.exists(tar_file_dst):
            print("Dataset already downloaded. Extracting...")
            _extract_dataset_from_tar(
                tar_file_name, data_dir, remove_tar_after_extracting
            )
            return
        # download the tar file and extract from it
        else:
            print('Dataset not found. Downloading...')
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
            return

class ConcatDataset(tf.data.Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.total_len = sum([len(ds) for ds in dataset_list])

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        for ds in self.dataset_list:
            if index < len(ds):
                return ds[index]
            index -= len(ds)
        raise IndexError("Index out of range")

class CustomImageFolder(tf.keras.utils.Sequence):
    """
    A class that takes one folder at a time and loads a set number of images in a folder and assigns them a specific class
    """

    def __init__(self, folder_path, class_index, limit=None, preprocess_func=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.image_paths = [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.preprocess_func = preprocess_func

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = tf.keras.preprocessing.image.img_to_array(img)

        if self.preprocess_func:
            img = self.preprocess_func(img)

        label = tf.convert_to_tensor(self.class_index, dtype=tf.int64)
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

    # Prepares the train and test data lists by applying the necessary transformations.
    def _prepare_data_lists(
        self, train_combinations, test_combinations, root_dir, augment
    ):
        preprocess_input = tf.keras.applications.resnet.preprocess_input

        if augment:
            train_transforms = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.2,
            )
        else:
            train_transforms = ImageDataGenerator(
                preprocessing_function=preprocess_input,
            )

        test_transforms = ImageDataGenerator(
            preprocessing_function=preprocess_input,
        )

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
                            root_dir, f"{0 if not self.type1 else ind}/{location}/{cls}"
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
                data = image_dataset_from_directory(
                    directory=path,
                    labels="inferred",
                    label_mode="categorical",
                    class_names=None,
                    color_mode="rgb",
                    batch_size=32,
                    image_size=(224, 224),
                    shuffle=True,
                    seed=None,
                    validation_split=None,
                    subset=None,
                    interpolation="bilinear",
                    follow_links=False,
                )
                data_list.append(data)

        return data_list

def _check_images_availability(root_dir: str, dataset_type: str) -> bool:
    # Get the combinations for the given dataset type
    root_dir = root_dir.split("/spawrious224/")[0] # in case people pass in the wrong root_dir
    combinations = get_combinations(dataset_type.lower())

    # Extract the train and test combinations
    train_combinations = combinations["train_combinations"]
    test_combinations = combinations["test_combinations"]

    # Check if the relevant images for each combination are present in the root directory
    for combination in [train_combinations, test_combinations]:
        for classes, comb_list in combination.items():
            for ind, location_limit in enumerate(comb_list):
                if isinstance(location_limit, tuple):
                    location, limit = location_limit
                else:
                    location, limit = location_limit, None

                for cls in classes:
                    path = os.path.join(
                        root_dir,
                        "spawrious224",
                        f"{0 if not dataset_type.lower().startswith('o2o') else ind}/{location}/{cls}",
                    )

                    # If the path does not exist or there are no relevant images, return False
                    if not os.path.exists(path) or not any(
                        img.endswith((".png", ".jpg", ".jpeg")) for img in os.listdir(path)
                    ):
                        return False

    # If all the required images are present, return True
    return True
def get_tensorflow_dataset(dataset_name: str, root_dir: str):
    """
    Returns the dataset as a tensorflow dataset, and downloads it if it is not already available.
    """
    root_dir = root_dir.split("/spawrious224/")[0] # in case people pass in the wrong root_dir
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

if __name__ == '__main__':
    # get_spawrious_dataset('./test_dir','m2m_easy')
    root_dir = "/home/aengusl/Desktop/Projects/OOD_workshop/spawrious/data/"
    dataset_type = "m2m_easy"
    result = _check_images_availability(root_dir, dataset_type)
    print(result)
