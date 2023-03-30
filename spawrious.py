import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

import urllib
import tarfile

# download function
def download_spawrious(data_dir, remove=True):
    dst = os.path.join(data_dir, "spawrious.tar.gz")
    urllib.request.urlretrieve('https://www.dropbox.com/s/wc9mwza5yk66i83/spawrious224.tar.gz?dl=1', dst)
    tar = tarfile.open(dst, "r:gz")
    tar.extractall(os.path.dirname(dst))
    tar.close()
    if remove:
        os.remove(dst)



def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


## Spawrious base class
class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, train_combinations, test_combinations, root_dir, augment=True, type1=False):
        self.input_shape = (3,224,224)
        self.num_classes = 4

        self.type1 = type1

        train_data_list = []
        test_data_list = []

        self.class_list = ["bulldog","corgi","dachshund","labrador"]

        test_transforms_list = [
            transforms.Resize((self.input_shape[1],self.input_shape[2])),
            transforms.transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        train_transforms_list = [
            transforms.Resize((self.input_shape[1],self.input_shape[2])),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # Build test and validation transforms
        test_transforms = transforms.transforms.Compose(test_transforms_list)

        # Build training data transforms
        if augment:
            train_transforms = transforms.transforms.Compose(train_transforms_list)
        else:
            train_transforms = test_transforms

        # Make train_data_list
        if isinstance(train_combinations, dict):
            for_each_class_group = []
            cg_index = 0
            for classes,comb_list in train_combinations.items():
                for_each_class_group.append([])
                for ind, (location,limit) in enumerate(comb_list):

                    path = os.path.join(root_dir, f"{0}/{location}/")
                    if self.type1: path = os.path.join(root_dir, f"{ind}/{location}/")
                    data = ImageFolder(
                        root=path, transform=train_transforms
                    )

                    classes_idx = [data.class_to_idx[c] for c in classes]
                    to_keep_idx = []
                    for class_to_limit in classes_idx:
                        count_limit=0
                        for i in range(len(data)):
                            if data[i][1] == class_to_limit:
                                to_keep_idx.append(i)
                                count_limit+=1
                            if count_limit>=limit:
                                break

                    subset = Subset(data, to_keep_idx)

                    for_each_class_group[cg_index].append(subset) 
                cg_index+=1
            for group in range(len(for_each_class_group[0])):
                train_data_list.append(ConcatDataset([
                    for_each_class_group[k][group] for k in range(len(for_each_class_group))
                ]))
        else:
            for location in train_combinations:

                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(
                    root=path, transform=train_transforms
                )

                train_data_list.append(data) 

        # Make test_data_list
        if isinstance(test_combinations, dict):
            for_each_class_group = []
            cg_index = 0
            for classes,comb_list in test_combinations.items():
                for_each_class_group.append([])
                for ind,location in enumerate(comb_list):

                    path = os.path.join(root_dir, f"{0}/{location}/")
                    if self.type1: path = os.path.join(root_dir, f"{ind}/{location}/")
                    data = ImageFolder(
                        root=path, transform=test_transforms
                    )

                    classes_idx = [data.class_to_idx[c] for c in classes]
                    to_keep_idx = [i for i in range(len(data)) if data.imgs[i][1] in classes_idx]

                    subset = Subset(data, to_keep_idx)

                    for_each_class_group[cg_index].append(subset) 
                cg_index+=1
            for group in range(len(for_each_class_group[0])):
                test_data_list.append(ConcatDataset([
                    for_each_class_group[k][group] for k in range(len(for_each_class_group))
                ]))
        else:
            for ind, location in enumerate(test_combinations):

                path = os.path.join(root_dir, f"{0}/{location}/")
                if self.type1: path = os.path.join(root_dir, f"{ind}/{location}/")
                data = ImageFolder(root=path, transform=test_transforms)

                test_data_list.append(data) 

        # Concatenate test datasets 
        test_data = ConcatDataset(test_data_list)
 
        self.datasets = [test_data] + train_data_list
    
    def prepend_path(self,to_prepend):
        ## loop through the datasets concats and subsets to find each ImageFolder type dataset and prepend its root
        for one in self.datasets[0].datasets+self.datasets[1:]:
            for two in one.datasets:
                two.dataset.root = os.path.join(to_prepend,two.dataset.root.replace('./data/',''))
                for idx in range(len(two.dataset.samples)): ## loop trough each sample and edit its path
                    two.dataset.samples[idx] = (
                        os.path.join(to_prepend,two.dataset.samples[idx][0].replace('./data/','')),
                        two.dataset.samples[idx][1]
                    ) 

    def build_type1_combination(self,group,test,filler):
        total = 3168
        counts = [int(0.97*total),int(0.87*total)]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[0],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[1],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[2],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[3],counts[1])],
            ## filler
            ("bulldog","dachshund","labrador","corgi"):[(filler,total-counts[0]),(filler,total-counts[1])],
        }
        ## TEST
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[0]],
            ("dachshund",):[test[1], test[1]],
            ("labrador",):[test[2], test[2]],
            ("corgi",):[test[3], test[3]],
        }
        return combinations

    def build_type2_combination(self,group,test):
        total = 3168
        counts = [total,total]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[1],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[0],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[3],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[2],counts[1])],
        }
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[1]],
            ("dachshund",):[test[1], test[0]],
            ("labrador",):[test[2], test[3]],
            ("corgi",):[test[3], test[2]],
        }
        return combinations


## Spawrious classes for each Spawrious dataset 
class SpuriousLocationType1_1(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ["desert","jungle","dirt","snow"]
        test = ["dirt","snow","desert","jungle"]
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams["data_augmentation"], type1=True)


class SpuriousLocationType1_2(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ['mountain', 'beach', 'dirt', 'jungle']
        test = ['jungle', 'dirt', 'beach', 'snow']
        filler = "desert"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams["data_augmentation"], type1=True)


class SpuriousLocationType1_3(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ['jungle', 'mountain', 'snow', 'desert']
        test = ['mountain', 'snow', 'desert', 'jungle']
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams["data_augmentation"], type1=True)


class SpuriousLocationType2_1(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ["dirt","jungle","snow","beach"]
        test = ["snow","beach","dirt","jungle"]
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams["data_augmentation"])

class SpuriousLocationType2_2(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ['desert', 'mountain', 'dirt', 'jungle']
        test = ['dirt', 'jungle', 'mountain', 'desert']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams["data_augmentation"]) 


class SpuriousLocationType2_3(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ['beach', 'snow', 'mountain', 'desert']
        test = ['desert', 'mountain', 'beach', 'snow']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams["data_augmentation"])


## List of functions that load pytorch dataset files to save the time from building the dataset everytime (part of the dataset files downloaded)
def SpawriousO2O_easy(root_dir):
    filename = "sc11.pth"
    path = os.path.join(root_dir,filename)
    dataset = torch.load(path)
    dataset.prepend_path(root_dir)
    return dataset

def SpawriousO2O_medium(root_dir):
    filename = "sc12.pth"
    path = os.path.join(root_dir,filename)
    dataset = torch.load(path)
    dataset.prepend_path(root_dir)
    return dataset

def SpawriousO2O_hard(root_dir):
    filename = "sc13.pth"
    path = os.path.join(root_dir,filename)
    dataset = torch.load(path)
    dataset.prepend_path(root_dir)
    return dataset

def SpawriousM2M_hard(root_dir):
    filename = "sc21.pth"
    path = os.path.join(root_dir,filename)
    dataset = torch.load(path)
    dataset.prepend_path(root_dir)
    return dataset

def SpawriousM2M_easy(root_dir):
    filename = "sc22.pth"
    path = os.path.join(root_dir,filename)
    dataset = torch.load(path)
    dataset.prepend_path(root_dir)
    return dataset

def SpawriousM2M_medium(root_dir):
    filename = "sc23.pth"
    path = os.path.join(root_dir,filename)
    dataset = torch.load(path)
    dataset.prepend_path(root_dir)
    return dataset



