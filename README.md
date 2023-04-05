# Spawrious

Spawrious is a OOD classification benchmark in the computer vision domain ([link to paper](https://arxiv.org/abs/2303.05470)). It consists of two separate OOD challenges: one-to-one and many-to-many spurious correlations. 

The dataset contains images of 4 dog breeds, found in 6 locations. The entire dataset consists of ~152,000 images, but each challenge only requires a subset of this. As a result, the repo allows users to only download the mimimal dataset required for a given spawrious challenge. 

## Datasets
- [entire_dataset](https://www.dropbox.com/s/e40j553480h3f3s/spawrious224.tar.gz?dl=1)
- [one-to-one easy](https://www.dropbox.com/s/kwhiv60ihxe3owy/spawrious__o2o_easy.tar.gz?dl=1)
- [one-to-one medium](https://www.dropbox.com/s/x03gkhdwar5kht4/spawrious224__o2o_medium.tar.gz?dl=1)
- [one-to-one hard](https://www.dropbox.com/s/p1ry121m2gjj158/spawrious__o2o_hard.tar.gz?dl=1)
- [many-to-many (all)](https://www.dropbox.com/s/5usem63nfub266y/spawrious__m2m.tar.gz?dl=1)

## Downloading the dataset and running an experiment

Datasets take the following names: `entire_dataset`; `o2o_easy`; `o2o_medium`; `o2o_hard`; `m2m`. Running the command below for the first time downloads the appropriate dataset at a usee specified user directory. 

```
python example.py --root_dir <path to data dir> --dataset_name <one of the list above>
```


## Licensing

This project is licensed under the terms of the MIT license.
