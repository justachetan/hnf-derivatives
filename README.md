# Accurate Differential Operators for Hybrid Neural Fields

This repository contains the associated code for the paper titled

>[Accurate Differential Operators for Hybrid Neural Fields](https://arxiv.org/abs/2312.05984). *[Aditya Chetan](https://justachetan.github.io), [Guandao Yang](https://www.guandaoyang.com/), [Zichen Wang](https://zichenwang01.github.io/), [Steve Marschner](https://www.cs.cornell.edu/~srm/), [Bharath Hariharan](https://www.cs.cornell.edu/~bharathh/)*.

## Setup

For setting up the environments required for training the models and running the rendering experiments, use the following commands:
```bash
cd setup
bash setup.sh
```

## Experiments

### Training

For training your own models: 
1. First place your mesh that is normalized such that it lies within the $[-1, 1]^3$ hypercube in the `data` folder. 
2. Then, create a config using one of the examples shared in the `configs` folder. In most cases, it should be as simple as replacing the path to the mesh with your own.
3. Then, run the following command:
```bash
python3 train.py configs/<your_config>.yaml
```
4. If you want to make any changes to any other hyperparameters from the command line, here is an example of how to do it, shown using the learning rate:
```bash
python3 train.py configs/<your_config>.yaml --hparams trainer.opt.lr=0.001
```


## Updates

- **[2023/12/10]** Code release coming soon!
