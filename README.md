# Accurate Differential Operators for Hybrid Neural Fields

This repository contains the associated code for the paper titled

>[Accurate Differential Operators for Hybrid Neural Fields](https://arxiv.org/abs/2312.05984). *[Aditya Chetan](https://justachetan.github.io), [Guandao Yang](https://www.guandaoyang.com/), [Zichen Wang](https://zichenwang01.github.io/), [Steve Marschner](https://www.cs.cornell.edu/~srm/), [Bharath Hariharan](https://www.cs.cornell.edu/~bharathh/)*.


## Updates

- **[2023/12/22]** Initial code release.
- **[2023/12/10]** Code release coming soon!

## Setup

For setting up the environments required for training the models and running the rendering demo, please follow the steps given in [`setup`](setup/README.md).

## Experiments

### Training

For training your own models: 

0. First activate the conda environment for training using:
```bash
conda activate hnf-train
```
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
5. For fine-tuning, follow the same commands as training, except that you need to specify the path to the checkpoint you want to fine-tune from:
```bash 
python3 train.py configs/<your_config>.yaml --resume --pretrained <path_to_checkpoint>
```

### Rendering

In order to view rendering results:


0. First activate the conda environment for rendering using:
```bash
conda activate hnf-render
```
1. Now open the notebook `rendering.ipynb` and set the kernel to `hnf-render`.
2. Select the shape you want in the dropdown and run the cells in order. 
3. Feel free to add your own shapes by training models as described above and adding settings for the shape in the `settings_dict` variable in the notebook.



## Citation

If you found the code in this repository useful, please consider citing our paper:

```
@misc{chetan2023accurate,
      title={Accurate Differential Operators for Hybrid Neural Fields}, 
      author={Aditya Chetan and Guandao Yang and Zichen Wang and Steve Marschner and Bharath Hariharan},
      year={2023},
      eprint={2312.05984},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

We thank the authors of [torch-ngp](https://github.com/ashawkey/torch-ngp), [ldif](https://github.com/google/ldif), [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for making their code publicly available. 
