<p align="center">
 <img src=".github/aviary.svg" alt="Aviary" height=175>
</p>

<h1 align="center">Aviary</h1>

<h4 align="center">

[![Tests](https://github.com/janosh/aviary/workflows/Tests/badge.svg)](https://github.com/janosh/aviary/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/aviary/master.svg)](https://results.pre-commit.ci/latest/github/janosh/aviary/master)
[![License](https://img.shields.io/github/license/janosh/aviary?label=License)](/license)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/janosh/aviary?label=Repo+Size)](https://github.com/janosh/aviary/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/janosh/aviary?label=Last+Commit)](https://github.com/janosh/aviary/commits)

</h4>

> Forked from [@CompRhys/roost](https://github.com/CompRhys/roost).

**R**epresentati**o**n Learning fr**o**m **St**oichiometry

## Premise

In materials discovery applications often we know the composition of trial materials but have little knowledge about the structure.

Many current SOTA results within the field of machine learning for materials discovery are reliant on knowledge of the structure of the material. This means that such models can only be applied to systems that have undergone structural characterization. As structural characterization is a time-consuming process whether done experimentally or via the use of ab-initio methods the use of structures as our model inputs is a prohibitive bottleneck to many materials screening applications we would like to pursue.

One approach for avoiding the structure bottleneck is to develop models that learn from the stoichiometry alone. In this work, we show that via a novel recasting of how we view the stoichiometry of a material we can leverage a message-passing neural network to learn materials properties whilst remaining agnostic to the structure. The proposed model exhibits increased sample efficiency compared to more widely used descriptor-based approaches. This work draws inspiration from recent progress in using graph-based methods for the study of small molecules and crystalline materials.

## Environment Setup

To use `aviary` you need to create an environment with the correct dependencies. Using `Anaconda` this can be accomplished with the follow commands:

```bash
conda create -n aviary python
conda activate aviary
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch scikit-learn tqdm pandas
```

`${CUDA}` Should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your system CUDA version. (You can find your CUDA version via `nvidia-smi`)

You may encounter issues getting the correct installation of either `PyTorch` or `PyTorch_Scatter` for your system requirements if so please check the following pages [PyTorch](https://pytorch.org/get-started/locally/), [PyTorch-Scatter](https://github.com/rusty1s/pytorch_scatter)

## Aviary Setup

Once you have setup an environment with the correct dependencies you can install `aviary` using the following commands:

```bash
conda activate aviary
git clone https://github.com/CompRhys/aviary
cd aviary
python setup.py sdist
pip install -e .
```

This will install the library in an editable state allowing for advanced users to make changes as desired.

## Example Use

In order to test your installation you can do so by running the following example from the top of your `aviary` directory:

```sh
cd /path/to/aviary/
python examples/aviary-example.py --train --evaluate --epochs 10
```

This command runs a default task for 10 epochs -- experimental band gap regression using the data from Zhou et al. (See `data/` folder for reference). This default task has been set up to work out of the box without any changes and to give a flavour of how the model can be used.

If you want to use your own data set on a regression task this can be done with:

```sh
python examples/aviary-example.py --data-path /path/to/your/data/data.csv --train
```

You can then test your model with:

```sh
python examples/aviary-example.py --test-path /path/to/testset.csv --evaluate
```

The model takes input in the form csv files with materials-ids, composition strings and target values as the columns.

| material-id | composition | target |
| ----------- | ----------- | ------ |
| foo-1       | Fe2O3       | 2.3    |
| foo-2       | La2CuO4     | 4.3    |

Basic hints about more advanced use of the model (i.e. classification, robust losses, ensembles, tensorboard logging etc..)
are available via the command:

```sh
python examples/aviary-example.py --help
```

This will output the various command-line flags that can be used to control the code.

## Cite This Work

If you use this code please cite our work for which this model was built:

[Predicting materials properties without crystal structure: Deep representation learning from stoichiometry](https://arxiv.org/abs/1910.00617)

```tex
@article{goodall2019predicting,
  title={Predicting materials properties without crystal structure: Deep representation learning from stoichiometry},
  author={Goodall, Rhys EA and Lee, Alpha A},
  journal={arXiv preprint arXiv:1910.00617},
  year={2019}
}
```

## Work Using Aviary

A critical examination of compound stability predictions from machine-learned formation energies [[Paper]](https://www.nature.com/articles/s41524-020-00362-y) [[arXiv]](https://arxiv.org/abs/2001.10591)

If you have used Aviary in your work please contact me and I will add your paper here.

## Acknowledgements

The open-source implementation of `cgcnn` available [here](https://github.com/txie-93/cgcnn) provided significant initial inspiration for how to structure this code-base.

## Disclaimer

This is research code shared without support or any guarantee on its quality. However, please do raise an issue or submit a pull request if you spot something wrong or that could be improved and I will try my best to solve it.

## To Do

Parallelize the code over multiple GPUs. Currently `DataParallel` crashes as subsets of the batch have different sizes due to the use of lists of lists rather than zero-padding.

```py
if (n_gpus := torch.cuda.device_count() > 1) and (device in ["cuda", torch.device("cuda")]):
    print(f"Running on {n_gpus} GPUs")
    model = nn.DataParallel(model)
```
