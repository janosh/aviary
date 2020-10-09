# CSD3 Guide

## Submissions Scripts

To submit a CPU or GPU job, use `sbatch hpc/(c|g)pu_submit` after editing those files appropriately. Rather than changing all the parameters directly in those files, you can pass in variables defined directly on the command line into those scripts as follows:

```sh
sbatch --export var1='foo',var2='bar' hpc/(c|g)pu_submit
```

and then using those variables as e.g.

```sh
CMD="echo var1 is '$var1' and var2 is '$var1'"
```

in the submission scripts. To change the job name and run time from the command line, use `sbatch -J job_name -t 1:0:0` (time format `h:m:s`). So in full:

```sh
sbatch -J roost-mnf -t 1:0:0 --export args='-use_mnf -resume -model_name=mnf_roost' hpc/gpu_submit
```

## Environment

To setup dependencies, use `conda`

```sh
conda create -n py38 python
pip install tqdm pandas pymatgen torch_scatter torch numpy scipy scikit_learn
```

## Running Short Experiments

Short interactive sessions are a good way to ensure a long job submitted via `(c|g)pu_submit` will run without errors in the actual HPC environment.

[To request a 10-minute interactive CPU session](https://docs.hpc.cam.ac.uk/hpc/user-guide/interactive.html#sintr):

```sh
sintr -A LEE-SL3-CPU -p skylake -N1 -n1 -t 0:10:0 --qos=INTR
module load rhel7/default-peta4
script job_name.log
```

Useful for testing a job will run successfully in the actual environment it's going to run in without having to queue much.

The last line `script job_name.log` is optional but useful as it ensures everything printed to the terminal during the interactive session will be recorded in `job_name.log`. [See `script` docs](https://man7.org/linux/man-pages/man1/script.1.html).

To use service level 2, include your CRSId, i.e. `LEE-JR769-SL2-CPU` instead of `LEE-SL3-CPU`.

Similarly, for a 10-minute interactive GPU session:

```sh
sintr -A LEE-SL3-GPU -p pascal -N1 -n1 -t 0:10:0 --qos=INTR --gres=gpu:1
module load rhel7/default-gpu
script job_name.log
```

Before doing anything, that requires a GPU, remember to load

```sh
module load rhel7/default-gpu
```

To specify CUDA version:

```sh
module load cuda/11.0
```

Check current version with `nvcc --version`.

Needed for example to install `pytorch-scatter`. Else you get:

> gcc: error: unrecognized command line option ‘-std=c++14’
> error: command 'gcc' failed with exit status 1
> ERROR: Failed building wheel for torch-scatter

To check available hardware:

```sh
nvidia-smi
```

This should print something like

```text
Thu Oct  8 20:15:44 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  On   | 00000000:04:00.0 Off |                    0 |
| N/A   35C    P0    28W / 250W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Debugging Tips

If the interactive window won't launch over SSH, see [vscode-python#12560](https://github.com/microsoft/vscode-python/issues/12560).
