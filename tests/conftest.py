from os.path import relpath
from shutil import rmtree

import pytest
import torch

from roost.utils import make_model_dir


@pytest.fixture(autouse=True)
def rm_test_dirs():

    # ensure reproducible results, applies to all tests
    torch.manual_seed(0)

    yield rm_test_dirs  # provide the fixture value

    model_dir = make_model_dir("tests", run_id="")
    print(f"teardown: removing test model directory '{relpath(model_dir)}'")

    rmtree(model_dir)
