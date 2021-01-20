from os.path import abspath, dirname

from .data import WyckoffData, collate_batch
from .descriptor_network import DescriptorNetwork
from .model import Wren

ROOT = dirname(dirname(abspath(__file__)))


def bold(text):
    """Turn text string bold when printed.
    https://stackoverflow.com/q/8924173/4034025
    """
    return f"\033[1m{text}\033[0m"
