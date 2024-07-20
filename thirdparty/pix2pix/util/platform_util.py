# This code is a simple abstraction around GPU support in pytorch
#
# The author has made slight modifications to Pix2Pix codebase,
#   because the original Pix2Pix code has hardcoded CUDA/CPU usage,
#   so author created these functions to support M1(Apple Silicon) in addition to CUDA/CPU.
#   - Allow executing with devices without a supported GPU (Ex: Machine with non-supported GPU)
#   - Allow executing on M1 devices using MPS (Ex: Mac Mini)
#   - Allow running on CUDA supported devices (Ex: Paperspace rented machines)
#
import enum
import platform
from typing import List

import torch

# Detect if the system is running Apple Silicon (M1, M2, etc)
#   Ref: https://stackoverflow.com/a/77381029
APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"
# Detect if Apple Silicon with Pytorch MPS support is available
#   Ref: https://pytorch.org/docs/stable/notes/mps.html
MPS_PRESENT = APPLE_SILICON \
              and hasattr(torch.backends, 'mps') \
              and torch.backends.mps.is_available() \
              and torch.backends.mps.is_built()
# Detect if CUDA is available and is not an Apple Silicon device
CUDA_PRESENT = not APPLE_SILICON and hasattr(torch.backends, 'cuda') and hasattr(torch, 'cuda') \
               and torch.cuda.is_available() \
               and torch.backends.cuda.is_built()


# Enum of GPU types
class GpuPlatform(enum.Enum):
    CUDA = 1
    MPS = 2
    CPU = 3


# At the beginning, platform will be set to the most suitable one
# However, based on gpu_ids argument to below cross-gpu functions, it will be set to CPU
#    (this is to handle disabling GPU, see base_options)

PLATFORM = GpuPlatform.CPU

if CUDA_PRESENT:
    PLATFORM = GpuPlatform.CUDA
elif MPS_PRESENT:
    PLATFORM = GpuPlatform.MPS


# The code below is based on the changes needed to support MPS backend. It is wrapped with these functions,
#      to make it easy to switch between CPU, CUDA and MPS based on the executed device.

def device(gpu_ids: List) -> torch.device:
    """
    Create a torch device based on provided gpu_ids
    :param gpu_ids: a list of gpu ids, if empty CPU device is built
    :return: a torch device object
    """
    global PLATFORM
    # Fallback to CPU
    if not gpu_ids:
        PLATFORM = GpuPlatform.CPU
        return torch.device("cpu")

    if PLATFORM == GpuPlatform.CUDA:
        return torch.device(f"cuda:{gpu_ids[0]}")
    if PLATFORM == GpuPlatform.MPS:
        return torch.device("mps")
    return torch.device("cpu")


def model_to_gpu(net, gpu_ids: List):
    """
    Move given network to GPU
    :param net: network
    :param gpu_ids: list of gpu ids, if empty no need to move
    :return: network
    """
    global PLATFORM
    # Fallback to CPU
    if not gpu_ids:
        PLATFORM = GpuPlatform.CPU
        return net

    if PLATFORM == GpuPlatform.CUDA:
        net.to(gpu_ids[0])
    elif PLATFORM == GpuPlatform.MPS:
        net.to('mps')
    return net


def setup_device(gpu_ids: List):
    """
    Setup device at the beginning of the app execution
      This only set CUDA device, MPS works without setting it.
    :param gpu_ids: List of gpu_ids, if empty assume device is CPU
    """
    print("----------- Setting up torch device. --------")
    print("user provided gpu_ids =", repr(gpu_ids))
    global PLATFORM
    if not gpu_ids:
        # Fallback to CPU
        PLATFORM = GpuPlatform.CPU
    elif PLATFORM == GpuPlatform.CUDA:
        torch.cuda.set_device(gpu_ids[0])
    print("platform =", repr(PLATFORM))
    print("---------------------------------------------")
