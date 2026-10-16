"""Verify CUDA logical device maps to the exact idle GPU UUID selected by NVML.

Torch 2.4 lacks a device-properties uuid attribute, so use its CUDA runtime's
stable cudaDeviceGetPCIBusId API, then resolve that PCI device's NVIDIA UUID.
This check runs only after the idle GPU guard, before any trainer is started.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('gpu_json', type=Path)
    args = p.parse_args()
    selected = json.loads(args.gpu_json.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1,3'
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
    assert selected['physical'] in [1, 3] and not selected['occupied']
    libs = list(Path(sys.prefix).glob('lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12'))
    if not libs:
        raise RuntimeError('Cannot locate the existing CUDA runtime')
    runtime = ctypes.CDLL(str(libs[0]))
    function = runtime.cudaDeviceGetPCIBusId
    function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    function.restype = ctypes.c_int
    bus = ctypes.create_string_buffer(64)
    code = function(bus, len(bus), selected['logical'])
    if code:
        raise RuntimeError(f'CUDA PCI bus query failed: {code}')
    pci_bus = bus.value.decode()
    uuid = subprocess.check_output(['nvidia-smi', '-i', pci_bus, '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    if uuid != selected['uuid']:
        raise RuntimeError(f'CUDA mapping mismatch: expected {selected["uuid"]}, got {uuid}')
    print(json.dumps(dict(**selected, runtime_pci_bus=pci_bus, verified_uuid=uuid), indent=2))


if __name__ == '__main__':
    main()
