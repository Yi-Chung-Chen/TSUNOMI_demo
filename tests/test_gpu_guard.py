import os
import pytest
from scripts.wait_for_idle_gpu import available_gpus


def test_existing_compute_process_prevents_idle_selection(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,3')
    def fake_query(command, text):
        assert command[0] == 'nvidia-smi' and command[1:3] == ['-i', '1,3']
        if '--query-compute-apps=gpu_uuid,pid' in command:
            return 'GPU-one, 12345\n'
        return '1, GPU-one, 0, 100\n3, GPU-three, 0, 0\n'
    monkeypatch.setattr('scripts.wait_for_idle_gpu.subprocess.check_output', fake_query)
    states, free = available_gpus()
    assert len(states) == 2
    assert [(s['physical'], s['logical']) for s in free] == [(3, 1)]


def test_wrong_visibility_rejected(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,2')
    with pytest.raises(RuntimeError, match='exactly 1,3'):
        available_gpus()
