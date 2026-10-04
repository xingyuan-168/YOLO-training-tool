"""Owned helper processes terminate on cancellation or timeout."""

import subprocess
import sys
import time

import pytest

from yolo_workbench.runtime import run_process
from yolo_workbench.storage import OperationCancelled


def test_helper_collects_both_streams():
    result = run_process(
        [sys.executable, "-c", "import sys; print('output'); print('diagnostic',file=sys.stderr)"],
        timeout=10,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "output"
    assert result.stderr.strip() == "diagnostic"


@pytest.mark.parametrize("mode", ["cancel", "timeout"])
def test_helper_interrupts_running_owned_process(tmp_path, mode):
    import psutil

    pid_file = tmp_path / "pid.txt"
    script = "import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(30)"
    started = time.monotonic()
    kwargs = {"timeout": 1} if mode == "timeout" else {"cancel": lambda: pid_file.exists()}
    error = subprocess.TimeoutExpired if mode == "timeout" else OperationCancelled
    with pytest.raises(error):
        run_process([sys.executable, "-c", script, str(pid_file)], **kwargs)
    assert time.monotonic() - started < 10
    assert pid_file.exists()
    assert not psutil.pid_exists(int(pid_file.read_text()))
