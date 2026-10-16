"""Check SSH command quoting locally without connecting to a remote machine."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_remote_launcher_quotes_prompt_and_paths(tmp_path):
    launcher = Path(__file__).parents[1] / "scripts" / "venturi-4090d"
    tools = tmp_path / "tools"
    tools.mkdir()
    ssh = tools / "ssh"
    ssh.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    ssh.chmod(0o755)
    directory = tmp_path / "remote space ' quoted"
    directory.mkdir()
    python = tools / "fake python"
    python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'path': os.environ['PYTHONPATH']}))\n"
    )
    python.chmod(0o755)
    prompt = "don't expand $(touch UNWANTED); \"quotes\"\nsecond line"
    env = {
        **os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
        "VENTURI_REMOTE_DIR": str(directory), "VENTURI_REMOTE_PYTHON": str(python),
        "VENTURI_REMOTE_MODEL": "/model with spaces", "VENTURI_REMOTE_SIPHON": "/siphon",
        "VENTURI_SSH_HOST": "test-host",
    }
    result = subprocess.run(
        ["bash", str(launcher), "generate", "--prompt", prompt], env=env,
        check=True, capture_output=True, text=True,
    )
    ssh_args = json.loads(result.stdout)
    assert ssh_args[-2] == "test-host"
    assert "ForwardAgent=no" in ssh_args and "BatchMode=yes" in ssh_args
    # Execute only the captured command against the local fake runtime.
    remote = subprocess.run(
        ["bash", "-c", ssh_args[-1]], check=True, capture_output=True, text=True,
    )
    payload = json.loads(remote.stdout)
    assert payload["args"] == [
        "-m", "venturi", "generate", "/model with spaces", "--prompt", prompt,
    ]
    assert payload["path"] == f"{directory}/src:/siphon"
    assert not (directory / "UNWANTED").exists()
