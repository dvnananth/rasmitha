import os
import csv
import subprocess
import sys


def test_cli_iv(tmp_path):
    out_csv = tmp_path / "out.csv"
    cmd = [sys.executable, "-m", "pharmasim.cli", "iv1c", "--cl", "1.0", "--v", "10.0", "--dose", "100", "--n", "2", "--every", "12", "--csv", str(out_csv)]
    subprocess.check_call(cmd, cwd=os.getcwd())
    assert out_csv.exists()
    with open(out_csv, newline="") as f:
        rows = list(csv.reader(f))
        assert len(rows) > 2
        assert rows[0][0] == "time_h"
