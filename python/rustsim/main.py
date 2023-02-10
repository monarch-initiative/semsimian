# Python wrapper code for rustsim.
import rustsim
from pathlib import Path

proj_dir = Path(__file__).resolve().parents[3]
test_data_dir = proj_dir/"tests/data/"
# This doesn't work for now.
rustsim.run(test_data_dir/"test_set.tsv", test_data_dir/"closures.tsv")