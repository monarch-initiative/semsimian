# Python wrapper code for rustsim.
import rustsim
from pathlib import Path


def main():
    proj_dir = Path(__file__).resolve().parents[2]
    test_data_dir = proj_dir / "tests/data/"
    rustsim.run(str(test_data_dir / "test_set.tsv"), str(test_data_dir / "closures.tsv"))


if __name__ == "__main__":
    main()
