"""Python wrapper code for semsimian."""
from pathlib import Path

import semsimian


def main():
    """Python code for semsimian."""
    proj_dir = Path(__file__).resolve().parents[2]
    test_data_dir = proj_dir / "tests/data/"
    semsimian.run(str(test_data_dir / "test_set.tsv"), str(test_data_dir / "closures.tsv"))


if __name__ == "__main__":
    main()
