import sys
from pathlib import Path

from preparation.utils import transport_one_file


def main() -> None:
    for s in snakemake.input.files:
        try:
            inp = Path(s)
            out = Path(s.replace("data/input/ML", "data/output/ML"))
            out.parent.mkdir(parents=True, exist_ok=True)
            transport_one_file(inp, out)
        except Exception as e:
            print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
