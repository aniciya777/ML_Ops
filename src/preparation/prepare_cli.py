import os.path
import sys
from pathlib import Path

from utils import transport_one_file  # type: ignore


def main() -> None:
    for s in map(str.strip, sys.stdin):
        try:
            inp = Path(s)
            assert os.path.exists(inp), f'{inp} does not exist'
            out = Path(
                s
                .replace('/Whatsapp', '')
                .replace("data/input/ML", "data/output/ML")
                + '.wav'
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            transport_one_file(inp, out)
        except Exception as e:
            print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
