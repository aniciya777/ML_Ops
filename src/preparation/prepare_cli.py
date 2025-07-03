import sys
from pathlib import Path

from utils import transport_one_file  # type: ignore


def main() -> None:
    for s in sys.stdin:
        try:
            inp, out = map(Path, s.strip().split(';'))
            out.parent.mkdir(parents=True, exist_ok=True)
            transport_one_file(inp, out)
        except Exception as e:
            print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
