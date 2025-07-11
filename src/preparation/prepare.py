import subprocess
import sys


def main():
    args = ["snakemake", "-f", "-c1", "preprocessing"] + sys.argv[1:]
    return subprocess.call(args)


if __name__ == '__main__':
    sys.exit(main())
