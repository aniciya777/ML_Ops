#!/bin/sh

echo "Run the download from DVC ..."
uv run dvc pull -q data/input
echo "DVC loading is completed"
uv run snakemake --cores 1 -f
sleep infinity
