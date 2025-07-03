import glob
import os
import itertools

from dotenv import load_dotenv
from snakemake.io import touch


load_dotenv()

models = [
    f"data/models/model{i + 1}.keras"
    for i in range(5)
]
LABELS = [
    "Catch",
    "Gun",
    "Index",
    "Like",
    "Relax",
    "Rock"
]
SUPPORTED_EXTENSIONS = [
    'wav',
    'ogg',
    'opus',
    'mp3',
]
input_files = []
for label, ext in itertools.product(LABELS, SUPPORTED_EXTENSIONS):
    input_files += glob.glob(f"data/input/ML/{label}/*.{ext}")
    input_files += glob.glob(f"data/input/ML/Watsapp/{label}/*.{ext}")


rule all:
    input:
        "comparison_versions.md"
    shell:
        'git add {input[0]} \n'
        'git commit -a -m "Update {input[0]}"'


rule validation:
    input:
        "data/comparison_of_revisions.txt",
        "data/.make_spec_done",
        marker="data/.pre_validation_done"
    output:
        "comparison_versions.md"
    script:
        "src/validation/validation.py"


rule pre_validation:
    input:
        "data/models.dvc"
    output:
        marker=touch("data/.pre_validation_done")
    shell:
        'dvc push \n'
        'git commit -a -m "Save new models" \n'
        'git log -1 --format="%H" >> data/comparison_of_revisions.txt'


rule dvc_commit_models:
    input:
        models
    output:
        "data/models.dvc"
    shell:
        "yes | dvc commit"


rule train:
    input:
        "data/spec_ds.dvc"
    output:
        models
    shell:
        "uv run train"


rule dvc_commit_and_push_spec:
    input:
        marker="data/.make_spec_done"
    output:
        "data/spec_ds.dvc"
    shell:
        "yes | dvc commit \n"
        "dvc push"


rule make_spectrogram:
    input:
        "data/output.dvc"
    output:
        marker=touch("data/.make_spec_done")
    shell:
        "uv run make_spec"


rule dvc_commit_and_push_wav:
    input:
        marker="data/.preprocessing_done"
    output:
        "data/output.dvc"
    shell:
        "yes | dvc commit \n"
        "dvc push"


rule preprocessing:
    input:
        files=input_files
    output:
        marker=touch("data/.preprocessing_done")
    script:
        "src/preparation/prepare_snakemake.py"
