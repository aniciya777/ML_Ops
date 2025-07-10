import glob
import itertools
import shlex

from dotenv import load_dotenv
from snakemake.io import touch


load_dotenv()

BATCH_SIZES = [16, 32, 64]

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
    input_files += glob.glob(f"data/input/ML/Whatsapp/{label}/*.{ext}")


rule all:
    input:
        "comparison_versions.md"
    shell:
        'git add {input[0]} \n'
        'git commit -a -m "Update {input[0]}"'


rule validation:
    input:
        "data/spec_ds/label_names.npy",
        "data/spec_ds/test_specs/dataset_spec.pb",
        "data/spec_ds/test_specs/snapshot.metadata",
        marker="data/.models_pushed_done"
    output:
        "comparison_versions.md"
    shell:
        f"uv run validate -n 3"


rule dvc_commit_and_push_models:
    input:
        models
    output:
        "data/models.dvc",
        marker=touch("data/.models_pushed_done")
    shell:
        "dvc add data/models \n"
        "yes | dvc commit || true\n"
        "dvc push"


rule train:
    input:
        "data/spec_ds.dvc"
    output:
        models
    shell:
        "uv run train"


rule dvc_commit_and_push_spec:
    input:
        "data/spec_ds/label_names.npy",
        "data/spec_ds/test_specs/dataset_spec.pb",
        "data/spec_ds/test_specs/snapshot.metadata",
        "data/spec_ds/train_specs/dataset_spec.pb",
        "data/spec_ds/train_specs/snapshot.metadata",
        marker="data/.make_spec_done"
    output:
        "data/spec_ds.dvc"
    shell:
        "dvc add data/spec_ds \n"
        "yes | dvc commit || true \n"
        "dvc push"


rule make_spectrogram:
    input:
        "data/output.dvc",
        marker="data/.output_pushed_done"
    output:
        "data/spec_ds/label_names.npy",
        "data/spec_ds/test_specs/dataset_spec.pb",
        "data/spec_ds/test_specs/snapshot.metadata",
        "data/spec_ds/train_specs/dataset_spec.pb",
        "data/spec_ds/train_specs/snapshot.metadata",
        marker=touch("data/.make_spec_done")
    shell:
        "uv run make_spec"


rule dvc_commit_and_push_output:
    input:
        marker="data/.preprocessing_done"
    output:
        "data/output.dvc",
        marker=touch("data/.output_pushed_done")
    shell:
        """
        mkdir -p data/models
        mkdir -p data/spec_ds
        dvc add data/output
        yes | dvc commit || true
        dvc push
        """


rule preprocessing:
    input:
        files=input_files
    output:
        marker=touch("data/.preprocessing_done")
    params:
        files=lambda wildcards, input: " ".join(shlex.quote(f) for f in input.files)
    shell:
        r"""
        for f in {params.files}; do
            echo "$f"
        done | uv run src/preparation/prepare_cli.py
        """
