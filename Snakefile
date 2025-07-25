import glob
import itertools
import shlex

from dotenv import load_dotenv
from snakemake.io import touch

from train.config import Config


load_dotenv()

BATCH_SIZES = [16, 32, 64]


def models_for(batch_size):
    return [
        f"data/models/batch_{batch_size}/model{i + 1}.keras"
        for i in range(Config.NUM_FOLDS)
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
for label, ext in itertools.product(LABELS,SUPPORTED_EXTENSIONS):
    input_files += glob.glob(f"data/input/ML/{label}/*.{ext}")
    input_files += glob.glob(f"data/input/ML/Whatsapp/{label}/*.{ext}")


rule all:
    input:
        "comparison_versions.md"
    shell:
        'git add {input[0]} \n'
        'git add "static/plots/*" \n'
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
        "rm static/plots/*.png || true \n"
        f"uv run validate -n {len(BATCH_SIZES)}"


rule dvc_commit_and_push_models:
    input:
        expand("data/.train_done_{bs}", bs=BATCH_SIZES),
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
        dir=directory("data/models/batch{bs}"),
        marker=touch("data/.train_done_{bs}")
    params:
        batch_size=lambda wildcards: wildcards.bs
    wildcard_constraints:
        batch_size="|".join(str(bs) for bs in BATCH_SIZES)
    shell:
        "mkdir -p {output.dir} \n"
        "uv run train --batch_size {wildcards.bs} --output-dir \"{output.dir}\""


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
