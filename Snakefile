import os

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


rule all:
    input:
        "comparison_versions.md"
    shell:
        'git add {input[0]} \n'
        'git commit -a -m "Update {input[0]}"'


rule validation:
    input:
        "data/comparison_of_revisions.txt",
        "data/spec_ds/label_names.npy",
        "data/spec_ds/test_specs/dataset_spec.pb",
        "data/spec_ds/test_specs/snapshot.metadata",
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
        "data/spec_ds/label_names.npy",
        "data/spec_ds/test_specs/dataset_spec.pb",
        "data/spec_ds/test_specs/snapshot.metadata",
        "data/spec_ds/train_specs/dataset_spec.pb",
        "data/spec_ds/train_specs/snapshot.metadata"
    output:
        "data/spec_ds.dvc"
    shell:
        "yes | dvc commit \n"
        "dvc push"


rule make_spectrogram:
    input:
        "data/output.dvc"
    output:
        "data/spec_ds/label_names.npy",
        "data/spec_ds/test_specs/dataset_spec.pb",
        "data/spec_ds/test_specs/snapshot.metadata",
        "data/spec_ds/train_specs/dataset_spec.pb",
        "data/spec_ds/train_specs/snapshot.metadata"
    shell:
        "uv run make_spec"


rule dvc_commit_and_push_wav:
    input:
        *(
            f"data/output/ML/{lbl}/*.wav"
            for lbl in LABELS
        ),
        marker="data/.preprocessing_done"
    output:
        "data/output.dvc"
    shell:
        "yes | dvc commit \n"
        "dvc push"
