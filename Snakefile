import os

from dotenv import load_dotenv


load_dotenv()

models = [
    f"data/models/model{i + 1}.keras"
    for i in range(5)
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
        marker="data/.pre_validation_done"
    output:
        "comparison_versions.md"
    script:
        "src/validation/validation.py"


rule pre_validation:
    input:
        "data/models.dvc"
    output:
        marker="data/.pre_validation_done"
    shell:
        'dvc push \n'
        'git commit -a -m "Save new models" \n'
        'git log -1 --format="%H" >> data/comparison_of_revisions.txt \n'
        'touch {output.marker}'


rule dvc_commit_models:
    input:
        models
    output:
        "data/models.dvc"
    shell:
        "dvc commit"


rule train:
    input:
        "data/spec_ds.dvc"
    output:
        models
    shell:
        "uv run train"


