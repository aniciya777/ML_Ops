import os

from dotenv import load_dotenv


load_dotenv()

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
    'mp3',
    'opus',
    'ogg'
]

rule all:
    input:
        "comparison_versions.md"
    shell:
        'git add {input[0]} \n'
        'git commit -a -m "Update {input[0]}"'


rule validation:
    input:
        "data/сomparison_of_revisions.txt"
    output:
        "comparison_versions.md"
    script:
        "src/validation/validation.py"


rule pre_validation:
    input:
        "data/models.dvc"
    output:
        "data/сomparison_of_revisions.txt"
    shell:
        'dvc pull \n'
        'git -a -m "Save new models" \n'
        'git log -1 --format="%H" >> {output[0]}'
