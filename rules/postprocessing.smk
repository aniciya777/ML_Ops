rule validation:
    input:
        "../data/comparison_of_revisions.txt",
        marker="../data/.pre_validation_done"
    output:
        "../comparison_versions.md"
    script:
        "../src/validation/validation.py"


rule pre_validation:
    input:
        "../data/models.dvc"
    output:
        marker="../data/.pre_validation_done"
    shell:
        'dvc push \n'
        'git commit -a -m "Save new models" \n'
        'git log -1 --format="%H" >> ../data/comparison_of_revisions.txt \n'
        'touch {output.marker}'
