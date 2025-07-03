include: "rules/postprocessing.smk"


rule all:
    input:
        "comparison_versions.md"
    shell:
        'git add {input[0]} \n'
        'git commit -a -m "Update {input[0]}"'
