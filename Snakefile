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
    output:
        touch('.last_run_stamp')
    shell:
        'git add '