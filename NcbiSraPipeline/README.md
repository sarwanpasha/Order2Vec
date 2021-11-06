# NCBI SRA data analysis pipeline

Running this pipeline requires `snakemake`
(https://snakemake.readthedocs.io/en/stable/) as well as:

* a few other standard bioinformatics tools including `samtools`,
`bwa`, `bcftools`, `tabix` (for indexing VCF files),

* the `python3` interpreter, as well as

* some standard tools, *e.g.*, `awk`, `grep`, etc., that can be found
on most linux/unix systems.
