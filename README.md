# DFT_code
Amplifying Pathological Signals in cfDNA End-Motif Profiles via Discrete Fourier Transform

## Description


## Overview
![Pipeline Diagram](assets/workflow.svg)


## Project Structure


## Installation
We recommend using Conda to manage the environment; while the pipeline is fully CPU-compatible for accessibility, a CUDA-enabled PyTorch setup is recommended for significantly faster decomposition.

```bash
# Clone the repository
git clone https://github.com/Upupdownn/BED_code.git
cd BED_code

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate bed_analysis
```

## Preparation

The BED pipeline supports two input formats for starting the analysis. You can either provide raw alignment files (BAM) or pre-processed fragment files (TSV).

**1. Input Fragment Data**

* **Option A: BAM Files**

  * Standard genomic alignment files are supported.

  * **Requirements:** Files must be sorted and indexed (e.g., sample.bam and sample.bam.bai).

* **Option B: Fragment TSV Files**

  * If you have already extracted fragment information, provide a TSV file with a header and the following columns: chr, start, end, mapq, and strand.

  * **Example Data:** For testing purposes, sample fragment TSV files are provided in the `examples/frag_file/` directory.

**2. Reference Genome (2bit format)**

A `.2bit` file of the reference genome is required for sequence extraction and end-motif frequency calculation. Please choose the version (e.g., hg19 or hg38) that matches your alignment.

To run the provided example dataset, you can download the hg19 reference genome using the following command:

```Bash
# Download hg19 reference genome from UCSC
wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.2bit
```

**Note:** Ensure the path to the .2bit file is correctly passed to the --tb_file argument in the workflow script.