"""
Generating loom files using scvelo

11/18/2025
Marissa Esteban
"""

import scvelo as scv
from scvelo.preprocessing import velocyto as vcy


bam = "possorted_genome_bam.bam"
gtf = "genes.gtf"
out = "sample.loom"

vcy.convert_bam_to_loom(bam, gtf, output_file=out)
