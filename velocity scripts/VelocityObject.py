"""
Defining and RNA Velocity object. User can specify loom and h5ad path 
and will create a new VelocityObject

Author: Marissa Esteban
Date: December 11th 2025
"""

# Setup
import scanpy as sc
import scvelo as scv
import anndata
import loompy
import mygene
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import os


class VelocityObject:

    def __init__(self, anndata_path, loom_paths):
        """
        Create anndata and ldata objects, including cleaning the names.

        anndata_path (str): Path to .h5ad file
        loom_paths (list): list of paths to all loom files (len == 1 ok)
        """

        # velocity computed on HVGs
        self.use_hvgs = False

        # loading and cleaning ADATA
        self.adata = sc.read_h5ad(anndata_path)
        self.adata.obs.index = [x.split('-')[0] for x in self.adata.obs.index]    # TTTGGAGTCGGTTGTA-1_1


        # loading and cleaning LDATA
        temp_loom_path = os.path.dirname(anndata_path) + "/temp_combined.loom"
        if os.path.exists(temp_loom_path):
            os.remove(temp_loom_path)

        loompy.combine(loom_paths, temp_loom_path)

        self.ldata = sc.read_loom(temp_loom_path)
        self.ldata.obs.index = [x.split(':')[1] for x in self.ldata.obs.index]    # possorted_genome_bam_751VB:AAAGTCCGTAAGGTCGx
        self.ldata.obs.index = [x[:-1] for x in self.ldata.obs.index] 
        if os.path.exists(temp_loom_path):
            os.remove(temp_loom_path)

        # cleaning functions
        self.convertGeneNames()
        self.handleDuplicates()
        self.handleSubset()

        # integrating adata and ldata
        self.integrate()

        self.inspect()

        print("VelocityObject initialized successfully.")

    def convertGeneNames(self):
        """
        Convert loom files from ENSEMBL ID's to gene ids
        """

        mg = mygene.MyGeneInfo()
        gene_index = self.ldata.var.index.to_series()

        # only grab the genes that have ENSEMBL IDs
        is_ensembl = gene_index.str.startswith("ENSMUSG")

        print("\n==== Converting ENSEMBL Genes ====")
        print("Total ldata genes:", gene_index.shape[0])
        print("ENSEMBL-like genes:", is_ensembl.sum())
        print("Already-symbol-like genes:", (~is_ensembl).sum(), '\n')

        # only query the ENSMUSG IDs
        ens_ids = gene_index[is_ensembl].unique().tolist()

        results = mg.querymany(
            ens_ids,
            scopes="ensemblgene",   # or "ensembl.gene" depending on mygene version, but this usually works
            fields="symbol",
            species="mouse",
            as_dataframe=False
        )

        # Build mapping dict ENSMUSG -> symbol
        ens_to_symbol = {r["query"]: r.get("symbol", None) for r in results}

        # QC on mapping
        total_ens = len(ens_ids)
        mapped_ens = sum(v is not None for v in ens_to_symbol.values())
        print("Mapped ENSMUSG → symbol:", mapped_ens)
        print("Unmapped ENSMUSG:", total_ens - mapped_ens)

        # Create a 'symbol' column:
        # default: keep existing name
        self.ldata.var["symbol"] = gene_index.copy()

        # replace ENSMUSG entries with their symbol
        for g in gene_index[is_ensembl]:
            sym = ens_to_symbol.get(g, None)
            if sym is not None:
                self.ldata.var.at[g, "symbol"] = sym
        
        self.ldata.var.index = self.ldata.var["symbol"]

        gene_index = self.ldata.var.index.to_series()
        is_ensembl = gene_index.str.startswith("ENSMUSG")

        print("Total ENSMUSG IDs after mapping: ", is_ensembl.sum())

    def handleDuplicates(self):
        """
        Handle duplicate cells and genes
        """
        print("\n=== Removing duplicates ===")

        # ---- adata ----
        n_cells_adata = self.adata.n_obs
        n_genes_adata = self.adata.n_vars

        cell_dups_adata = self.adata.obs.index.duplicated().sum()
        gene_dups_adata = self.adata.var.index.duplicated().sum()

        print("adata")
        print(f"Cells (original): {n_cells_adata}")
        print(f"Genes (original): {n_genes_adata}")
        print(f"Duplicate cell barcodes: {cell_dups_adata}")
        print(f"Duplicate genes: {gene_dups_adata}")

        # ---- ldata ----
        n_cells_ldata = self.ldata.n_obs
        n_genes_ldata = self.ldata.n_vars

        cell_dups_ldata = self.ldata.obs.index.duplicated().sum()
        gene_dups_ldata = self.ldata.var.index.duplicated().sum()

        print("ldata")
        print(f"Cells (original): {n_cells_ldata}")
        print(f"Genes (original): {n_genes_ldata}")
        print(f"Duplicate cell barcodes: {cell_dups_ldata}")
        print(f"Duplicate genes: {gene_dups_ldata}")

        self.adata = self.adata[~self.adata.obs.index.duplicated(keep='first'), :]
        self.ldata = self.ldata[~self.ldata.obs.index.duplicated(keep='first'), :]


        # Intersect with Seurat genes
        self.ldata.var_names_make_unique()  # This handles any duplicate gene symbol

        print(f"\n adata after dedup: {self.adata.shape}")
        print(f"ldata after dedup: {self.ldata.shape}")

    def handleSubset(self):
        """
        Subset by common CELL BARCODES and GENES between andata and ldata
        """

        # Find common barcodes
        common_barcodes = set(self.adata.obs.index).intersection(set(self.ldata.obs.index))
        print(f"\n=== Subsetting cells by GENES and BARCODES ===")
        print(f"Overlapping Cells: {len(common_barcodes)}")

        # Subset both datasets to only common cells
        # Convert to list and sort for reproducibility
        common_barcodes = sorted(list(common_barcodes))

        self.adata = self.adata[common_barcodes, :]
        self.ldata = self.ldata[common_barcodes, :]

        # Now find common genes between the two datasets
        common_genes = self.adata.var.index.intersection(self.ldata.var.index)
        print("Overlapping genes:", len(common_genes))

        # Subset both datasets to common genes
        self.adata = self.adata[:, common_genes]
        self.ldata = self.ldata[:, common_genes]

        print(f"\nadata shape: {self.adata.shape}")
        print(f"ldata shape: {self.ldata.shape}")

        # Verify cell barcodes match
        print("Cells match:", (self.adata.obs.index == self.ldata.obs.index).all())

        # If they don't match, reorder ldata_subset to match adata_subset:
        self.ldata = self.ldata[self.adata.obs.index, :].copy()

        # Double-check genes are in same order
        print("Genes match:", (self.adata.var.index == self.ldata.var.index).all())

    def integrate(self):
            """
            Integrate and spliced and unspliced counts to adata
            """

            # Add velocity layers to adata_subset
            self.adata.layers["spliced"] = self.ldata.layers["spliced"].copy()
            self.adata.layers["unspliced"] = self.ldata.layers["unspliced"].copy()
            self.adata.layers["ambiguous"] = self.ldata.layers["ambiguous"].copy()

            # Verify they were added
            print("\nLayers in adata_subset:", list(self.adata.layers.keys()))
            print("Spliced shape:", self.adata.layers["spliced"].shape)
            print("Unspliced shape:", self.adata.layers["unspliced"].shape)

    def scVeloPreprocess(self, min_shared_counts=None, n_pcs=None, n_neighbors=None, use_hvgs=None, n_top_genes=None):
        """
        Will run scv.pp.filter_and_normalize(), scv.pp.highly_variable_genes(), and scv.pp.moments()
        Function creates a new instance variable self.adata_hvgs with the new highly variable 
        """
        min_shared_counts = 20 if min_shared_counts is None else min_shared_counts
        n_pcs             = 30 if n_pcs is None else n_pcs
        n_neighbors       = 30 if n_neighbors is None else n_neighbors
        self.use_hvgs     = self.use_hvgs if use_hvgs is None else use_hvgs
        n_top_genes       = 2000 if n_top_genes is None else n_top_genes

        # pre-process
        scv.pp.filter_and_normalize(self.adata, min_shared_counts)

        # subset to HVGs
        if self.use_hvgs:
            sc.pp.highly_variable_genes(self.adata, n_top_genes)
            self.adata = self.adata[:, self.adata.var['highly_variable']].copy()

        # recompute neighbors 
        scv.pp.moments(self.adata, n_pcs, n_neighbors)

    def computeVelocity(self, n_top_genes=None, mode=None):
        """
        Compute velocity on adata object. This step takes a while
        """

        if mode is None:
            mode = 'dynamical'
        elif mode == 's':
            mode = 'stochastic'
        elif mode == 'd':
            mode = 'dynamical'

        # compute velocity ( this step will take a while )
        try:
            scv.tl.recover_dynamics(self.adata, n_top_genes=500)
            scv.tl.velocity(self.adata, mode=mode)
            scv.tl.velocity_graph(self.adata)
        except Exception as e:
            print("Error while fitting dynamics:", e)
            print("Current gene index:", self.adata.var_names[self.adata.var['fit_alpha'].isna()].tolist()[:10])

    # functions to inspect data
    def inspect(self):
        """
        Inspect anndata and ldata objects
        """

        print("\n=== Seurat-derived AnnData (adata) ===")
        print("Cells (obs):", self.adata.n_obs)
        print("Genes (var):", self.adata.n_vars)
        print("First 10 cell names:", self.adata.obs.index[:10].tolist())
        print("First 10 gene names:", self.adata.var.index[:10].tolist())

        print("\n=== Loom Spliced/Unspliced Data (ldata) ===")
        print("Cells (obs):", self.ldata.n_obs)
        print("Genes (var):", self.ldata.n_vars)
        print("First 10 cell names:", self.ldata.obs.index[:10].tolist())
        print("First 10 gene names:", self.ldata.var.index[:10].tolist())

        print("\nLayer Shapes in ldata")
        for layer in ["spliced", "unspliced", "ambiguous"]:
            if layer in self.ldata.layers:
                print(f"{layer}: {self.ldata.layers[layer].shape}")

    def checkDuplicates(self):
        """
        Check duplicates in adta and ldata
        """

        print("\n\n=== Checking for dupicates ===")
        print("Total cells in adata:", len(self.adata.obs.index))
        print("Unique cells in adata:", len(self.adata.obs.index.unique()))
        print("Duplicate barcodes in adata:", len(self.adata.obs.index) - len(self.adata.obs.index.unique()))

        print("\nTotal cells in ldata:", len(self.ldata.obs.index))
        print("Unique cells in ldata:", len(self.ldata.obs.index.unique()))
        print("Duplicate barcodes in ldata:", len(self.ldata.obs.index) - len(self.ldata.obs.index.unique()))

        # Show some duplicates if they exist
        from collections import Counter
        adata_counts = Counter(self.adata.obs.index)
        adata_duplicates = {bc: count for bc, count in adata_counts.items() if count > 1}
        print(f"\nNumber of duplicated barcodes: {len(adata_duplicates)}")
        if adata_duplicates:
            print("Examples:", list(adata_duplicates.items())[:5])

    # getter functions
    def getAdata(self):
        """
        Returns adata object
        """

        return self.adata