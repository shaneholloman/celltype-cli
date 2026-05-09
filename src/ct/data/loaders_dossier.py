"""
Data loaders for dossier-specific datasets.

Each loader uses _find_file() from the parent module, @lru_cache for
singleton loading, and returns pandas DataFrames.
"""

import pandas as pd
from functools import lru_cache

from ct.data.loaders import _find_file


# Bronze search subdirectories for each data source
_BRONZE_SEARCH_SUBDIRS = {
    "gnomad": ["gene_context/genomic/gnomad", "gnomad"],
    "impc": ["gene_context/mouse_models/impc", "impc"],
    "mgi": ["gene_context/mouse_models/mgi", "mgi"],
    "dgidb": ["gene_context/drug_target/dgidb", "dgidb"],
    "omnipath": ["gene_context/networks/omnipath", "omnipath"],
}


@lru_cache(maxsize=1)
def load_gnomad_constraints() -> pd.DataFrame:
    """Load gnomAD v4.1 constraint metrics.

    Returns DataFrame with columns: gene, transcript, canonical,
    lof_hc_lc.pLI, lof.oe_ci.upper, lof.oe, mis.z_score, syn.z_score, etc.
    Filtered to canonical transcripts for unique gene lookup.
    """
    path = _find_file(
        "gnomad.v4.1.constraint_metrics.tsv",
        subdirs=_BRONZE_SEARCH_SUBDIRS["gnomad"],
    )
    if path is None:
        raise FileNotFoundError(
            "gnomAD constraint data not found. "
            "Download with: bash scripts/download_datasources/gnomad.sh\n"
            "Or set: ct config set data.base /path/to/bronze"
        )
    df = pd.read_csv(path, sep="\t")
    # Filter to canonical transcripts for unique gene lookup
    if "canonical" in df.columns:
        df_canonical = df[df["canonical"] == True]  # noqa: E712
        if len(df_canonical) > 0:
            df = df_canonical
    return df


@lru_cache(maxsize=1)
def load_impc_phenotypes() -> pd.DataFrame:
    """Load IMPC genotype-phenotype associations (actual phenotype calls).

    Uses genotype_phenotype.csv which contains the phenotype calls with MP terms,
    rather than statistical_results.csv which has parameter-level measurements.

    Returns DataFrame with columns including: marker_symbol, mp_term_id,
    mp_term_name, p_value, effect_size, phenotyping_center, zygosity, etc.
    """
    # Prefer genotype_phenotype.csv (has actual MP term phenotype calls)
    path = _find_file(
        "genotype_phenotype.csv",
        subdirs=_BRONZE_SEARCH_SUBDIRS["impc"],
    )
    if path is None:
        # Fall back to statistical_results.csv
        path = _find_file(
            "statistical_results.csv",
            subdirs=_BRONZE_SEARCH_SUBDIRS["impc"],
        )
    if path is None:
        raise FileNotFoundError(
            "IMPC phenotype data not found. "
            "Download with: python scripts/download_datasources/impc.py"
        )
    return pd.read_csv(path, low_memory=False)


@lru_cache(maxsize=1)
def load_impc_gene_list() -> pd.DataFrame:
    """Load IMPC gene list with human orthologs."""
    path = _find_file(
        "gene_list.csv",
        subdirs=_BRONZE_SEARCH_SUBDIRS["impc"],
    )
    if path is None:
        raise FileNotFoundError("IMPC gene list not found.")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_mgi_phenotypes() -> pd.DataFrame:
    """Load MGI gene-phenotype associations.

    The MGI_GenePheno.rpt file is tab-separated with no header.
    Columns: allelic_composition, allele_symbol, allele_id, genetic_background,
    mp_id, phenotype, ref_id.
    """
    path = _find_file(
        "MGI_GenePheno.rpt",
        subdirs=_BRONZE_SEARCH_SUBDIRS["mgi"],
    )
    if path is None:
        raise FileNotFoundError(
            "MGI gene-phenotype data not found. "
            "Download with: bash scripts/download_datasources/mgi.sh"
        )
    return pd.read_csv(
        path, sep="\t", header=None,
        names=[
            "allelic_composition", "allele_symbol", "allele_id",
            "genetic_background", "mp_id", "phenotype", "ref_id",
        ],
    )


@lru_cache(maxsize=1)
def load_mgi_human_phenotypes() -> pd.DataFrame:
    """Load MGI human-mouse disease connection data."""
    path = _find_file(
        "HMD_HumanPhenotype.rpt",
        subdirs=_BRONZE_SEARCH_SUBDIRS["mgi"],
    )
    if path is None:
        raise FileNotFoundError("MGI human phenotype data not found.")
    return pd.read_csv(path, sep="\t", header=None)


@lru_cache(maxsize=1)
def load_dgidb_interactions() -> pd.DataFrame:
    """Load DGIdb drug-gene interactions.

    Returns DataFrame with columns: gene_name, drug_name, drug_concept_id,
    interaction_score, interaction_types.
    """
    path = _find_file(
        "interactions.tsv",
        subdirs=_BRONZE_SEARCH_SUBDIRS["dgidb"],
    )
    if path is None:
        raise FileNotFoundError("DGIdb interactions data not found.")
    return pd.read_csv(path, sep="\t")


@lru_cache(maxsize=1)
def load_omnipath_interactions() -> pd.DataFrame:
    """Load OmniPath protein-protein interactions."""
    path = _find_file(
        "omnipath_interactions_human.tsv",
        subdirs=_BRONZE_SEARCH_SUBDIRS["omnipath"],
    )
    if path is None:
        raise FileNotFoundError("OmniPath interactions data not found.")
    return pd.read_csv(path, sep="\t")


@lru_cache(maxsize=1)
def load_omnipath_tf_targets() -> pd.DataFrame:
    """Load OmniPath TF-target interactions."""
    path = _find_file(
        "omnipath_tf_target_human.tsv",
        subdirs=_BRONZE_SEARCH_SUBDIRS["omnipath"],
    )
    if path is None:
        raise FileNotFoundError("OmniPath TF target data not found.")
    return pd.read_csv(path, sep="\t")
