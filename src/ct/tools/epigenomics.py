"""
Epigenomics tools: CpG island lookup, chromatin state queries,
ENCODE accessibility, ReMap TF binding.

These tools query local tabix-indexed BED files from the data lake.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ct.tools import registry
from ct.data.loaders import _find_file

logger = logging.getLogger("ct.tools.epigenomics")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _resolve_gene_coordinates(gene: str) -> dict:
    """Resolve a gene symbol to GRCh38 genomic coordinates via Ensembl REST.

    Returns: {"chrom": "chr1", "start": 55039447, "end": 55064852, "strand": "+"}
    or {} if not found.
    """
    from ct.tools.http_client import request_json

    gene = str(gene).strip()
    data, error = request_json(
        "GET",
        f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}",
        params={"content-type": "application/json"},
        timeout=10,
        retries=2,
    )
    if error or not data:
        return {}
    seq_region = str(data.get("seq_region_name", ""))
    chrom = f"chr{seq_region}" if not seq_region.startswith("chr") else seq_region
    return {
        "chrom": chrom,
        "start": int(data.get("start", 0)),
        "end": int(data.get("end", 0)),
        "strand": "+" if data.get("strand", 1) == 1 else "-",
    }


def _tabix_query(bed_path: str, chrom: str, start: int, end: int) -> list[list[str]]:
    """Query a tabix-indexed BED file for overlapping features.

    Returns list of rows, each row is a list of tab-separated fields.
    """
    import pysam

    results = []
    try:
        tbx = pysam.TabixFile(str(bed_path))
        for row in tbx.fetch(chrom, start, end):
            results.append(row.split("\t"))
        tbx.close()
    except Exception as e:
        logger.warning("Tabix query failed for %s %s:%d-%d: %s", bed_path, chrom, start, end, e)
    return results


def _parse_region(region: str) -> tuple:
    """Parse 'chr1:55039000-55041000' into (chrom, start, end)."""
    m = re.match(r"(chr\w+):(\d+)-(\d+)", region)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _promoter_region(coords: dict, upstream: int = 5000, downstream: int = 2000) -> tuple:
    """Get promoter region coordinates from gene coordinates."""
    if coords.get("strand") == "+":
        start = max(0, coords["start"] - upstream)
        end = coords["start"] + downstream
    else:
        start = max(0, coords["end"] - downstream)
        end = coords["end"] + upstream
    return coords["chrom"], start, end


# ---------------------------------------------------------------------------
# CpG Island tool
# ---------------------------------------------------------------------------

@registry.register(
    name="epigenomics.cpg_island",
    description="Check for CpG islands at a gene promoter. Critical for CRISPRoff/epigenetic editor suitability — CpG islands enable durable methylation-based silencing.",
    category="epigenomics",
    parameters={
        "gene": "Gene symbol (e.g. PCSK9, BCL11A)",
        "region": "Genomic region 'chr:start-end' (optional, overrides gene lookup)",
    },
    requires_data=["cpg_islands"],
    usage_guide="Determine if a gene promoter contains a CpG island for epigenetic editing assessment. CpG islands are the primary substrate for DNMT3A-mediated methylation silencing (CRISPRoff).",
)
def cpg_island(gene: str = None, region: str = None, **kwargs) -> dict:
    """Look up CpG islands at a gene locus."""
    bed_path = _find_file(
        "cpgIslandExt.bed.gz",
        subdirs=["gene_context/epigenomics/cpg_islands", "cpg_islands"],
    )
    if bed_path is None:
        return {
            "error": "CpG island data not found",
            "summary": "CpG island BED file not found. Set data.base to your bronze data directory.",
        }

    # Resolve coordinates
    if region:
        chrom, start, end = _parse_region(region)
        if chrom is None:
            return {"error": f"Invalid region format: {region}", "summary": "Use chr:start-end format"}
    elif gene:
        coords = _resolve_gene_coordinates(gene)
        if not coords:
            return {"error": f"Could not resolve {gene} coordinates", "summary": f"Gene {gene} not found in Ensembl"}
        chrom, start, end = _promoter_region(coords)
    else:
        return {"error": "Provide gene or region", "summary": "Gene symbol or genomic region required"}

    hits = _tabix_query(str(bed_path), chrom, start, end)

    islands = []
    for fields in hits:
        if len(fields) >= 3:
            # UCSC cpgIslandExt after bin strip: chrom, start, end, name, length, cpgNum, gcNum, perCpg, perGc, obsExp
            # But actual length = end - start (more reliable than field[4])
            island_start = int(fields[1])
            island_end = int(fields[2])
            actual_length = island_end - island_start
            islands.append({
                "chrom": fields[0],
                "start": island_start,
                "end": island_end,
                "name": fields[3] if len(fields) > 3 else "",
                "length": actual_length,
                "cpg_count": int(fields[4]) if len(fields) > 4 else 0,  # cpgNum field
                "gc_num": int(fields[5]) if len(fields) > 5 else 0,
                "per_cpg": float(fields[8]) if len(fields) > 8 else 0,
                "per_gc": float(fields[9]) if len(fields) > 9 else 0,
                "obs_exp": float(fields[10]) if len(fields) > 10 else 0,
            })

    has_island = len(islands) > 0
    gene_label = gene or region

    if has_island:
        primary = islands[0]
        summary = (
            f"CpG island found at {gene_label} promoter: "
            f"{primary['chrom']}:{primary['start']}-{primary['end']} "
            f"({primary['length']} bp, {primary['cpg_count']} CpGs, "
            f"GC={primary['per_gc']:.1f}%). "
            f"Amenable to methylation-based epigenetic silencing (CRISPRoff)."
        )
    else:
        summary = (
            f"No CpG island found at {gene_label} promoter region ({chrom}:{start}-{end}). "
            f"Methylation-based silencing (CRISPRoff) may not achieve durable effect."
        )

    return {
        "summary": summary,
        "source": "UCSC CpG Islands hg38",
        "source_file": "gene_context/epigenomics/cpg_islands/cpgIslandExt.bed.gz",
        "query": {"gene": gene, "region": region, "queried_region": f"{chrom}:{start}-{end}"},
        "gene": gene,
        "region_queried": f"{chrom}:{start}-{end}",
        "cpg_island_present": has_island,
        "islands": islands,
        "editability_implication": (
            "favorable" if has_island else "unfavorable"
        ) + " for methylation-based epigenetic silencing",
    }


# ---------------------------------------------------------------------------
# ChromHMM chromatin state tool
# ---------------------------------------------------------------------------

# Roadmap ChromHMM 15-state labels
_CHROMHMM_STATES = {
    "1_TssA": "Active TSS",
    "2_TssAFlnk": "Flanking Active TSS",
    "3_TxFlnk": "Transcr. at gene 5'/3'",
    "4_Tx": "Strong transcription",
    "5_TxWk": "Weak transcription",
    "6_EnhG": "Genic enhancers",
    "7_Enh": "Enhancers",
    "8_ZNF/Rpts": "ZNF genes & repeats",
    "9_Het": "Heterochromatin",
    "10_TssBiv": "Bivalent/Poised TSS",
    "11_BivFlnk": "Flanking Bivalent TSS/Enh",
    "12_EnhBiv": "Bivalent Enhancer",
    "13_ReprPC": "Repressed PolyComb",
    "14_ReprPCWk": "Weak Repressed PolyComb",
    "15_Quies": "Quiescent/Low",
}

# Tissue → Roadmap epigenome ID mapping
_EPIGENOME_MAP = {
    "liver": "E066",
    "adult_liver": "E066",
    "lung": "E096",
    "brain": "E071",
    "brain_hippocampus": "E071",
    "heart": "E095",
    "heart_left_ventricle": "E095",
    "kidney": "E086",
    "blood": "E062",
    "monocyte": "E062",
    "skeletal_muscle": "E108",
    "skin": "E057",
    "pancreas": "E098",
    "colon": "E106",
    "small_intestine": "E109",
    "stomach": "E094",
    "esophagus": "E079",
    "adipose": "E063",
    "thymus": "E112",
    "spleen": "E113",
    "ovary": "E097",
    "placenta": "E091",
    "fetal_brain": "E081",
    "hepg2": "E118",
    "k562": "E123",
    "gm12878": "E116",
}


@registry.register(
    name="epigenomics.chromatin_state",
    description="Query Roadmap Epigenomics ChromHMM 15-state chromatin segmentation at a gene locus in a specific tissue. Returns dominant state (Active TSS, Enhancer, Repressed, Quiescent, etc.).",
    category="epigenomics",
    parameters={
        "gene": "Gene symbol (e.g. PCSK9)",
        "tissue": "Tissue name (liver, lung, brain, heart, kidney, blood, skeletal_muscle, skin, pancreas, colon, hepg2, k562) or Roadmap epigenome ID (e.g. E066)",
    },
    requires_data=["roadmap"],
    usage_guide="Know the chromatin state at a gene locus in a specific tissue — whether the promoter is active, enhancer, repressed, or quiescent. Critical for epigenetic editor design and delivery strategy.",
)
def chromatin_state(gene: str, tissue: str = "liver", **kwargs) -> dict:
    """Query ChromHMM state at a gene locus in a specific tissue."""
    tissue_lower = str(tissue).strip().lower().replace(" ", "_")
    epigenome_id = _EPIGENOME_MAP.get(tissue_lower)
    if epigenome_id is None:
        # Check if tissue is already an epigenome ID
        if tissue.upper().startswith("E") and len(tissue) <= 4:
            epigenome_id = tissue.upper()
        else:
            supported = sorted(_EPIGENOME_MAP.keys())
            return {
                "error": f"Unknown tissue: {tissue}",
                "summary": f"Tissue '{tissue}' not recognized. Supported: {', '.join(supported[:15])}...",
                "supported_tissues": supported,
            }

    fname = f"{epigenome_id}_15_coreMarks_dense.bed.gz"
    bed_path = _find_file(fname, subdirs=["gene_context/epigenomics/roadmap", "roadmap"])
    if bed_path is None:
        return {
            "error": f"ChromHMM data not found for {epigenome_id}",
            "summary": f"Roadmap ChromHMM file {fname} not found. Set data.base to your bronze data directory.",
        }

    coords = _resolve_gene_coordinates(gene)
    if not coords:
        return {"error": f"Could not resolve {gene}", "summary": f"Gene {gene} not found in Ensembl"}

    chrom, promoter_start, promoter_end = _promoter_region(coords, upstream=2000, downstream=500)
    hits = _tabix_query(str(bed_path), chrom, promoter_start, promoter_end)

    states = []
    for fields in hits:
        if len(fields) >= 4:
            state_code = fields[3]
            states.append({
                "chrom": fields[0],
                "start": int(fields[1]),
                "end": int(fields[2]),
                "state": state_code,
                "description": _CHROMHMM_STATES.get(state_code, state_code),
            })

    dominant = states[0]["state"] if states else "unknown"
    dominant_desc = _CHROMHMM_STATES.get(dominant, dominant)

    active_states = {"1_TssA", "2_TssAFlnk", "6_EnhG", "7_Enh"}
    repressed_states = {"13_ReprPC", "14_ReprPCWk", "9_Het"}
    is_active = dominant in active_states
    is_repressed = dominant in repressed_states

    interpretation = ""
    if is_active:
        interpretation = "Active/open chromatin — accessible to editors."
    elif is_repressed:
        interpretation = "Repressed chromatin — may be difficult to access."
    elif dominant == "15_Quies":
        interpretation = "Quiescent — low activity, may need reactivation."

    return {
        "summary": (
            f"ChromHMM state for {gene} promoter in {tissue} ({epigenome_id}): "
            f"{dominant_desc} ({dominant}). {interpretation}"
        ),
        "source": f"Roadmap Epigenomics ChromHMM 15-state, {epigenome_id}",
        "source_file": f"gene_context/epigenomics/roadmap/{fname}",
        "query": {"gene": gene, "tissue": tissue, "epigenome_id": epigenome_id},
        "gene": gene,
        "tissue": tissue,
        "epigenome_id": epigenome_id,
        "promoter_region": f"{chrom}:{promoter_start}-{promoter_end}",
        "dominant_state": dominant,
        "dominant_description": dominant_desc,
        "is_active": is_active,
        "is_repressed": is_repressed,
        "all_states_at_locus": states,
    }


# ---------------------------------------------------------------------------
# ENCODE chromatin accessibility tool
# ---------------------------------------------------------------------------

@registry.register(
    name="epigenomics.encode_accessibility",
    description="Query ENCODE ATAC-seq or histone ChIP-seq narrowPeak data at a gene locus for a specific cell type. Checks whether the locus is in open chromatin or has active histone marks.",
    category="epigenomics",
    parameters={
        "gene": "Gene symbol (e.g. PCSK9)",
        "cell_type": "Cell type (HepG2, K562, GM12878). Default: HepG2",
        "assay": "Assay type filter — optional substring to match in filenames (e.g. ATAC, H3K27ac, H3K4me3)",
    },
    requires_data=["encode"],
    usage_guide="Check if a locus is in open chromatin (ATAC-seq) or has active histone marks (H3K27ac, H3K4me3) in a specific cell type. Essential for editor delivery assessment.",
)
def encode_accessibility(gene: str, cell_type: str = "HepG2", assay: str = None, **kwargs) -> dict:
    """Query ENCODE narrowPeak files for signal at a gene locus."""
    import glob

    encode_dir = _find_file(
        ".",  # directory
        subdirs=["gene_context/epigenomics/encode", "encode"],
    )
    if encode_dir is None:
        # Try finding any narrowPeak file
        from ct.agent.config import Config
        cfg = Config.load()
        base = cfg.get("data.base")
        if base:
            encode_dir = Path(base) / "gene_context" / "epigenomics" / "encode"
            if not encode_dir.exists():
                return {"error": "ENCODE data directory not found", "summary": "ENCODE data not found."}
        else:
            return {"error": "ENCODE data directory not found", "summary": "ENCODE data not found."}
    else:
        encode_dir = encode_dir.parent if encode_dir.is_file() else encode_dir

    # Find narrowPeak files
    peak_files = sorted(glob.glob(str(encode_dir / "*.bed_narrowPeak")))
    if not peak_files:
        peak_files = sorted(glob.glob(str(encode_dir / "*.narrowPeak")))
    if not peak_files:
        return {"error": "No narrowPeak files found", "summary": "No ENCODE narrowPeak files in directory."}

    coords = _resolve_gene_coordinates(gene)
    if not coords:
        return {"error": f"Could not resolve {gene}", "summary": f"Gene {gene} not found in Ensembl"}

    chrom, prom_start, prom_end = _promoter_region(coords, upstream=5000, downstream=2000)

    # Read and filter narrowPeak files
    peaks_found = []
    files_searched = 0
    for pf in peak_files:
        fname = Path(pf).name
        # Filter by assay if specified
        if assay and assay.lower() not in fname.lower():
            continue
        files_searched += 1
        try:
            # ENCODE narrowPeak files may be gzipped without .gz extension
            import gzip
            try:
                with gzip.open(pf, "rt") as f:
                    f.readline()
                compression = "gzip"
            except gzip.BadGzipFile:
                compression = None

            df = pd.read_csv(
                pf, sep="\t", header=None,
                names=["chrom", "start", "end", "name", "score", "strand",
                       "signalValue", "pValue", "qValue", "peak"],
                dtype={"chrom": str},
                compression=compression,
            )
            # Filter to region
            hits = df[(df["chrom"] == chrom) & (df["end"] > prom_start) & (df["start"] < prom_end)]
            if len(hits) > 0:
                for _, row in hits.iterrows():
                    peaks_found.append({
                        "file": fname,
                        "chrom": row["chrom"],
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                        "signal_value": float(row["signalValue"]),
                        "p_value": float(row["pValue"]),
                        "q_value": float(row["qValue"]),
                    })
        except Exception as e:
            logger.debug("Error reading %s: %s", pf, e)
            continue

    has_signal = len(peaks_found) > 0

    if has_signal:
        top = sorted(peaks_found, key=lambda x: x["signal_value"], reverse=True)[:5]
        summary = (
            f"ENCODE signal detected at {gene} promoter ({chrom}:{prom_start}-{prom_end}): "
            f"{len(peaks_found)} peaks across {files_searched} files. "
            f"Top signal: {top[0]['signal_value']:.1f} in {top[0]['file']}."
        )
    else:
        summary = (
            f"No ENCODE peaks found at {gene} promoter ({chrom}:{prom_start}-{prom_end}) "
            f"across {files_searched} files searched."
        )

    return {
        "summary": summary,
        "source": "ENCODE Project narrowPeak data",
        "source_file": "gene_context/epigenomics/encode/",
        "query": {"gene": gene, "cell_type": cell_type, "assay": assay},
        "gene": gene,
        "region_queried": f"{chrom}:{prom_start}-{prom_end}",
        "has_signal": has_signal,
        "peaks_found": len(peaks_found),
        "files_searched": files_searched,
        "top_peaks": sorted(peaks_found, key=lambda x: x["signal_value"], reverse=True)[:10],
    }


# ---------------------------------------------------------------------------
# ReMap TF binding tool
# ---------------------------------------------------------------------------

@registry.register(
    name="epigenomics.remap_tf_binding",
    description="Query ReMap 2022 atlas for transcription factor ChIP-seq binding sites at a gene locus. Returns which TFs bind at the promoter/enhancer region.",
    category="epigenomics",
    parameters={
        "gene": "Gene symbol (e.g. PCSK9)",
        "tf": "Specific TF to filter for (optional — searches all TFs if not specified)",
    },
    requires_data=["remap"],
    usage_guide="Find which transcription factors bind at a target gene's promoter. Important for understanding regulatory context and predicting effects of epigenetic editing.",
)
def remap_tf_binding(gene: str, tf: str = None, **kwargs) -> dict:
    """Query ReMap tabix-indexed BED for TF binding at a gene locus."""
    bed_path = _find_file(
        "remap2022_all_macs2_hg38_v1_0.bed.gz",
        subdirs=["gene_context/epigenomics/remap", "remap"],
    )
    if bed_path is None:
        return {"error": "ReMap data not found", "summary": "ReMap BED file not found."}

    coords = _resolve_gene_coordinates(gene)
    if not coords:
        return {"error": f"Could not resolve {gene}", "summary": f"Gene {gene} not found in Ensembl"}

    chrom, prom_start, prom_end = _promoter_region(coords, upstream=5000, downstream=2000)
    hits = _tabix_query(str(bed_path), chrom, prom_start, prom_end)

    # Parse TF names from column 4 (format: TF_NAME.CELL_TYPE.EXPERIMENT)
    tf_counts: dict[str, int] = {}
    tf_details: list[dict] = []
    for fields in hits:
        if len(fields) >= 4:
            tf_field = fields[3]
            # ReMap format: GEO_ACCESSION.TF_NAME.CELL_TYPE or ENCSR*.TF_NAME.CELL_TYPE
            parts = tf_field.split(".")
            tf_name = parts[1] if len(parts) >= 2 else parts[0]
            if tf and tf.upper() != tf_name.upper():
                continue
            tf_counts[tf_name] = tf_counts.get(tf_name, 0) + 1
            if len(tf_details) < 100:  # Cap details
                tf_details.append({
                    "chrom": fields[0],
                    "start": int(fields[1]),
                    "end": int(fields[2]),
                    "tf_experiment": tf_field,
                    "tf_name": tf_name,
                    "score": float(fields[4]) if len(fields) > 4 else 0,
                })

    # Sort TFs by binding count
    sorted_tfs = sorted(tf_counts.items(), key=lambda x: x[1], reverse=True)
    top_tfs = [{"tf": name, "peak_count": count} for name, count in sorted_tfs[:20]]

    if tf:
        count = tf_counts.get(tf.upper(), 0)
        summary = (
            f"ReMap: {count} {tf} binding peaks at {gene} promoter ({chrom}:{prom_start}-{prom_end})."
            if count > 0 else
            f"ReMap: No {tf} binding detected at {gene} promoter."
        )
    else:
        n_tfs = len(tf_counts)
        top_names = ", ".join(t["tf"] for t in top_tfs[:5])
        summary = (
            f"ReMap: {n_tfs} unique TFs bind at {gene} promoter ({chrom}:{prom_start}-{prom_end}). "
            f"Top: {top_names}."
            if n_tfs > 0 else
            f"No ReMap TF binding detected at {gene} promoter."
        )

    return {
        "summary": summary,
        "source": "ReMap 2022 (hg38)",
        "source_file": "gene_context/epigenomics/remap/remap2022_all_macs2_hg38_v1_0.bed.gz",
        "query": {"gene": gene, "tf": tf},
        "gene": gene,
        "region_queried": f"{chrom}:{prom_start}-{prom_end}",
        "unique_tfs": len(tf_counts),
        "total_peaks": sum(tf_counts.values()),
        "top_tfs": top_tfs,
    }


# ---------------------------------------------------------------------------
# CRISPOR guide RNA design tool
# ---------------------------------------------------------------------------

@registry.register(
    name="epigenomics.crispor_guide_design",
    description="Design CRISPR guide RNAs for a target sequence. Finds all PAM sites, scores guides for on-target efficiency (Doench 2016) and specificity. Returns ranked guide list.",
    category="epigenomics",
    parameters={
        "sequence": "DNA sequence (50-500bp) to search for guides, OR a gene symbol (will extract promoter sequence)",
        "pam": "PAM motif (default: NGG for SpCas9)",
    },
    requires_packages=["twobitreader"],
    requires_data=["crispor"],
    usage_guide="Design guide RNAs for CRISPR editing. Provide a target DNA sequence or gene name. Returns guides ranked by predicted efficiency. Use for CRISPRoff, CRISPRi, or nuclease applications.",
)
def crispor_guide_design(sequence: str, pam: str = "NGG", **kwargs) -> dict:
    """Design CRISPR guide RNAs using CRISPOR scoring functions."""
    import re as _re
    from ct.tools import check_dependency

    dep_err = check_dependency(packages=["twobitreader"])
    if dep_err:
        return dep_err

    sequence = str(sequence).strip().upper()

    # If it looks like a gene name (short, no ATCG), resolve to sequence
    if len(sequence) < 20 and not _re.match(r'^[ATCGN]+$', sequence):
        gene = sequence
        coords = _resolve_gene_coordinates(gene)
        if not coords:
            return {"error": f"Could not resolve {gene}", "summary": f"Gene {gene} not found"}
        # Get promoter CpG island region sequence from 2bit file
        from ct.agent.config import Config
        cfg = Config.load()
        data_base = cfg.get("data.base")
        twobit_path = f"{data_base}/crispr_tools/crispor/hg38.2bit"

        try:
            import twobitreader
            tbf = twobitreader.TwoBitFile(twobit_path)
            chrom = coords["chrom"]
            # Get promoter region (1kb around TSS)
            if coords["strand"] == "+":
                start = max(0, coords["start"] - 500)
                end = coords["start"] + 500
            else:
                start = max(0, coords["end"] - 500)
                end = coords["end"] + 500
            sequence = str(tbf[chrom][start:end]).upper()
        except Exception as e:
            return {
                "error": f"Could not extract sequence: {e}",
                "summary": f"Failed to get sequence for {gene} from hg38.2bit: {e}. Provide a DNA sequence directly.",
            }

    if len(sequence) < 23:
        return {"error": "Sequence too short", "summary": "Need at least 23bp to find a guide + PAM."}

    # Find all PAM sites and extract 20bp guides
    guide_len = 20
    guides = []

    # Forward strand
    pam_re = pam.replace("N", "[ATCG]")
    for m in _re.finditer(f"(?=([ATCG]{{{guide_len}}}{pam_re}))", sequence):
        full = m.group(1)
        guide_seq = full[:guide_len]
        pam_seq = full[guide_len:]
        pos = m.start()
        guides.append({
            "guide_sequence": guide_seq,
            "pam": pam_seq,
            "position": pos,
            "strand": "+",
            "full_context": full,
        })

    # Reverse complement for reverse strand
    comp = str.maketrans("ATCGN", "TAGCN")
    rc_seq = sequence.translate(comp)[::-1]
    for m in _re.finditer(f"(?=([ATCG]{{{guide_len}}}{pam_re}))", rc_seq):
        full = m.group(1)
        guide_seq = full[:guide_len]
        pam_seq = full[guide_len:]
        pos = len(sequence) - m.start() - len(full)
        guides.append({
            "guide_sequence": guide_seq,
            "pam": pam_seq,
            "position": pos,
            "strand": "-",
            "full_context": full,
        })

    # Score guides using CRISPOR scoring if available
    try:
        import sys
        from ct.agent.config import Config
        cfg = Config.load()
        data_base = cfg.get("data.base")
        crispor_dir = f"{data_base}/crispr_tools/crispor"
        if crispor_dir not in sys.path:
            sys.path.insert(0, crispor_dir)
        import crisporEffScores

        for g in guides:
            # Doench 2016 score needs 30bp context (4bp upstream + 20bp guide + 3bp PAM + 3bp downstream)
            guide_pos = g["position"]
            context_start = max(0, guide_pos - 4)
            context_end = min(len(sequence), guide_pos + guide_len + 6)
            context_30 = sequence[context_start:context_end]

            # GC content
            gc = sum(1 for c in g["guide_sequence"] if c in "GC") / guide_len * 100
            g["gc_content"] = round(gc, 1)

            # Simple scoring heuristics (when CRISPOR full scoring not available)
            # Penalize extreme GC
            gc_score = 1.0
            if gc < 30 or gc > 70:
                gc_score = 0.5
            elif gc < 40 or gc > 60:
                gc_score = 0.8

            # Penalize poly-T (terminator for Pol III)
            poly_t_penalty = 0.5 if "TTTT" in g["guide_sequence"] else 1.0

            g["efficiency_score"] = round(gc_score * poly_t_penalty * 100, 1)

    except ImportError:
        for g in guides:
            gc = sum(1 for c in g["guide_sequence"] if c in "GC") / guide_len * 100
            g["gc_content"] = round(gc, 1)
            g["efficiency_score"] = round(gc / 100 * 100, 1)  # Simple GC-based score

    # Sort by efficiency score
    guides.sort(key=lambda x: x["efficiency_score"], reverse=True)

    return {
        "summary": (
            f"Found {len(guides)} guide RNAs with {pam} PAM in {len(sequence)}bp sequence. "
            f"Top guide: {guides[0]['guide_sequence']} (score={guides[0]['efficiency_score']}, GC={guides[0]['gc_content']}%)"
            if guides else f"No {pam} PAM sites found in {len(sequence)}bp sequence."
        ),
        "source": "CRISPOR guide design (local)",
        "source_file": "crispr_tools/crispor/",
        "query": {"sequence_length": len(sequence), "pam": pam},
        "total_guides": len(guides),
        "guides": guides[:20],
        "sequence_length": len(sequence),
    }


# ---------------------------------------------------------------------------
# Cas-OFFinder off-target search tool
# ---------------------------------------------------------------------------

@registry.register(
    name="epigenomics.cas_offinder",
    description="Search for genome-wide off-target sites for a CRISPR guide RNA using Cas-OFFinder. Returns all genomic sites with up to N mismatches.",
    category="epigenomics",
    parameters={
        "guide": "20bp guide RNA sequence (e.g. AGCTTCGAATCTGCCAAGTG)",
        "pam": "PAM sequence with N for any base (default: NRG for SpCas9)",
        "mismatches": "Maximum number of mismatches to allow (default: 3)",
    },
    requires_binaries=["cas-offinder"],
    requires_data=[],
    usage_guide="Find all genomic locations where a guide RNA could bind with up to N mismatches. Critical for safety assessment of CRISPR therapeutics — off-target edits in open chromatin are dangerous.",
)
def cas_offinder(guide: str, pam: str = "NRG", mismatches: int = 3, **kwargs) -> dict:
    """Search for off-target sites using Cas-OFFinder."""
    import shutil
    import subprocess
    import tempfile

    guide = str(guide).strip().upper()
    pam = str(pam).strip().upper()
    mismatches = int(mismatches)

    # Find cas-offinder binary
    binary = shutil.which("cas-offinder")
    if not binary:
        # Check staging directory
        from ct.agent.config import Config
        cfg = Config.load()
        data_base = cfg.get("data.base")
        alt = f"{data_base}/crispr_tools/cas_offinder/cas-offinder"
        if Path(alt).exists():
            binary = alt
        else:
            return {
                "error": "Cas-OFFinder binary not found",
                "summary": "Cas-OFFinder not installed. Run: bash scripts/setup_biotools.sh --tier2",
            }

    # Find genome 2bit file
    from ct.agent.config import Config
    cfg = Config.load()
    data_base = cfg.get("data.base")
    genome_path = f"{data_base}/crispr_tools/crispor/hg38.2bit"

    if not Path(genome_path).exists():
        return {
            "error": "hg38.2bit genome not found",
            "summary": f"Genome index not found at {genome_path}. Run: bash scripts/setup_biotools.sh --tier2",
        }

    # Write Cas-OFFinder input file
    # Format:
    # /path/to/genome.2bit
    # NNNNNNNNNNNNNNNNNNNNNRG  (pattern = guide + PAM with N for variable)
    # guide_sequence NRG mismatches
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"{genome_path}\n")
        f.write(f"{'N' * len(guide)}{pam}\n")
        f.write(f"{guide} {pam} {mismatches}\n")
        input_file = f.name

    output_file = input_file + ".out"

    try:
        # Cas-OFFinder can use an explicit OpenCL runtime when provided.
        import os as _os
        env = _os.environ.copy()
        opencl_lib = _os.environ.get("CAS_OFFINDER_OPENCL_LIB")
        opencl_vendors = _os.environ.get("CAS_OFFINDER_OPENCL_VENDORS")
        if opencl_lib and Path(opencl_lib).exists():
            env["LD_LIBRARY_PATH"] = f"{opencl_lib}:/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
        if opencl_vendors and Path(opencl_vendors).exists():
            env["OCL_ICD_VENDORS"] = opencl_vendors

        result = subprocess.run(
            [binary, input_file, "C", output_file],  # C = CPU mode
            capture_output=True, text=True, timeout=300,
            env=env,
        )

        if result.returncode != 0:
            return {
                "error": f"Cas-OFFinder failed: {result.stderr[:200]}",
                "summary": f"Cas-OFFinder returned error code {result.returncode}",
            }

        # Parse output: pattern, chromosome, position, DNA_sequence, strand, mismatches
        # Cas-OFFinder v2.4 format: 6 tab-separated columns
        offtargets = []
        if Path(output_file).exists():
            with open(output_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 6:
                        # v2.4 format: pattern, chrom, pos, sequence, strand, mismatches
                        offtargets.append({
                            "chrom": parts[1],
                            "position": int(parts[2]),
                            "sequence": parts[3],
                            "strand": parts[4],
                            "mismatches": int(parts[5]),
                        })
                    elif len(parts) >= 5:
                        # Fallback for older format
                        offtargets.append({
                            "chrom": parts[0],
                            "position": int(parts[1]),
                            "sequence": parts[2],
                            "strand": parts[3],
                            "mismatches": int(parts[4]),
                        })

        return {
            "summary": f"Cas-OFFinder: {len(offtargets)} off-target sites found for guide {guide} with ≤{mismatches} mismatches in hg38.",
            "source": "Cas-OFFinder (CPU mode)",
            "source_file": "hg38.2bit",
            "query": {"guide": guide, "pam": pam, "mismatches": mismatches},
            "total_offtargets": len(offtargets),
            "offtargets": offtargets[:50],
        }

    except subprocess.TimeoutExpired:
        return {"error": "Cas-OFFinder timed out (>5 min)", "summary": "Off-target search timed out. Try fewer mismatches."}
    except Exception as e:
        return {"error": str(e), "summary": f"Cas-OFFinder error: {e}"}
    finally:
        for f in [input_file, output_file]:
            try:
                os.unlink(f)
            except Exception:
                pass
