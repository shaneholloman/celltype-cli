"""
Data catalog for the CellType data lake.

Structured registry of all downloaded biological data sources.
Injected into the agent system prompt so it knows what data is available
and can query it via run_python or dedicated tools.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataSource:
    """A registered data source in the data lake."""
    name: str
    id: str
    category: str
    path: str            # Relative to DATA_ROOT
    format: str          # tsv, csv, parquet, bed_tabix, vcf, json, gmt, h5ad, xml, fasta
    size: str
    description: str
    key_columns: list[str] = field(default_factory=list)
    load_hint: str = ""
    source_url: str = ""
    version: str = ""
    license: str = "Open access"


# ---------------------------------------------------------------------------
# Full catalog. Paths are relative to the configured ``data.base`` directory.
# ---------------------------------------------------------------------------

CATALOG: list[DataSource] = [
    # ===== GENOMIC / GENETIC =====
    DataSource(
        name="gnomAD Constraint",
        id="gnomad_constraint",
        category="Genomic",
        path="gene_context/genomic/gnomad/gnomad.v4.1.constraint_metrics.tsv",
        format="tsv",
        size="96 MB",
        description="Loss-of-function constraint for ~19,700 genes. pLI>0.9=haploinsufficient, lof.oe_ci.upper<0.35=constrained.",
        key_columns=["gene", "transcript", "canonical", "lof_hc_lc.pLI", "lof.oe_ci.upper", "lof.oe", "mis.z_score", "syn.z_score"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/genomic/gnomad/gnomad.v4.1.constraint_metrics.tsv", sep="\\t")',
        source_url="https://gnomad.broadinstitute.org",
        version="v4.1",
    ),
    DataSource(
        name="GeneBass",
        id="genebass",
        category="Genomic",
        path="gene_context/genomic/genebass/",
        format="tsv",
        size="20 MB (exported)",
        description="UK Biobank 500K exome gene burden associations (399K significant gene-phenotype pairs, p<0.001). Natural human LoF phenotypes for ~18K genes. PCSK9 pLoF → LDL p=3.6e-132.",
        key_columns=["gene_symbol", "annotation", "description", "Pvalue_Burden", "BETA_Burden", "SE_Burden", "n_cases", "total_variants_pheno"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/genomic/genebass/genebass_significant_burden.tsv.bgz", sep="\\t", compression="gzip")',
        source_url="https://genebass.org",
        version="500K",
    ),
    DataSource(
        name="ClinVar",
        id="clinvar",
        category="Genomic",
        path="gene_context/genomic/clinvar/",
        format="vcf",
        size="7.0 GB",
        description="Clinical variant classifications. GRCh37/38 VCFs (tabix-indexed) + variant_summary.txt + full XML.",
        key_columns=["GeneSymbol", "ClinicalSignificance", "ReviewStatus", "PhenotypeList"],
        load_hint='import pysam; vcf = pysam.VariantFile(str(DATA_ROOT / "gene_context/genomic/clinvar/clinvar_GRCh38.vcf.gz"))',
        source_url="https://www.ncbi.nlm.nih.gov/clinvar/",
        version="latest",
    ),
    DataSource(
        name="GWAS Catalog",
        id="gwas_catalog",
        category="Genomic",
        path="gene_context/genomic/gwas_catalog/",
        format="tsv",
        size="245 MB",
        description="NHGRI-EBI GWAS associations, studies, and ancestry data.",
        key_columns=["MAPPED_GENE", "DISEASE/TRAIT", "P-VALUE", "OR or BETA", "SNPS", "PUBMEDID"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/genomic/gwas_catalog/gwas_catalog_associations_full.zip", sep="\\t", compression="zip")',
        source_url="https://www.ebi.ac.uk/gwas/",
        version="2026-03",
    ),
    DataSource(
        name="GTEx",
        id="gtex",
        category="Genomic",
        path="gene_context/genomic/gtex/",
        format="tsv",
        size="2.2 GB",
        description="Tissue expression across 54 tissues. Median TPM per gene + sample-level data.",
        key_columns=["Name", "Description", "Adipose_Subcutaneous", "Brain_Cortex", "Liver", "Lung", "Heart_Left_Ventricle"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/genomic/gtex/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz", sep="\\t", skiprows=2, compression="gzip")',
        source_url="https://gtexportal.org",
        version="v10",
    ),
    DataSource(
        name="Ensembl",
        id="ensembl",
        category="Genomic",
        path="gene_context/genomic/ensembl/",
        format="gtf",
        size="1.1 GB",
        description="Human gene annotations (GTF/GFF3), primary assembly FASTA, protein/cDNA/ncRNA FASTA.",
        key_columns=["seqname", "source", "feature", "start", "end", "gene_name", "gene_biotype"],
        load_hint='import gzip; lines = gzip.open(DATA_ROOT / "gene_context/genomic/ensembl/Homo_sapiens.GRCh38.114.gtf.gz", "rt")',
        source_url="https://ensembl.org",
        version="Release 114",
    ),

    # ===== OPEN TARGETS =====
    DataSource(
        name="Open Targets",
        id="open_targets",
        category="Open Targets",
        path="gene_context/targets/open_targets/25.03/",
        format="parquet",
        size="8.5 GB",
        description="19 Parquet datasets: associations (6 levels), evidence (23 sources), interactions, pharmacogenomics, baseline expression, GO, reactome, mouse phenotypes.",
        key_columns=["targetId", "diseaseId", "score", "datasourceId"],
        load_hint='pd.read_parquet(DATA_ROOT / "gene_context/targets/open_targets/25.03/associationByOverallDirect/")',
        source_url="https://platform.opentargets.org",
        version="25.03",
    ),

    # ===== PATHWAYS =====
    DataSource(
        name="MSigDB",
        id="msigdb",
        category="Pathways",
        path="gene_context/pathways/msigdb/",
        format="gmt",
        size="34 MB",
        description="24 GMT gene set collections: hallmarks, GO, oncogenic, immunologic, cell type signatures.",
        key_columns=["gene_set_name", "description_url", "gene1", "gene2", "..."],
        load_hint='[line.strip().split("\\t") for line in open(DATA_ROOT / "gene_context/pathways/msigdb/h.all.v2024.1.Hs.symbols.gmt")]',
        source_url="https://www.gsea-msigdb.org",
        version="v2024.1",
    ),
    DataSource(
        name="Reactome",
        id="reactome",
        category="Pathways",
        path="gene_context/pathways/reactome/",
        format="tsv",
        size="705 MB",
        description="Curated pathway hierarchy and cross-references (NCBI, UniProt, Ensembl, ChEBI).",
        key_columns=["pathway_id", "pathway_name", "species"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/pathways/reactome/NCBI2Reactome_All_Levels.txt", sep="\\t", header=None)',
        source_url="https://reactome.org",
        version="2026-03",
    ),
    DataSource(
        name="KEGG",
        id="kegg",
        category="Pathways",
        path="gene_context/pathways/kegg/",
        format="xml",
        size="21 MB",
        description="370 human pathway KGML files + compound/enzyme/disease lists.",
        key_columns=["pathway_id", "pathway_name", "gene_entries"],
        load_hint='import xml.etree.ElementTree as ET; tree = ET.parse(DATA_ROOT / "gene_context/pathways/kegg/hsa00010.xml")',
        source_url="https://www.kegg.jp",
        version="2026-03",
        license="Academic use (commercial requires license)",
    ),
    DataSource(
        name="Gene Ontology",
        id="go",
        category="Pathways",
        path="gene_context/pathways/go/",
        format="obo",
        size="50 MB",
        description="GO ontology (OBO format) + human gene annotations (GAF format).",
        key_columns=["DB_Object_Symbol", "GO_ID", "Aspect", "Evidence_Code"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/pathways/go/goa_human.gaf.gz", sep="\\t", comment="!", header=None, compression="gzip")',
        source_url="http://geneontology.org",
        version="2026-01",
    ),
    DataSource(
        name="Enrichr",
        id="enrichr",
        category="Pathways",
        path="gene_context/pathways/enrichr/",
        format="gmt",
        size="238 MB",
        description="79 gene set libraries as GMT files (Achilles, Allen Brain Atlas, aging, etc.).",
        key_columns=["gene_set_name", "genes"],
        load_hint='import glob; gmts = glob.glob(str(DATA_ROOT / "gene_context/pathways/enrichr/*.gmt"))',
        source_url="https://maayanlab.cloud/Enrichr/",
        version="2026-03",
    ),

    # ===== NETWORKS =====
    DataSource(
        name="STRING",
        id="string",
        category="Networks",
        path="gene_context/networks/string/",
        format="tsv",
        size="212 MB",
        description="Protein-protein interactions v12.0. Detailed links with per-channel scores.",
        key_columns=["protein1", "protein2", "combined_score", "experimental", "database", "textmining"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/networks/string/9606.protein.links.detailed.v12.0.txt.gz", sep=" ", compression="gzip")',
        source_url="https://string-db.org",
        version="v12.0",
    ),
    DataSource(
        name="OmniPath",
        id="omnipath",
        category="Networks",
        path="gene_context/networks/omnipath/",
        format="tsv",
        size="2.2 GB",
        description="Integrated signaling: interactions, TF targets, miRNA targets, ligand-receptor, complexes, annotations.",
        key_columns=["source", "target", "is_directed", "is_stimulation", "is_inhibition", "consensus_direction"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/networks/omnipath/omnipath_interactions_human.tsv", sep="\\t")',
        source_url="https://omnipathdb.org",
        version="2026-03",
    ),
    DataSource(
        name="BioGRID",
        id="biogrid",
        category="Networks",
        path="gene_context/networks/biogrid/",
        format="tsv",
        size="2.3 GB",
        description="Protein-protein interactions in MITAB format, all organisms.",
        key_columns=["ID_Interactor_A", "ID_Interactor_B", "Alt_IDs_A", "Alt_IDs_B", "Interaction_Detection_Method"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/networks/biogrid/BIOGRID-ALL-5.0.255.mitab.txt", sep="\\t", header=0)',
        source_url="https://thebiogrid.org",
        version="5.0.255",
    ),

    # ===== EPIGENOMICS =====
    DataSource(
        name="CpG Islands",
        id="cpg_islands",
        category="Epigenomics",
        path="gene_context/epigenomics/cpg_islands/cpgIslandExt.bed.gz",
        format="bed_tabix",
        size="1.5 MB",
        description="UCSC hg38 CpG islands, bgzip+tabix indexed. Critical for CRISPRoff/epigenetic editor assessment.",
        key_columns=["chrom", "start", "end", "name", "length", "cpgNum", "gcNum", "perCpg", "perGc", "obsExp"],
        load_hint='import pysam; tbx = pysam.TabixFile(str(DATA_ROOT / "gene_context/epigenomics/cpg_islands/cpgIslandExt.bed.gz")); list(tbx.fetch("chr1", 55039000, 55041000))',
        source_url="https://genome.ucsc.edu",
        version="hg38",
    ),
    DataSource(
        name="Roadmap ChromHMM",
        id="roadmap_chromhmm",
        category="Epigenomics",
        path="gene_context/epigenomics/roadmap/",
        format="bed_tabix",
        size="542 MB",
        description="15-state chromatin segmentation for 125 epigenomes. Files: E0XX_15_coreMarks_dense.bed.gz (bgzip+tabix). States: 1_TssA, 2_TssAFlnk, ..., 15_Quies.",
        key_columns=["chrom", "start", "end", "state", "score"],
        load_hint='import pysam; tbx = pysam.TabixFile(str(DATA_ROOT / "gene_context/epigenomics/roadmap/E066_15_coreMarks_dense.bed.gz")); list(tbx.fetch("chr1", 55039000, 55065000))',
        source_url="https://egg2.wustl.edu/roadmap/",
        version="2015",
    ),
    DataSource(
        name="ReMap",
        id="remap",
        category="Epigenomics",
        path="gene_context/epigenomics/remap/remap2022_all_macs2_hg38_v1_0.bed.gz",
        format="bed_tabix",
        size="4.3 GB",
        description="ReMap 2022 TF ChIP-seq peaks for hg38, bgzip+tabix indexed. Column 4 = TF name.",
        key_columns=["chrom", "start", "end", "TF_name", "score"],
        load_hint='import pysam; tbx = pysam.TabixFile(str(DATA_ROOT / "gene_context/epigenomics/remap/remap2022_all_macs2_hg38_v1_0.bed.gz")); list(tbx.fetch("chr1", 55039000, 55041000))',
        source_url="https://remap2022.univ-amu.fr",
        version="2022",
    ),
    DataSource(
        name="ENCODE",
        id="encode",
        category="Epigenomics",
        path="gene_context/epigenomics/encode/",
        format="bed",
        size="273 MB",
        description="ATAC-seq + H3K27ac + H3K4me3 narrowPeak files for priority cell types (HepG2, K562, GM12878, etc.).",
        key_columns=["chrom", "start", "end", "name", "score", "signalValue", "pValue", "qValue", "peak"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/epigenomics/encode/<filename>.bed_narrowPeak", sep="\\t", header=None, names=["chrom","start","end","name","score","strand","signalValue","pValue","qValue","peak"])',
        source_url="https://www.encodeproject.org",
        version="2026-03",
    ),
    DataSource(
        name="ABC Enhancer-Gene Model",
        id="abc_model",
        category="Epigenomics",
        path="gene_context/epigenomics/abc_model/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz",
        format="tsv",
        size="325 MB",
        description="Activity-by-Contact enhancer-gene predictions (Nasser et al. 2021). Links enhancers to target genes across 131 biosamples using Hi-C + chromatin accessibility. Critical for predicting collateral regulatory effects of epigenetic editing.",
        key_columns=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score", "CellType", "isSelfPromoter", "distance"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/epigenomics/abc_model/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz", sep="\\t", compression="gzip")',
        source_url="https://www.engreitzlab.org/resources/",
        version="Nasser2021",
    ),
    DataSource(
        name="ChIP-Atlas",
        id="chip_atlas",
        category="Epigenomics",
        path="gene_context/epigenomics/chip_atlas/",
        format="tsv",
        size="348 MB",
        description="ChIP-Atlas experiment metadata: 572K experiments (138K hg38) covering 22K TFs, 35K ATAC-seq, 25K histone ChIP-seq. Metadata for experiment discovery; individual BED files fetchable on-demand from chip-atlas.dbcls.jp.",
        key_columns=["experiment_id", "genome", "antigen_class", "antigen", "cell_type_class", "cell_type"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/epigenomics/chip_atlas/experimentList.tab", sep="\\t", header=None, on_bad_lines="skip")',
        source_url="https://chip-atlas.org",
        version="2026-03",
    ),
    DataSource(
        name="4DN Hi-C",
        id="4dn_hic",
        category="Epigenomics",
        path="gene_context/epigenomics/4dn_hic/",
        format="json",
        size="484 KB",
        description="4D Nucleome Hi-C experiment metadata and processed file references.",
        key_columns=["experiment_type", "biosample", "lab", "files"],
        load_hint='json.load(open(DATA_ROOT / "gene_context/epigenomics/4dn_hic/4dn_hic_experiments.json"))',
        source_url="https://data.4dnucleome.org",
        version="2026-03",
    ),
    DataSource(
        name="BLUEPRINT",
        id="blueprint",
        category="Epigenomics",
        path="gene_context/epigenomics/blueprint/",
        format="json",
        size="28 KB",
        description="Hematopoietic epigenome experiment metadata from IHEC/BLUEPRINT.",
        key_columns=["experiment_id", "cell_type", "assay"],
        load_hint='json.load(open(DATA_ROOT / "gene_context/epigenomics/blueprint/blueprint_experiments.json"))',
        source_url="https://www.blueprint-epigenome.eu",
        version="2026-03",
    ),

    # ===== PROTEIN / STRUCTURE =====
    DataSource(
        name="UniProt",
        id="uniprot",
        category="Protein",
        path="gene_context/protein/uniprot/",
        format="fasta",
        size="200 MB",
        description="Swiss-Prot human reviewed entries: FASTA (~20,400 proteins), XML (full annotations), ID mapping.",
        key_columns=["accession", "entry_name", "protein_name", "gene_name", "organism"],
        load_hint='from Bio import SeqIO; records = list(SeqIO.parse(DATA_ROOT / "gene_context/protein/uniprot/uniprot_sprot_human.fasta.gz", "fasta"))',
        source_url="https://www.uniprot.org",
        version="2026-03",
    ),
    DataSource(
        name="InterPro",
        id="interpro",
        category="Protein",
        path="gene_context/protein/interpro/",
        format="tsv",
        size="16 GB",
        description="Protein domain annotations: full protein2ipr mapping, entry list, hierarchy.",
        key_columns=["uniprot_accession", "interpro_id", "interpro_name", "start", "end"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/protein/interpro/protein2ipr.dat.gz", sep="\\t", header=None, compression="gzip")',
        source_url="https://www.ebi.ac.uk/interpro/",
        version="2026-03",
    ),
    DataSource(
        name="PDB",
        id="pdb",
        category="Protein",
        path="gene_context/protein/pdb/",
        format="mmcif",
        size="42 MB",
        description="200 recent human protein structures in mmCIF format.",
        key_columns=["pdb_id", "resolution", "method"],
        load_hint='import glob; cifs = glob.glob(str(DATA_ROOT / "gene_context/protein/pdb/*.cif"))',
        source_url="https://www.rcsb.org",
        version="2026-03",
    ),
    DataSource(
        name="PubChem",
        id="pubchem",
        category="Protein",
        path="gene_context/protein/pubchem/",
        format="tsv",
        size="906 MB",
        description="Compound-gene links: CID-MeSH mappings + filtered synonyms for drug-like compounds.",
        key_columns=["CID", "MeSH_term"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/protein/pubchem/CID-MeSH", sep="\\t", header=None, names=["CID","MeSH"])',
        source_url="https://pubchem.ncbi.nlm.nih.gov",
        version="2026-03",
    ),

    # ===== DRUG / CHEMICAL =====
    DataSource(
        name="DGIdb",
        id="dgidb",
        category="Drug",
        path="gene_context/drug_target/dgidb/interactions.tsv",
        format="tsv",
        size="4.1 MB",
        description="69,907 drug-gene interactions with gene, drug, scores, and interaction types.",
        key_columns=["gene_name", "drug_name", "drug_concept_id", "interaction_score", "interaction_types"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/drug_target/dgidb/interactions.tsv", sep="\\t")',
        source_url="https://www.dgidb.org",
        version="2026-03",
    ),
    DataSource(
        name="TTD",
        id="ttd",
        category="Drug",
        path="gene_context/drug_target/ttd/",
        format="tsv",
        size="19 MB",
        description="Therapeutic Target Database: target info, drug info, cross-matching.",
        key_columns=["TargetID", "TargetName", "TargetType", "DrugName", "ClinicalStatus", "Indication"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/drug_target/ttd/P1-01-TTD_target_download.txt", sep="\\t", header=None)',
        source_url="https://db.idrblab.net/ttd/",
        version="2026-03",
    ),

    # ===== METABOLIC =====
    DataSource(
        name="HMDB",
        id="hmdb",
        category="Metabolic",
        path="gene_context/metabolic/hmdb/",
        format="xml",
        size="1.0 GB",
        description="Human Metabolome Database: 220K+ metabolites (XML), protein/enzyme annotations (XML), chemical structures (SDF). Metabolite-disease associations, pathways, biofluid concentrations, enzyme substrates.",
        key_columns=["accession", "name", "chemical_formula", "diseases", "pathways", "protein_associations", "biological_properties"],
        load_hint='import zipfile; z = zipfile.ZipFile(DATA_ROOT / "gene_context/metabolic/hmdb/hmdb_metabolites.zip"); # Parse XML with ElementTree',
        source_url="https://hmdb.ca",
        version="2021-11",
    ),
    DataSource(
        name="Recon3D",
        id="recon3d",
        category="Metabolic",
        path="gene_context/metabolic/recon3d/",
        format="json",
        size="2.1 MB",
        description="Genome-scale metabolic model: 5,835 metabolites, 10,600 reactions, 2,248 genes.",
        key_columns=["metabolites", "reactions", "genes"],
        load_hint='json.load(open(DATA_ROOT / "gene_context/metabolic/recon3d/Recon3D.json"))',
        source_url="https://www.vmh.life",
        version="3.0",
    ),
    DataSource(
        name="Human-GEM",
        id="human_gem",
        category="Metabolic",
        path="gene_context/metabolic/human_metabolic_atlas/",
        format="xml",
        size="54 MB",
        description="Human Metabolic Atlas v1.19.0: genome-scale model (XML, YML, gene/reaction/metabolite TSVs).",
        key_columns=["gene_id", "gene_name", "reaction_id", "metabolite_id"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/metabolic/human_metabolic_atlas/genes.tsv", sep="\\t")',
        source_url="https://metabolicatlas.org",
        version="v1.19.0",
    ),

    # ===== MOUSE MODELS =====
    DataSource(
        name="IMPC",
        id="impc",
        category="Mouse Models",
        path="gene_context/mouse_models/impc/",
        format="csv",
        size="93 MB",
        description="International Mouse Phenotyping Consortium: 9,000+ KO phenotypes with p-values.",
        key_columns=["marker_symbol", "mp_term_id", "mp_term_name", "p_value", "effect_size", "phenotyping_center"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/mouse_models/impc/statistical_results.csv")',
        source_url="https://www.mousephenotype.org",
        version="2026-03",
    ),
    DataSource(
        name="MGI",
        id="mgi",
        category="Mouse Models",
        path="gene_context/mouse_models/mgi/",
        format="tsv",
        size="89 MB",
        description="Mouse Genome Informatics: gene-phenotype associations + human ortholog mappings.",
        key_columns=["allele_symbol", "marker_symbol", "mp_id", "phenotype"],
        load_hint='pd.read_csv(DATA_ROOT / "gene_context/mouse_models/mgi/MGI_GenePheno.rpt", sep="\\t", header=None)',
        source_url="https://www.informatics.jax.org",
        version="2026-03",
    ),

    # ===== SAFETY =====
    DataSource(
        name="FAERS",
        id="faers",
        category="Safety",
        path="safety/faers/",
        format="json",
        size="34 GB",
        description="FDA Adverse Event Reporting System: 578 quarterly JSON bulk files from OpenFDA.",
        key_columns=["safetyreportid", "serious", "receivedate", "patient.drug", "patient.reaction"],
        load_hint='import zipfile, json; z = zipfile.ZipFile(DATA_ROOT / "safety/faers/drug-event-0001-of-0004.json.zip"); data = json.loads(z.read(z.namelist()[0]))',
        source_url="https://open.fda.gov",
        version="2026-03",
    ),
    DataSource(
        name="IEDB",
        id="iedb",
        category="Safety",
        path="safety/iedb/",
        format="csv",
        size="43 MB",
        description="Immune Epitope Database: T-cell epitope data v3.",
        key_columns=["epitope_sequence", "source_organism", "mhc_allele", "assay_type"],
        load_hint='import zipfile; z = zipfile.ZipFile(DATA_ROOT / "safety/iedb/tcell_full_v3.zip"); pd.read_csv(z.open(z.namelist()[0]))',
        source_url="https://www.iedb.org",
        version="v3",
    ),
    DataSource(
        name="Anti-target Panels",
        id="antitarget_panels",
        category="Safety",
        path="safety/antitarget_panels/",
        format="json",
        size="8 KB",
        description="Curated safety-liability targets: hERG, CYP450s, transporters, PDEs, GPCRs (8 panels).",
        key_columns=["panel", "genes.symbol", "genes.name", "genes.liability_category"],
        load_hint='json.load(open(DATA_ROOT / "safety/antitarget_panels/antitarget_panels.json"))',
        source_url="Internal curation",
        version="2026-03",
    ),

    # ===== CLINICAL =====
    DataSource(
        name="ClinicalTrials.gov",
        id="clinicaltrials_gov",
        category="Clinical",
        path="clinical/clinicaltrials_gov/",
        format="json",
        size="9.3 GB",
        description="~577K clinical trials via API v2 (paginated JSON files).",
        key_columns=["nctId", "briefTitle", "overallStatus", "phases", "conditions", "interventions"],
        load_hint='import json, glob; files = glob.glob(str(DATA_ROOT / "clinical/clinicaltrials_gov/*.json")); data = json.load(open(files[0]))',
        source_url="https://clinicaltrials.gov",
        version="2026-03",
    ),

    # ===== LITERATURE =====
    DataSource(
        name="OpenAlex",
        id="openalex",
        category="Literature",
        path="literature/openalex/",
        format="json",
        size="339 MB",
        description="Biology/medicine works (2023-2026, cited>10): 50 paginated JSON files.",
        key_columns=["id", "title", "cited_by_count", "publication_year", "primary_location"],
        load_hint='import json; data = json.load(open(DATA_ROOT / "literature/openalex/page_1.json"))',
        source_url="https://openalex.org",
        version="2026-03",
    ),
    DataSource(
        name="Europe PMC",
        id="europe_pmc",
        category="Literature",
        path="literature/europe_pmc/",
        format="json",
        size="92 MB",
        description="Gene therapy / CRISPR articles via REST API + PMC ID list.",
        key_columns=["id", "title", "authorString", "journalTitle", "pmid", "pmcid"],
        load_hint='import json; data = json.load(open(DATA_ROOT / "literature/europe_pmc/page_1.json"))',
        source_url="https://europepmc.org",
        version="2026-03",
    ),
    DataSource(
        name="bioRxiv / medRxiv",
        id="biorxiv_medrxiv",
        category="Literature",
        path="literature/biorxiv_medrxiv/",
        format="json",
        size="12 MB",
        description="Preprint metadata + abstracts (2023-2026).",
        key_columns=["doi", "title", "authors", "abstract", "category", "posted_date"],
        load_hint='import json; data = json.load(open(DATA_ROOT / "literature/biorxiv_medrxiv/biorxiv_recent.json"))',
        source_url="https://www.biorxiv.org",
        version="2026-03",
    ),
    DataSource(
        name="Semantic Scholar",
        id="semantic_scholar",
        category="Literature",
        path="literature/semantic_scholar/",
        format="json",
        size="928 KB",
        description="Biology papers via search API (50 pages).",
        key_columns=["paperId", "title", "year", "citationCount", "abstract"],
        load_hint='import json; data = json.load(open(DATA_ROOT / "literature/semantic_scholar/page_1.json"))',
        source_url="https://www.semanticscholar.org",
        version="2026-03",
    ),

    # ===== EXPRESSION =====
    DataSource(
        name="HPA",
        id="hpa",
        category="Expression",
        path="expression/hpa/",
        format="tsv",
        size="847 MB",
        description="Human Protein Atlas v23: normal tissue, pathology, subcellular, RNA tissue, single cell expression.",
        key_columns=["Gene", "Gene_name", "Tissue", "Cell_type", "Level", "Reliability"],
        load_hint='pd.read_csv(DATA_ROOT / "expression/hpa/normal_tissue.tsv", sep="\\t")',
        source_url="https://www.proteinatlas.org",
        version="v23",
    ),
    DataSource(
        name="TCGA",
        id="tcga",
        category="Expression",
        path="expression/tcga/",
        format="tsv",
        size="405 MB",
        description="GDC open-access STAR-Counts expression files (100 samples).",
        key_columns=["gene_id", "gene_name", "gene_type", "unstranded", "stranded_first", "stranded_second"],
        load_hint='pd.read_csv(DATA_ROOT / "expression/tcga/<sample>.tsv", sep="\\t", comment="#")',
        source_url="https://portal.gdc.cancer.gov",
        version="2026-03",
    ),
    DataSource(
        name="Cell Model Passports",
        id="cell_model_passports",
        category="Expression",
        path="expression/cell_model_passports/",
        format="json",
        size="2.8 MB",
        description="Sanger cell line metadata: identifiers, tissue, cancer type, mutations.",
        key_columns=["model_id", "tissue", "cancer_type", "sample_id"],
        load_hint='import json; data = json.load(open(DATA_ROOT / "expression/cell_model_passports/models.json"))',
        source_url="https://cellmodelpassports.sanger.ac.uk",
        version="2026-03",
    ),

    # ===== CANCER / DEPENDENCY =====
    DataSource(
        name="DepMap Proteomics",
        id="depmap_proteomics",
        category="Cancer",
        path="depmap/depmap/25Q3/",
        format="csv",
        size="61 MB",
        description="Harmonized mass-spec proteomics (Gygi lab, CCLE) 24Q4. Also accessible via existing DepMap tools.",
        key_columns=["cell_line", "protein", "abundance"],
        load_hint='pd.read_csv(DATA_ROOT / "depmap/depmap/25Q3/OmicsProteomicsMatrix.csv", index_col=0)',
        source_url="https://depmap.org",
        version="25Q3",
    ),
    DataSource(
        name="CPTAC",
        id="cptac",
        category="Cancer",
        path="cancer/cptac/",
        format="csv",
        size="18 MB",
        description="Clinical Proteomic Tumor Analysis Consortium: BRCA log2 ratio via PDC API.",
        key_columns=["gene", "log2_ratio", "tumor_type"],
        load_hint='pd.read_csv(DATA_ROOT / "cancer/cptac/cptac_brca_proteomics.csv")',
        source_url="https://proteomic.datacommons.cancer.gov",
        version="2026-03",
    ),
    DataSource(
        name="Gygi Proteomics",
        id="gygi_proteomics",
        category="Cancer",
        path="cancer/proteomics_gygi/",
        format="csv",
        size="68 MB",
        description="Normalized protein quantification matrix from Gygi lab.",
        key_columns=["protein", "cell_line", "LFC"],
        load_hint='pd.read_csv(DATA_ROOT / "cancer/proteomics_gygi/proteomics_gygi.csv", index_col=0)',
        source_url="https://gygi.hms.harvard.edu",
        version="2026-03",
    ),
    DataSource(
        name="PRISM 19Q4",
        id="prism_19q4",
        category="Cancer",
        path="depmap/depmap/prism/repurposing_19Q4/",
        format="csv",
        size="132 MB",
        description="Drug repurposing screen: primary/secondary screen LFC + treatment info. Also accessible via existing PRISM tools.",
        key_columns=["pert_name", "pert_dose", "ccle_name", "LFC"],
        load_hint='pd.read_csv(DATA_ROOT / "depmap/depmap/prism/repurposing_19Q4/secondary-screen-dose-response-curve-parameters.csv")',
        source_url="https://depmap.org/repurposing/",
        version="19Q4",
    ),

    # ===== CRISPR TOOLS =====
    DataSource(
        name="CRISPOR",
        id="crispor",
        category="CRISPR",
        path="crispr_tools/crispor/",
        format="script",
        size="372 KB",
        description="Guide RNA design CLI tool (Python script).",
        key_columns=[],
        load_hint="Python script — run via subprocess or import",
        source_url="http://crispor.tefor.net",
        version="latest",
    ),
    DataSource(
        name="Cas-OFFinder",
        id="cas_offinder",
        category="CRISPR",
        path="crispr_tools/cas_offinder/",
        format="binary",
        size="16 KB",
        description="Off-target search tool (requires OpenCL runtime).",
        key_columns=[],
        load_hint="Binary tool — run via subprocess",
        source_url="http://www.rgenome.net/cas-offinder/",
        version="latest",
    ),
    DataSource(
        name="GuideScan2",
        id="guidescan2",
        category="CRISPR",
        path="crispr_tools/guidescan2/",
        format="metadata",
        size="8 KB",
        description="Pre-computed CRISPR guide database metadata.",
        key_columns=[],
        load_hint="Database metadata files",
        source_url="https://guidescan.com",
        version="2.0",
    ),

    # ===== OTHER =====
    DataSource(
        name="BioThings",
        id="biothings",
        category="Other",
        path="biothings/",
        format="json",
        size="1.8 MB",
        description="MyGene.info human gene annotations (10K genes): NCBI, UniProt, GO, KEGG per gene.",
        key_columns=["symbol", "name", "entrezgene", "uniprot", "go", "pathway"],
        load_hint='import json, glob; files = glob.glob(str(DATA_ROOT / "biothings/*.json")); data = json.load(open(files[0]))',
        source_url="https://mygene.info",
        version="2026-03",
    ),

    # ===== PRE-EXISTING (already accessible via existing tools) =====
    DataSource(
        name="CellxGene",
        id="cellxgene",
        category="Pre-existing",
        path="cellxgene/",
        format="h5ad",
        size="~100 GB",
        description="1,265 single-cell H5AD datasets from CZ CELLxGENE Census. (Also accessible via existing cellxgene.query tool)",
        key_columns=["cell_type", "tissue", "gene", "expression"],
        load_hint='import scanpy as sc; adata = sc.read_h5ad(DATA_ROOT / "cellxgene/<dataset>.h5ad")',
        source_url="https://cellxgene.cziscience.com",
        version="2026-03",
    ),
    DataSource(
        name="ChEMBL 36",
        id="chembl",
        category="Pre-existing",
        path="chembl/chembl/36/",
        format="sqlite",
        size="~11 GB",
        description="Bioactivity database: SQLite, SDF, FASTA. (Also accessible via existing literature.chembl_query tool)",
        key_columns=["molecule_chembl_id", "target_chembl_id", "standard_value", "standard_type"],
        load_hint='import sqlite3; conn = sqlite3.connect(DATA_ROOT / "chembl/chembl/36/chembl_36.db")',
        source_url="https://www.ebi.ac.uk/chembl/",
        version="36",
    ),
    DataSource(
        name="DepMap 25Q3",
        id="depmap_25q3",
        category="Pre-existing",
        path="depmap/depmap/25Q3/",
        format="csv",
        size="~36 GB",
        description="CRISPR gene effect, expression, mutations, fusions, copy number. (Also accessible via existing target/viability tools)",
        key_columns=["ModelID", "gene", "gene_effect"],
        load_hint='pd.read_csv(DATA_ROOT / "depmap/depmap/25Q3/CRISPRGeneEffect.csv", index_col=0)',
        source_url="https://depmap.org",
        version="25Q3",
    ),
    DataSource(
        name="GEO",
        id="geo",
        category="Pre-existing",
        path="geo/",
        format="various",
        size="variable",
        description="Gene Expression Omnibus bulk series data. (Also accessible via existing omics.geo_search/geo_fetch tools)",
        key_columns=[],
        load_hint="Use existing omics.geo_search and omics.geo_fetch tools",
        source_url="https://www.ncbi.nlm.nih.gov/geo/",
        version="various",
    ),
    DataSource(
        name="PMC",
        id="pmc",
        category="Pre-existing",
        path="pmc/",
        format="various",
        size="variable",
        description="PubMed Central open-access full-text articles. (Also accessible via existing literature tools)",
        key_columns=[],
        load_hint="Use existing literature.pubmed_search tool",
        source_url="https://www.ncbi.nlm.nih.gov/pmc/",
        version="various",
    ),
    DataSource(
        name="Perturb-seq",
        id="perturb_seq",
        category="Pre-existing",
        path="perturb_seq_assorted/",
        format="h5ad",
        size="variable",
        description="scPerturb v1.4 + PerturbAtlas: 54 H5ADs + 2,066 experiments.",
        key_columns=["gene", "perturbation", "cell_type", "expression"],
        load_hint='import scanpy as sc; adata = sc.read_h5ad(DATA_ROOT / "scperturb/<dataset>.h5ad")',
        source_url="https://scperturb.org",
        version="v1.4",
    ),
    DataSource(
        name="L1000/LINCS",
        id="l1000_lincs",
        category="Pre-existing",
        path="lincs/",
        format="gctx",
        size="variable",
        description="LINCS gene expression connectivity map: Level 4+5 GCTX + metadata. (Also accessible via existing expression.l1000_similarity tool)",
        key_columns=["compound", "gene", "z_score"],
        load_hint="Use existing expression.l1000_similarity tool",
        source_url="https://clue.io",
        version="GSE92742",
    ),
]


# ---------------------------------------------------------------------------
# Catalog accessors
# ---------------------------------------------------------------------------

def get_catalog() -> list[DataSource]:
    """Return the full data catalog."""
    return CATALOG


def get_source_by_id(source_id: str) -> Optional[DataSource]:
    """Look up a data source by its ID."""
    for src in CATALOG:
        if src.id == source_id:
            return src
    return None


# ---------------------------------------------------------------------------
# System prompt rendering
# ---------------------------------------------------------------------------

_CATEGORY_ORDER = [
    "Genomic", "Open Targets", "Pathways", "Networks", "Epigenomics",
    "Protein", "Drug", "Metabolic", "Mouse Models", "Safety",
    "Clinical", "Literature", "Expression", "Cancer", "CRISPR",
    "Other", "Pre-existing",
]


def format_for_prompt(data_root: str = "DATA_ROOT") -> str:
    """Render the catalog as compact text for system prompt injection.

    Returns a ~3K token text block grouped by category.
    """
    lines: list[str] = []
    lines.append(
        f"You have access to a local data lake with {len(CATALOG)} curated "
        f"biological datasets via {data_root} (a Path variable in run_python). "
        f"Use run_python to load and analyze any of these directly.\n"
    )

    # Group by category
    by_cat: dict[str, list[DataSource]] = {}
    for src in CATALOG:
        by_cat.setdefault(src.category, []).append(src)

    for cat in _CATEGORY_ORDER:
        sources = by_cat.get(cat, [])
        if not sources:
            continue

        lines.append(f"### {cat}")
        for src in sources:
            cols = ", ".join(src.key_columns[:6]) if src.key_columns else ""
            cols_str = f"\n  Cols: {cols}" if cols else ""
            lines.append(
                f"- **{src.name}** `{src.path}` ({src.format}, {src.size})\n"
                f"  {src.description}{cols_str}"
            )
        lines.append("")

    # Loading patterns
    lines.append("### Loading Patterns")
    lines.append(
        "- TSV/CSV: `pd.read_csv(DATA_ROOT / \"path\", sep=\"\\t\")`\n"
        "- Parquet: `pd.read_parquet(DATA_ROOT / \"path/\")`\n"
        "- Tabix BED: `import pysam; tbx = pysam.TabixFile(str(DATA_ROOT / \"path\")); list(tbx.fetch(\"chr1\", start, end))`\n"
        "- VCF: `import pysam; vcf = pysam.VariantFile(str(DATA_ROOT / \"path\"))`\n"
        "- GMT: Each line = `name\\tdescription\\tgene1\\tgene2\\t...`\n"
        "- JSON: `json.load(open(DATA_ROOT / \"path\"))`\n"
        "- H5AD: `import scanpy as sc; adata = sc.read_h5ad(DATA_ROOT / \"path\")`"
    )

    return "\n".join(lines)


def get_data_catalog_prompt(config) -> Optional[str]:
    """Get the data catalog prompt text if data.base is configured.

    Args:
        config: ct Config instance.

    Returns:
        Catalog prompt text, or None if data.base is not configured.
    """
    data_base = config.get("data.base") if config else None
    if not data_base:
        return None
    data_path = Path(data_base)
    if not data_path.exists():
        return None
    return format_for_prompt("DATA_ROOT")
