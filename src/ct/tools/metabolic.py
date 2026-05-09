"""
Metabolic tools: flux balance analysis using COBRApy with Recon3D/Human-GEM models.
"""

import logging
from pathlib import Path

from ct.tools import registry, check_dependency

logger = logging.getLogger("ct.tools.metabolic")


@registry.register(
    name="metabolic.flux_balance",
    description="Predict metabolic consequences of gene knockout using flux balance analysis (FBA). Loads Recon3D or Human-GEM genome-scale metabolic model, simulates gene knockout, reports growth rate change and affected metabolic fluxes.",
    category="metabolic",
    parameters={
        "gene": "Gene symbol to knock out (e.g. PCSK9, HK1, LDHA)",
        "model": "Metabolic model to use: 'recon3d' or 'human_gem' (default: recon3d)",
    },
    requires_packages=["cobra"],
    requires_data=["recon3d"],
    usage_guide="Predict what happens metabolically when a gene is knocked out. Uses constraint-based metabolic modeling (FBA). Important for assessing metabolic side effects of gene editing.",
)
def flux_balance(gene: str, model: str = "recon3d", **kwargs) -> dict:
    """Run flux balance analysis for a gene knockout."""
    # Check dependencies
    dep_error = check_dependency(packages=["cobra"])
    if dep_error:
        return dep_error

    import cobra

    gene = str(gene).strip().upper()

    # Resolve model path
    from ct.agent.config import Config
    cfg = Config.load()
    data_base = cfg.get("data.base")

    model_paths = {
        "recon3d": Path(data_base) / "gene_context" / "metabolic" / "recon3d" / "Recon3D.json.gz",
        "human_gem": Path(data_base) / "gene_context" / "metabolic" / "human_metabolic_atlas" / "HumanGEM.yml",
    }

    model_name = model.lower().replace("-", "_").replace(" ", "_")
    model_path = model_paths.get(model_name)
    if model_path is None:
        return {
            "error": f"Unknown model: {model}",
            "summary": f"Model '{model}' not recognized. Available: recon3d, human_gem",
        }

    if not model_path.exists():
        # Try alternative file names
        alt_paths = list(model_path.parent.glob("*.json")) + list(model_path.parent.glob("*.xml"))
        if alt_paths:
            model_path = alt_paths[0]
        else:
            return {
                "error": f"Model file not found: {model_path}",
                "summary": f"Metabolic model not found at {model_path}. Set data.base to your bronze directory.",
            }

    try:
        logger.info("Loading metabolic model from %s", model_path)
        import gzip as _gzip
        fname = model_path.name
        if fname.endswith(".json.gz"):
            with _gzip.open(str(model_path), "rt") as f:
                cobra_model = cobra.io.load_json_model(f)
        elif fname.endswith(".json"):
            cobra_model = cobra.io.load_json_model(str(model_path))
        elif fname.endswith(".xml.gz") or fname.endswith(".sbml.gz"):
            cobra_model = cobra.io.read_sbml_model(str(model_path))
        elif model_path.suffix in (".xml", ".sbml"):
            cobra_model = cobra.io.read_sbml_model(str(model_path))
        elif model_path.suffix in (".yml", ".yaml"):
            cobra_model = cobra.io.load_yaml_model(str(model_path))
        else:
            return {"error": f"Unsupported model format: {model_path.suffix}", "summary": "Use JSON, XML, or YAML model files."}
    except Exception as e:
        return {"error": f"Failed to load model: {e}", "summary": f"Error loading metabolic model: {e}"}

    # Find the gene in the model
    gene_lower = gene.lower()
    model_genes = {g.id: g for g in cobra_model.genes}
    # Try exact match, then case-insensitive
    target_gene = model_genes.get(gene) or model_genes.get(gene_lower)
    if target_gene is None:
        # Try partial match
        candidates = [g for g in cobra_model.genes if gene_lower in g.id.lower() or gene_lower in g.name.lower()]
        if candidates:
            target_gene = candidates[0]
        else:
            return {
                "summary": f"Gene {gene} not found in {model_name} model ({len(cobra_model.genes)} genes). The gene may not have a direct metabolic role.",
                "source": f"{model_name} genome-scale metabolic model",
                "source_file": str(model_path.relative_to(Path(data_base))),
                "query": {"gene": gene, "model": model_name},
                "gene_found": False,
                "model_genes": len(cobra_model.genes),
            }

    # Run FBA: wild-type first
    try:
        wt_solution = cobra_model.optimize()
        wt_growth = wt_solution.objective_value
    except Exception as e:
        return {"error": f"FBA failed: {e}", "summary": f"Flux balance analysis failed: {e}"}

    # Knock out the gene
    with cobra_model:
        target_gene.knock_out()
        try:
            ko_solution = cobra_model.optimize()
            ko_growth = ko_solution.objective_value
        except Exception:
            ko_growth = 0.0

    # Compute affected reactions
    affected_reactions = []
    for rxn in target_gene.reactions:
        affected_reactions.append({
            "reaction_id": rxn.id,
            "reaction_name": rxn.name,
            "subsystem": rxn.subsystem,
            "wt_flux": wt_solution.fluxes.get(rxn.id, 0.0),
        })

    growth_change = ko_growth - wt_growth
    growth_pct = (growth_change / wt_growth * 100) if wt_growth != 0 else 0

    return {
        "summary": (
            f"FBA for {gene} knockout in {model_name}: "
            f"growth rate {wt_growth:.4f} → {ko_growth:.4f} "
            f"({growth_pct:+.1f}% change). "
            f"{len(affected_reactions)} reactions affected."
        ),
        "source": f"{model_name} genome-scale metabolic model via COBRApy",
        "source_file": str(model_path.relative_to(Path(data_base))),
        "query": {"gene": gene, "model": model_name},
        "gene_found": True,
        "gene_id": target_gene.id,
        "wt_growth_rate": round(wt_growth, 6),
        "ko_growth_rate": round(ko_growth, 6),
        "growth_change_pct": round(growth_pct, 2),
        "affected_reactions": affected_reactions,
        "model_genes": len(cobra_model.genes),
        "model_reactions": len(cobra_model.reactions),
    }
