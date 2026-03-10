"""OpenFold2 structure prediction."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

OPENFOLD_DIR = "/opt/openfold"
AA_SEQUENCE_PATTERN = re.compile(r"^[ARNDCQEGHILKMFPSTWYV]+$")
MAX_SEQUENCE_LENGTH = 1000
ALLOWED_MODEL_IDS = {1, 2, 3, 4, 5}
ALIGNMENT_FILE_NAMES = {
    "uniref90": "uniref90_hits.a3m",
    "small_bfd": "small_bfd_hits.a3m",
    "mgnify": "mgnify_hits.a3m",
    "uniprot": "uniprot_hits.a3m",
    "pdb70": "pdb70_hits.hhr",
}


def _get_gpu_vram_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0


def _monitor_vram(stop_event, results) -> None:
    peak = 0
    while not stop_event.is_set():
        vram = _get_gpu_vram_mb()
        if vram > peak:
            peak = vram
        results["peak"] = peak
        stop_event.wait(0.5)
    vram = _get_gpu_vram_mb()
    if vram > peak:
        results["peak"] = vram


def _normalize_alignment_entry(db_name: str, payload: Any) -> dict:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Alignment entries must be objects keyed by format.")
    normalized = {}
    for fmt_name, fmt_payload in payload.items():
        if fmt_name not in {"a3m", "hhr"}:
            raise ValueError(f"Unsupported alignment format `{fmt_name}` for `{db_name}`.")
        if not isinstance(fmt_payload, dict):
            raise ValueError("Alignment format payloads must be objects.")
        alignment = str(fmt_payload.get("alignment", "")).strip()
        declared_format = str(fmt_payload.get("format", "")).strip().lower()
        if not alignment:
            raise ValueError("Alignment payloads require non-empty `alignment` text.")
        if declared_format != fmt_name:
            raise ValueError(f"Alignment payload format mismatch for `{db_name}` / `{fmt_name}`.")
        normalized[fmt_name] = {"alignment": alignment, "format": declared_format}
    return normalized


def normalize_args(args: dict) -> dict:
    normalized = dict(args)
    sequence = str(normalized.get("sequence", "")).strip().upper()

    if "sequences" in normalized:
        raise ValueError(
            "OpenFold2 accepts only `sequence`. Use `structure.alphafold2_multimer` for multi-chain inputs."
        )
    if not sequence:
        raise ValueError("OpenFold2 requires `sequence` as a non-empty amino acid sequence string.")
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"OpenFold2 `sequence` must be at most {MAX_SEQUENCE_LENGTH} residues.")
    if not AA_SEQUENCE_PATTERN.fullmatch(sequence):
        raise ValueError("OpenFold2 `sequence` must contain only valid amino acid IUPAC symbols.")

    alignments = normalized.get("alignments") or {}
    if alignments and not isinstance(alignments, dict):
        raise ValueError("OpenFold2 `alignments` must be an object keyed by database name.")
    normalized_alignments = {
        str(db_name).strip().lower(): _normalize_alignment_entry(str(db_name).strip().lower(), payload)
        for db_name, payload in alignments.items()
    }

    selected_models = normalized.get("selected_models") or [1]
    if not isinstance(selected_models, list) or not selected_models:
        raise ValueError("OpenFold2 `selected_models` must be a non-empty list of model ids.")
    clean_models = []
    for model_id in selected_models:
        try:
            model_int = int(model_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("OpenFold2 `selected_models` entries must be integers.") from exc
        if model_int not in ALLOWED_MODEL_IDS:
            raise ValueError("OpenFold2 `selected_models` must contain values from 1 to 5.")
        if model_int not in clean_models:
            clean_models.append(model_int)

    normalized["sequence"] = sequence
    normalized["alignments"] = normalized_alignments
    normalized["selected_models"] = clean_models
    normalized["relax_prediction"] = bool(normalized.get("relax_prediction", False))
    return normalized


def _ensure_params_file(model_name: str) -> str:
    params_dir = os.environ.get("OPENFOLD_PARAMS_DIR", "/root/.cache/openfold/params")
    params_file = os.path.join(params_dir, f"params_{model_name}.npz")
    if not os.path.isfile(params_file):
        os.makedirs(params_dir, exist_ok=True)
        dl_script = os.path.join(OPENFOLD_DIR, "scripts", "download_alphafold_params.sh")
        parent = os.path.dirname(params_dir)
        subprocess.run(["bash", dl_script, parent], check=True, timeout=600)
    return params_file


def _write_alignments(align_dir: str, alignments: dict) -> None:
    os.makedirs(align_dir, exist_ok=True)
    for db_name, payload in alignments.items():
        for fmt_name, fmt_payload in payload.items():
            default_name = ALIGNMENT_FILE_NAMES.get(db_name)
            if default_name is None:
                default_name = f"{db_name}_hits.{fmt_name}"
            elif not default_name.endswith(f".{fmt_name}"):
                default_name = f"{db_name}_hits.{fmt_name}"
            with open(os.path.join(align_dir, default_name), "w", encoding="utf-8") as fh:
                fh.write(fmt_payload["alignment"])


def _extract_prediction(out_dir: str) -> tuple[str, float]:
    pdb_content = ""
    confidence = 0.0
    for root, _, files in os.walk(out_dir):
        for filename in sorted(files):
            if filename.endswith(".pdb"):
                with open(os.path.join(root, filename), encoding="utf-8") as fh:
                    pdb_content = fh.read()
                bfactors = []
                for line in pdb_content.split("\n"):
                    if line.startswith("ATOM") and " CA " in line:
                        try:
                            bfactors.append(float(line[60:66].strip()))
                        except (ValueError, IndexError):
                            pass
                if bfactors:
                    confidence = sum(bfactors) / len(bfactors)
                return pdb_content, confidence
    return "", 0.0


def run(sequence: str = "", session_id: str = "", **kwargs) -> dict:
    try:
        normalized = normalize_args({"sequence": sequence, "session_id": session_id, **kwargs})
    except ValueError as exc:
        return {"summary": f"Error: {exc}", "error": "invalid_sequence_input"}

    sequence = normalized["sequence"]
    alignments = normalized["alignments"]
    selected_models = normalized["selected_models"]
    relax_prediction = normalized["relax_prediction"]
    session_id = str(normalized.get("session_id", ""))

    t0 = time.time()
    seq_len = len(sequence)
    vram_before = _get_gpu_vram_mb()

    stop_event = threading.Event()
    vram_results = {"peak": vram_before}
    monitor = threading.Thread(target=_monitor_vram, args=(stop_event, vram_results), daemon=True)
    monitor.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_dir = os.path.join(tmpdir, "fasta")
        os.makedirs(fasta_dir)
        with open(os.path.join(fasta_dir, "query.fasta"), "w", encoding="utf-8") as fh:
            fh.write(f">query\n{sequence}\n")

        template_dir = os.path.join(tmpdir, "templates")
        os.makedirs(template_dir)
        with open(os.path.join(template_dir, "dummy.cif"), "w", encoding="utf-8") as fh:
            fh.write("data_dummy\n")

        align_root = os.path.join(tmpdir, "alignments")
        query_align_dir = os.path.join(align_root, "query")
        os.makedirs(query_align_dir)
        if alignments:
            _write_alignments(query_align_dir, alignments)

        predictions = []
        t_inference = time.time()
        for model_id in selected_models:
            out_dir = os.path.join(tmpdir, f"output_model_{model_id}")
            os.makedirs(out_dir, exist_ok=True)
            model_name = f"model_{model_id}"
            params_file = _ensure_params_file(model_name)
            cmd = [
                sys.executable,
                os.path.join(OPENFOLD_DIR, "run_pretrained_openfold.py"),
                fasta_dir,
                template_dir,
                "--output_dir",
                out_dir,
                "--model_device",
                "cuda:0",
                "--config_preset",
                model_name,
                "--jax_param_path",
                params_file,
            ]
            cmd.extend(["--use_precomputed_alignments", align_root])
            if not alignments:
                cmd.append("--use_single_seq_mode")
            if not relax_prediction:
                cmd.append("--skip_relaxation")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=OPENFOLD_DIR)
            if result.returncode != 0:
                stop_event.set()
                monitor.join(timeout=2)
                error_tail = f"STDERR:\n{result.stderr[-2000:]}\nSTDOUT:\n{result.stdout[-2000:]}"
                return {
                    "summary": f"Error: OpenFold2 failed on {model_name}: {error_tail}",
                    "error": error_tail,
                    "command": cmd,
                    "metrics": {
                        "vram_before_mb": vram_before,
                        "vram_peak_mb": vram_results["peak"],
                        "time_total_s": round(time.time() - t0, 2),
                    },
                }

            pdb_content, confidence = _extract_prediction(out_dir)
            if not pdb_content:
                stop_event.set()
                monitor.join(timeout=2)
                return {
                    "summary": f"OpenFold2 ran but produced no structure for {model_name}.",
                    "error": "no_output",
                }
            predictions.append(
                {
                    "model": model_name,
                    "confidence": confidence,
                    "pdb_content": pdb_content[:5000],
                }
            )

        t_inference = time.time() - t_inference
        stop_event.set()
        monitor.join(timeout=2)
        vram_after = _get_gpu_vram_mb()
        vram_peak = vram_results["peak"]

        predictions.sort(key=lambda item: item["confidence"], reverse=True)
        best = predictions[0]

        if session_id:
            workspace_dir = f"/vol/workspace/{session_id}"
            os.makedirs(workspace_dir, exist_ok=True)
            with open(f"{workspace_dir}/predicted_structure.pdb", "w", encoding="utf-8") as fh:
                fh.write(best["pdb_content"])

        return {
            "summary": (
                f"OpenFold2 prediction for {seq_len}-residue protein using {len(predictions)} model(s). "
                f"Best pLDDT: {best['confidence']:.1f}/100."
            ),
            "pdb_content": best["pdb_content"],
            "confidence": best["confidence"],
            "num_residues": seq_len,
            "predictions": predictions,
            "metrics": {
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after,
                "vram_peak_mb": vram_peak,
                "time_inference_s": round(t_inference, 2),
                "time_total_s": round(time.time() - t0, 2),
            },
        }
