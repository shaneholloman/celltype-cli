"""Local MMseqs2 MSA search implementation.

Runs a full ColabFold-style homology search against local UniRef30 and
ColabFold EnvDB databases using the mmseqs binary.

Database location is read from the COLABFOLD_DB environment variable
(default: /home/ubuntu/.cache/colabfold_db for local runs,
/vol/colabfold_db for Modal/Docker).

The search pipeline mirrors ColabFold's mmseqs_search_monomer():
  query FASTA -> createdb -> search (UniRef30, 3 iters) -> expandaln ->
  align -> filterresult -> result2msa -> (same for EnvDB) -> mergedbs -> unpack

Falls back to the ColabFold API when local databases are not available.
"""

import io
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

UNIREF_DB = "uniref30_2302_db"
ENV_DB = "colabfold_envdb_202108_db"

DEFAULT_DB_PATHS = [
    "/vol/colabfold_db",
    "/home/ubuntu/.cache/colabfold_db",
    os.path.expanduser("~/.cache/colabfold_db"),
]


def _find_db_base() -> Path:
    explicit = os.environ.get("COLABFOLD_DB")
    if explicit:
        p = Path(explicit)
        if p.joinpath(f"{UNIREF_DB}.dbtype").exists():
            return p
        raise FileNotFoundError(f"COLABFOLD_DB={explicit} does not contain {UNIREF_DB}.dbtype")

    for candidate in DEFAULT_DB_PATHS:
        p = Path(candidate)
        if p.joinpath(f"{UNIREF_DB}.dbtype").exists():
            return p

    raise FileNotFoundError(
        "ColabFold databases not found. Set COLABFOLD_DB env var or place databases in "
        + " or ".join(DEFAULT_DB_PATHS)
    )


def _find_mmseqs() -> str:
    for candidate in ["mmseqs", "/usr/local/bin/mmseqs", "/opt/mmseqs/bin/mmseqs"]:
        if shutil.which(candidate):
            return candidate
    raise FileNotFoundError("mmseqs binary not found in PATH")


def _run_mmseqs(mmseqs: str, params: list, check_exists: str = None):
    """Run an mmseqs subcommand. Skip if output already exists."""
    if check_exists and Path(check_exists).with_suffix(".dbtype").exists():
        logger.info(f"Skipping {params[0]}: {check_exists} already exists")
        return

    os.environ["MMSEQS_FORCE_MERGE"] = "1"
    os.environ["MMSEQS_CALL_DEPTH"] = "1"
    cmd = [mmseqs] + [str(p) for p in params]
    logger.info(f"mmseqs {' '.join(str(p) for p in params[:3])}...")
    subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _search_monomer(
    mmseqs: str,
    dbbase: Path,
    workdir: Path,
    use_env: bool = True,
    threads: int = 32,
    db_load_mode: int = 0,
    prefilter_mode: int = 1,
    sensitivity: float = 8.0,
):
    """ColabFold monomer search pipeline — mirrors mmseqs_search_monomer()."""
    uniref_db = UNIREF_DB
    env_db = ENV_DB

    has_index = dbbase.joinpath(f"{uniref_db}.idx").is_file() or dbbase.joinpath(f"{uniref_db}.idx.index").is_file()
    if not has_index:
        db_load_mode = 0
        sfx_seq, sfx_aln = "_seq", "_aln"
    else:
        sfx_seq, sfx_aln = ".idx", ".idx"

    # ColabFold filter defaults
    align_eval = 10
    qsc = 0.8
    max_accept = 100000

    search_param = [
        "--num-iterations", "3",
        "--db-load-mode", str(db_load_mode),
        "-a", "-e", "0.1",
        "--max-seqs", "10000",
        "--prefilter-mode", str(prefilter_mode),
        "-s", f"{sensitivity:.1f}",
    ]
    filter_param = [
        "--filter-msa", "1",
        "--filter-min-enable", "1000",
        "--diff", "3000",
        "--qid", "0.0,0.2,0.4,0.6,0.8,1.0",
        "--qsc", "0",
        "--max-seq-id", "0.95",
    ]
    expand_param = [
        "--expansion-mode", "0",
        "-e", "inf",
        "--expand-filter-clusters", "1",
        "--max-seq-id", "0.95",
    ]

    W = workdir
    DB = dbbase

    # --- UniRef30 search ---
    _run_mmseqs(mmseqs, [
        "search", W / "qdb", DB / uniref_db, W / "res", W / "tmp",
        "--threads", str(threads),
    ] + search_param, check_exists=str(W / "uniref.a3m"))

    _run_mmseqs(mmseqs, ["mvdb", W / "tmp/latest/profile_1", W / "prof_res"])
    _run_mmseqs(mmseqs, ["lndb", W / "qdb_h", W / "prof_res_h"])

    _run_mmseqs(mmseqs, [
        "expandaln", W / "qdb", DB / f"{uniref_db}{sfx_seq}",
        W / "res", DB / f"{uniref_db}{sfx_aln}", W / "res_exp",
        "--db-load-mode", str(db_load_mode), "--threads", str(threads),
    ] + expand_param, check_exists=str(W / "uniref.a3m"))

    _run_mmseqs(mmseqs, [
        "align", W / "prof_res", DB / f"{uniref_db}{sfx_seq}",
        W / "res_exp", W / "res_exp_realign",
        "--db-load-mode", str(db_load_mode),
        "-e", str(align_eval), "--max-accept", str(max_accept),
        "--threads", str(threads), "--alt-ali", "10", "-a",
    ], check_exists=str(W / "uniref.a3m"))

    _run_mmseqs(mmseqs, [
        "filterresult", W / "qdb", DB / f"{uniref_db}{sfx_seq}",
        W / "res_exp_realign", W / "res_exp_realign_filter",
        "--db-load-mode", str(db_load_mode),
        "--qid", "0", "--qsc", str(qsc), "--diff", "0",
        "--threads", str(threads), "--max-seq-id", "1.0",
        "--filter-min-enable", "100",
    ], check_exists=str(W / "uniref.a3m"))

    _run_mmseqs(mmseqs, [
        "result2msa", W / "qdb", DB / f"{uniref_db}{sfx_seq}",
        W / "res_exp_realign_filter", W / "uniref.a3m",
        "--msa-format-mode", "6",
        "--db-load-mode", str(db_load_mode), "--threads", str(threads),
    ] + filter_param)

    for db_name in ["res_exp_realign_filter", "res_exp_realign", "res_exp", "res"]:
        _run_mmseqs(mmseqs, ["rmdb", W / db_name])

    # --- EnvDB search ---
    if use_env:
        _run_mmseqs(mmseqs, [
            "search", W / "prof_res", DB / env_db, W / "res_env", W / "tmp3",
            "--threads", str(threads),
        ] + search_param, check_exists=str(W / "bfd.mgnify30.metaeuk30.smag30.a3m"))

        _run_mmseqs(mmseqs, [
            "expandaln", W / "prof_res", DB / f"{env_db}{sfx_seq}",
            W / "res_env", DB / f"{env_db}{sfx_aln}", W / "res_env_exp",
            "-e", "inf", "--expansion-mode", "0",
            "--db-load-mode", str(db_load_mode), "--threads", str(threads),
        ], check_exists=str(W / "bfd.mgnify30.metaeuk30.smag30.a3m"))

        _run_mmseqs(mmseqs, [
            "align", W / "tmp3/latest/profile_1", DB / f"{env_db}{sfx_seq}",
            W / "res_env_exp", W / "res_env_exp_realign",
            "--db-load-mode", str(db_load_mode),
            "-e", str(align_eval), "--max-accept", str(max_accept),
            "--threads", str(threads), "--alt-ali", "10", "-a",
        ], check_exists=str(W / "bfd.mgnify30.metaeuk30.smag30.a3m"))

        _run_mmseqs(mmseqs, [
            "filterresult", W / "qdb", DB / f"{env_db}{sfx_seq}",
            W / "res_env_exp_realign", W / "res_env_exp_realign_filter",
            "--db-load-mode", str(db_load_mode),
            "--qid", "0", "--qsc", str(qsc), "--diff", "0",
            "--max-seq-id", "1.0", "--threads", str(threads),
            "--filter-min-enable", "100",
        ], check_exists=str(W / "bfd.mgnify30.metaeuk30.smag30.a3m"))

        _run_mmseqs(mmseqs, [
            "result2msa", W / "qdb", DB / f"{env_db}{sfx_seq}",
            W / "res_env_exp_realign_filter", W / "bfd.mgnify30.metaeuk30.smag30.a3m",
            "--msa-format-mode", "6",
            "--db-load-mode", str(db_load_mode), "--threads", str(threads),
        ] + filter_param)

        for db_name in ["res_env_exp_realign_filter", "res_env_exp_realign", "res_env_exp", "res_env"]:
            _run_mmseqs(mmseqs, ["rmdb", W / db_name])

    # --- Merge ---
    if use_env:
        _run_mmseqs(mmseqs, [
            "mergedbs", W / "qdb", W / "final.a3m",
            W / "uniref.a3m", W / "bfd.mgnify30.metaeuk30.smag30.a3m",
        ])
        _run_mmseqs(mmseqs, ["rmdb", W / "bfd.mgnify30.metaeuk30.smag30.a3m"])
        _run_mmseqs(mmseqs, ["rmdb", W / "uniref.a3m"])
    else:
        _run_mmseqs(mmseqs, ["mvdb", W / "uniref.a3m", W / "final.a3m"])

    # Unpack to individual .a3m files
    _run_mmseqs(mmseqs, [
        "unpackdb", W / "final.a3m", W / ".",
        "--unpack-name-mode", "0", "--unpack-suffix", ".a3m",
    ])
    _run_mmseqs(mmseqs, ["rmdb", W / "final.a3m"])

    # Cleanup
    _run_mmseqs(mmseqs, ["rmdb", W / "prof_res"])
    _run_mmseqs(mmseqs, ["rmdb", W / "prof_res_h"])
    for tmp_dir in ["tmp", "tmp3"]:
        d = W / tmp_dir
        if d.exists():
            shutil.rmtree(d)


def _run_api_fallback(clean_seq: str, seq_len: int, database: str,
                      session_id: str, t0: float) -> dict:
    """Fallback to ColabFold API when local databases are not available."""
    import requests

    api_url = "https://api.colabfold.com"
    mode = "env" if database in ("all", "colabfold_envdb_202108", "colabfold_envdb") else "all"

    query = f">101\n{clean_seq}\n"
    t_submit = time.time()
    response = requests.post(
        f"{api_url}/ticket/msa", data={"q": query, "mode": mode}, timeout=30)
    response.raise_for_status()
    ticket = response.json()
    ticket_id = ticket.get("id", "")
    t_submit = time.time() - t_submit

    if not ticket_id:
        return {"summary": "Error: ColabFold API returned no ticket ID.", "error": "api_error"}

    t_search = time.time()
    for _ in range(120):
        status_resp = requests.get(f"{api_url}/ticket/{ticket_id}", timeout=10)
        status = status_resp.json()
        if status.get("status") == "COMPLETE":
            break
        elif status.get("status") == "ERROR":
            return {
                "summary": f"Error: ColabFold MSA search failed: {status.get('error', 'unknown')}",
                "error": "search_failed",
            }
        time.sleep(5)
    else:
        return {"summary": "Error: ColabFold MSA search timed out.", "error": "timeout"}
    t_search = time.time() - t_search

    t_download = time.time()
    result_resp = requests.get(f"{api_url}/result/download/{ticket_id}", timeout=60)
    result_resp.raise_for_status()

    msa_content = ""
    tar_data = io.BytesIO(result_resp.content)
    try:
        with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".a3m"):
                    f = tar.extractfile(member)
                    if f:
                        msa_content = f.read().decode("utf-8", errors="ignore")
                        break
    except tarfile.ReadError:
        msa_content = result_resp.text
    t_download = time.time() - t_download

    if not msa_content:
        return {"summary": "Error: No A3M alignment found.", "error": "no_alignment"}

    num_sequences = msa_content.count(">")
    if session_id:
        ws_dir = Path(f"/vol/workspace/{session_id}")
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "msa.a3m").write_text(msa_content)

    msa_clean = msa_content.encode("ascii", errors="ignore").decode("ascii")
    return {
        "summary": (
            f"MSA search (API fallback): found {num_sequences} homologous "
            f"sequences for {seq_len}-residue query."
        ),
        "msa": msa_clean,
        "num_sequences": num_sequences,
        "database": database,
        "query_length": seq_len,
        "msa_size_bytes": len(msa_content),
        "metrics": {
            "vram_before_mb": 0, "vram_peak_mb": 0,
            "time_submit_s": round(t_submit, 2),
            "time_search_s": round(t_search, 2),
            "time_download_s": round(t_download, 2),
            "time_total_s": round(time.time() - t0, 2),
            "hardware": "CPU-only (ColabFold API fallback)",
        },
    }


def run(sequence: str = "", database: str = "all",
        e_value: float = 0.0001, iterations: int = 1,
        session_id: str = "", **kwargs) -> dict:
    """Run local MMseqs2 MSA search against ColabFold databases.

    Falls back to the ColabFold API if local databases or mmseqs are unavailable.
    """
    if not sequence:
        return {"summary": "Error: No sequence provided.", "error": "no_sequence"}

    t0 = time.time()
    clean_seq = sequence.strip().upper().replace(" ", "").replace("\n", "")
    seq_len = len(clean_seq)

    try:
        mmseqs = _find_mmseqs()
        dbbase = _find_db_base()
    except FileNotFoundError:
        logger.info("Local databases/mmseqs not found, falling back to ColabFold API")
        try:
            return _run_api_fallback(clean_seq, seq_len, database, session_id, t0)
        except Exception as e:
            return {"summary": f"Error: API fallback also failed: {e}", "error": str(e)}

    use_env = database in ("all", "colabfold_envdb_202108", "colabfold_envdb")
    threads = min(os.cpu_count() or 8, 64)

    workdir = Path(tempfile.mkdtemp(prefix="msa_search_"))
    try:
        query_fasta = workdir / "query.fas"
        query_fasta.write_text(f">101\n{clean_seq}\n")

        t_createdb = time.time()
        _run_mmseqs(mmseqs, [
            "createdb", query_fasta, workdir / "qdb",
            "--shuffle", "0", "--dbtype", "1",
        ])
        t_createdb = time.time() - t_createdb

        # Write qdb.lookup for ColabFold compatibility
        (workdir / "qdb.lookup").write_text(f"0\t101\t0\n")

        t_search = time.time()
        _search_monomer(
            mmseqs=mmseqs,
            dbbase=dbbase,
            workdir=workdir,
            use_env=use_env,
            threads=threads,
        )
        t_search = time.time() - t_search

        a3m_file = workdir / "0.a3m"
        if not a3m_file.exists():
            return {
                "summary": "Error: MMseqs2 search produced no alignment output.",
                "error": "no_alignment",
            }

        msa_content = a3m_file.read_text()
        num_sequences = msa_content.count(">")

        if session_id:
            ws_dir = Path(f"/vol/workspace/{session_id}")
            ws_dir.mkdir(parents=True, exist_ok=True)
            (ws_dir / "msa.a3m").write_text(msa_content)

        msa_clean = msa_content.encode("ascii", errors="ignore").decode("ascii")

        return {
            "summary": (
                f"MSA search completed: found {num_sequences} homologous sequences "
                f"for {seq_len}-residue query."
                + (f" Searched UniRef30 + ColabFold EnvDB." if use_env else " Searched UniRef30 only.")
            ),
            "msa": msa_clean,
            "num_sequences": num_sequences,
            "database": database,
            "query_length": seq_len,
            "msa_size_bytes": len(msa_content),
            "metrics": {
                "vram_before_mb": 0,
                "vram_peak_mb": 0,
                "time_createdb_s": round(t_createdb, 2),
                "time_search_s": round(t_search, 2),
                "time_total_s": round(time.time() - t0, 2),
                "hardware": f"CPU-only (local MMseqs2, {threads} threads)",
            },
        }

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
        return {
            "summary": f"Error: MMseqs2 command failed: {e.cmd[1] if len(e.cmd) > 1 else 'unknown'} — {stderr[:500]}",
            "error": "mmseqs_error",
        }
    except Exception as e:
        return {
            "summary": f"Error: MSA search failed: {e}",
            "error": str(e),
        }
    finally:
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
