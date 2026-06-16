#!/usr/bin/env python3
"""
Scan a results directory and write per-(model, task) TOP-3 runs into final.txt.

Layout assumptions:

1) Any directory directly under results/ whose name looks like YYYY-MM-DD_HH-MM-SS
   is treated as:
       model = 'Llama2-7B-Instruct'
       task  = 'forget05'

2) Any directory under results/ whose name starts with 'tofu_' is treated as a
   model directory. Inside it, each subdirectory is a 'task'
   (e.g. forget10_train__retain90). Inside each task directory, we look for:

     * summary.txt / summarize.txt directly in the task directory, and
     * summary.txt / summarize.txt in any immediate subdirectories
       (e.g. dated run folders like 2025-11-28_19-58-13).

3) Any directory directly under results/ whose name starts with 'MUSE-' or 'MUSE_'
   (e.g. results/MUSE-news_target) is treated as:
       model = 'MUSE'
       task  = suffix after 'MUSE-' or 'MUSE_'

   For MUSE folders we recursively search *all* subfolders under the MUSE folder
   for summary.txt / summarize.txt.

   TOP-3 for MUSE is ranked by muse_assessment_score.

Standard (non-MUSE) scoring:
  retention2_avg  = avg(1 - Prob, 1 - ROUGE)
  composite_score = avg(retention2_avg, model_utility)

Outputs:
  results/final.txt with TOP-3 per (model, task) group, plus loss definitions.

LaTeX:
  - Standard blocks: existing 12-metric LaTeX lines (plus model_utility).
  - MUSE blocks: for each TOP row:
      \\ours{} & round(forget_verb*100,2) & round(forget_know*100,2)
              & round(privleak,2) & round(retain_know*100,2) \\\\
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------- Config -------------

DEFAULT_MIN_MODEL_UTILITY: float = 0.57

# ------------- Helpers -------------


def is_valid_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def get_agg_value(x: Any, default: float = float("nan")) -> float:
    """Some metrics are stored as {'agg_value': <float>}."""
    if isinstance(x, dict) and "agg_value" in x:
        return safe_float(x.get("agg_value"), default=default)
    return safe_float(x, default=default)


def fmt2(x: Any) -> str:
    """Format a float to two decimals, or 'nan'."""
    if not is_valid_number(x):
        return "nan"
    return f"{float(x):.2f}"


def fmt_pct2(x: Any) -> str:
    """Format a fraction as percent with 2 decimals, or 'nan'."""
    if not is_valid_number(x):
        return "nan"
    return f"{float(x) * 100.0:.2f}"


def get_rouge(d: Dict[str, Any]) -> float:
    for k in ("forget_Q_A_ROUGE", "forget_Q_A_Rouge", "forget_Q_A_rouge"):
        if k in d:
            return get_agg_value(d[k])
    return float("nan")


def get_prob(d: Dict[str, Any]) -> float:
    for k in ("forget_Q_A_Prob", "forget_Q_A_PROB", "forget_Q_A_prob"):
        if k in d:
            return get_agg_value(d[k])
    return float("nan")


def get_truth_ratio(d: Dict[str, Any]) -> float:
    return get_agg_value(d.get("forget_truth_ratio"))


LATEX_METRICS_ORDER: List[str] = [
    "one_minus_rouge",
    "one_minus_prob",
    "extraction_strength",
    "model_utility_real_authors_rougeL",
    "model_utility_real_authors_prob",
    "model_utility_real_authors_truth_ratio",
    "model_utility_world_facts_rougeL",
    "model_utility_world_facts_prob",
    "model_utility_world_facts_truth_ratio",
    "model_utility_retain_rougeL",
    "model_utility_retain_prob",
    "model_utility_retain_truth_ratio",
]


def compute_rows(
    summary_dict: Dict[str, Any],
    folder: str,
    file_name: str,
    model_name: str,
    task_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for iter_key, items in summary_dict.items():
        if not isinstance(iter_key, str) or not iter_key.startswith("iteration"):
            continue
        if not isinstance(items, list):
            continue

        for obj in items:
            if not isinstance(obj, dict):
                continue

            loss_id = obj.get("loss_id")
            topk_score = safe_float(obj.get("topk_score"))

            ous = obj.get("open_unlearning_summary", {}) or {}
            if not isinstance(ous, dict):
                ous = {}

            prob = get_prob(ous)
            rouge = get_rouge(ous)
            mu = get_agg_value(ous.get("model_utility"))
            truth = get_truth_ratio(ous)

            extraction_strength = get_agg_value(ous.get("extraction_strength"))
            privleak = get_agg_value(ous.get("privleak"))

            # Retain metrics
            retain_prob = get_agg_value(ous.get("model_utility_retain_prob"))
            retain_rouge = get_agg_value(ous.get("model_utility_retain_rougeL"))
            retain_truth = get_agg_value(ous.get("model_utility_retain_truth_ratio"))

            # Real authors metrics
            real_prob = get_agg_value(ous.get("model_utility_real_authors_prob"))
            real_rouge = get_agg_value(ous.get("model_utility_real_authors_rougeL"))
            real_truth = get_agg_value(ous.get("model_utility_real_authors_truth_ratio"))

            # World facts metrics
            wf_prob = get_agg_value(ous.get("model_utility_world_facts_prob"))
            wf_rouge = get_agg_value(ous.get("model_utility_world_facts_rougeL"))
            wf_truth = get_agg_value(ous.get("model_utility_world_facts_truth_ratio"))

            if is_valid_number(prob) and is_valid_number(rouge) and is_valid_number(mu):
                one_minus_prob = 1.0 - prob
                one_minus_rouge = 1.0 - rouge
                retention2 = (one_minus_prob + one_minus_rouge) / 2.0
                composite = (retention2 + mu) / 2.0
            else:
                retention2 = float("nan")
                composite = float("nan")
                one_minus_prob = float("nan")
                one_minus_rouge = float("nan")

            if (
                is_valid_number(prob)
                and is_valid_number(rouge)
                and is_valid_number(truth)
                and is_valid_number(mu)
            ):
                retention3 = ((1.0 - prob) + (1.0 - rouge) + truth) / 3.0
                composite3 = (retention3 + mu) / 2.0
            else:
                retention3 = float("nan")
                composite3 = float("nan")

            rows.append(
                {
                    "folder": folder,
                    "file": file_name,
                    "model": model_name,
                    "task": task_name,
                    "iteration": iter_key,
                    "loss_id": loss_id,
                    "topk_score": topk_score,
                    "forget_Q_A_Prob": prob,
                    "forget_Q_A_ROUGE": rouge,
                    "forget_truth_ratio": truth,
                    "model_utility": mu,
                    "extraction_strength": extraction_strength,
                    "privleak": privleak,
                    "one_minus_prob": one_minus_prob,
                    "one_minus_rouge": one_minus_rouge,
                    "retention2_avg": retention2,
                    "composite_score": composite,
                    "retention3_avg": retention3,
                    "composite3_score": composite3,
                    # Retain
                    "model_utility_retain_prob": retain_prob,
                    "model_utility_retain_rougeL": retain_rouge,
                    "model_utility_retain_truth_ratio": retain_truth,
                    # Real authors
                    "model_utility_real_authors_prob": real_prob,
                    "model_utility_real_authors_rougeL": real_rouge,
                    "model_utility_real_authors_truth_ratio": real_truth,
                    # World facts
                    "model_utility_world_facts_prob": wf_prob,
                    "model_utility_world_facts_rougeL": wf_rouge,
                    "model_utility_world_facts_truth_ratio": wf_truth,
                }
            )
    return rows


def compute_rows_muse(
    summary_dict: Dict[str, Any],
    folder: str,
    file_name: str,
    model_name: str,
    task_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for iter_key, items in summary_dict.items():
        if not isinstance(iter_key, str) or not iter_key.startswith("iteration"):
            continue
        if not isinstance(items, list):
            continue

        for obj in items:
            if not isinstance(obj, dict):
                continue

            loss_id = obj.get("loss_id")
            ous = obj.get("open_unlearning_summary", {}) or {}
            if not isinstance(ous, dict):
                ous = {}

            muse_score = get_agg_value(ous.get("muse_assessment_score"))
            if not is_valid_number(muse_score):
                muse_score = safe_float(obj.get("topk_score"))

            extr = get_agg_value(ous.get("extraction_strength"))
            priv = get_agg_value(ous.get("privleak"))
            fk = get_agg_value(ous.get("forget_knowmem_ROUGE"))
            fv = get_agg_value(ous.get("forget_verbmem_ROUGE"))
            rk = get_agg_value(ous.get("retain_knowmem_ROUGE"))

            mia_auc = float("nan")
            mia = ous.get("mia_min_k")
            if isinstance(mia, dict):
                mia_auc = get_agg_value(mia.get("auc"))

            rows.append(
                {
                    "folder": folder,
                    "file": file_name,
                    "model": model_name,
                    "task": task_name,
                    "iteration": iter_key,
                    "loss_id": loss_id,
                    "muse_assessment_score": muse_score,
                    "muse_extraction_strength": extr,
                    "muse_privleak": priv,
                    "forget_knowmem_ROUGE": fk,
                    "forget_verbmem_ROUGE": fv,
                    "retain_knowmem_ROUGE": rk,
                    "mia_auc": mia_auc,
                    "model_utility": float("nan"),
                    "composite_score": float("nan"),
                    "retention2_avg": float("nan"),
                }
            )
    return rows


DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def is_date_dir(name: str) -> bool:
    return bool(DATE_DIR_RE.match(name))


def find_summary_file(folder: Path) -> Optional[Path]:
    for candidate in ("summary.txt", "summarize.txt"):
        f = folder / candidate
        if f.exists() and f.is_file():
            return f
    return None


def find_summary_files_recursive(folder: Path) -> List[Path]:
    found: List[Path] = []
    for candidate in ("summary.txt", "summarize.txt"):
        for p in folder.rglob(candidate):
            if p.is_file():
                found.append(p)
    uniq = sorted({str(p) for p in found})
    return [Path(p) for p in uniq]


def scan_results(results_dir: str) -> List[Dict[str, Any]]:
    root = Path(results_dir).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: results directory '{results_dir}' not found at '{root}'.", file=sys.stderr)
        sys.exit(1)

    all_rows: List[Dict[str, Any]] = []

    DEFAULT_MODEL = "Llama2-7B-Instruct"
    DEFAULT_TASK = "forget05"

    for child in sorted([p for p in root.iterdir() if p.is_dir()]):
        name = child.name

        if is_date_dir(name):
            summary_path = find_summary_file(child)
            if summary_path is None:
                continue
            try:
                data = json.loads(summary_path.read_text())
            except Exception as e:
                print(f"WARNING: Skipping {summary_path} due to parse error: {e}", file=sys.stderr)
                continue
            folder_rel = str(child.relative_to(root))
            all_rows.extend(
                compute_rows(
                    data,
                    folder=folder_rel,
                    file_name=summary_path.name,
                    model_name=DEFAULT_MODEL,
                    task_name=DEFAULT_TASK,
                )
            )
            continue

        if name.startswith("tofu_"):
            model_name = name
            for task_dir in sorted([p for p in child.iterdir() if p.is_dir()]):
                task_name = task_dir.name

                summary_path = find_summary_file(task_dir)
                if summary_path is not None:
                    try:
                        data = json.loads(summary_path.read_text())
                    except Exception as e:
                        print(f"WARNING: Skipping {summary_path} due to parse error: {e}", file=sys.stderr)
                    else:
                        folder_rel = str(task_dir.relative_to(root))
                        all_rows.extend(
                            compute_rows(
                                data,
                                folder=folder_rel,
                                file_name=summary_path.name,
                                model_name=model_name,
                                task_name=task_name,
                            )
                        )

                for run_dir in sorted([p for p in task_dir.iterdir() if p.is_dir()]):
                    summary_path = find_summary_file(run_dir)
                    if summary_path is None:
                        continue
                    try:
                        data = json.loads(summary_path.read_text())
                    except Exception as e:
                        print(f"WARNING: Skipping {summary_path} due to parse error: {e}", file=sys.stderr)
                        continue
                    folder_rel = str(run_dir.relative_to(root))
                    all_rows.extend(
                        compute_rows(
                            data,
                            folder=folder_rel,
                            file_name=summary_path.name,
                            model_name=model_name,
                            task_name=task_name,
                        )
                    )
            continue

        if name.startswith("MUSE-") or name.startswith("MUSE_"):
            model_name = "MUSE"
            if "-" in name:
                task_name = name.split("-", 1)[1]
            elif "_" in name:
                task_name = name.split("_", 1)[1]
            else:
                task_name = name

            summary_files = find_summary_files_recursive(child)
            if not summary_files:
                continue

            for summary_path in summary_files:
                try:
                    data = json.loads(summary_path.read_text())
                except Exception as e:
                    print(f"WARNING: Skipping {summary_path} due to parse error: {e}", file=sys.stderr)
                    continue

                try:
                    folder_rel = str(summary_path.parent.relative_to(root))
                except ValueError:
                    folder_rel = str(summary_path.parent)

                all_rows.extend(
                    compute_rows_muse(
                        data,
                        folder=folder_rel,
                        file_name=summary_path.name,
                        model_name=model_name,
                        task_name=task_name,
                    )
                )
            continue

    return all_rows


def find_loss_definition(folder: Path, loss_id: Any, iteration_name: str) -> Optional[str]:
    try_files: List[Path] = []
    if iteration_name.startswith("iteration"):
        try_files.append(folder / f"{iteration_name}.txt")
    for i in range(4):
        p = folder / f"iteration{i}.txt"
        if p not in try_files:
            try_files.append(p)
    try_files.extend(sorted(folder.glob("iteration*.txt")))

    key = str(loss_id)
    seen = set()
    for f in try_files:
        if not f.exists() or f in seen:
            continue
        seen.add(f)
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and key in data and isinstance(data[key], dict):
            d = data[key]
            if "loss_fn_definition" in d and isinstance(d["loss_fn_definition"], str):
                return d["loss_fn_definition"]
    return None


def format_top(
    rows: List[Dict[str, Any]],
    score_key: str,
    retention_key: str,
    title: str,
    top_n: int = 3,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows_valid = [r for r in rows if is_valid_number(r.get(score_key))]
    rows_sorted = sorted(rows_valid, key=lambda r: r[score_key], reverse=True)
    topk = rows_sorted[:top_n]

    lines: List[str] = []
    lines.append(f"=== TOP {len(topk)} — {title} ===")
    header = (
        "rank iter       loss   comp      ret       mu        "
        "1-prob     1-rouge   truth       extr_str    privleak "
        "r_prob   r_rouge  r_truth  ra_prob ra_rouge ra_truth wf_prob wf_rouge wf_truth subfolder"
    )
    lines.append(header)

    latex_lines: List[str] = []

    for i, r in enumerate(topk, 1):
        comp = r[score_key]
        ret = r[retention_key]
        mu = r.get("model_utility", float("nan"))
        one_minus_prob = r.get("one_minus_prob", float("nan"))
        one_minus_rouge = r.get("one_minus_rouge", float("nan"))
        truth = r.get("forget_truth_ratio", float("nan"))
        extr = r.get("extraction_strength", float("nan"))
        priv = r.get("privleak", float("nan"))

        r_prob = r.get("model_utility_retain_prob", float("nan"))
        r_rouge = r.get("model_utility_retain_rougeL", float("nan"))
        r_truth = r.get("model_utility_retain_truth_ratio", float("nan"))

        ra_prob = r.get("model_utility_real_authors_prob", float("nan"))
        ra_rouge = r.get("model_utility_real_authors_rougeL", float("nan"))
        ra_truth = r.get("model_utility_real_authors_truth_ratio", float("nan"))

        wf_prob = r.get("model_utility_world_facts_prob", float("nan"))
        wf_rouge = r.get("model_utility_world_facts_rougeL", float("nan"))
        wf_truth = r.get("model_utility_world_facts_truth_ratio", float("nan"))

        line = (
            f"{i:<4} {r['iteration']:<10} {str(r['loss_id']):<6}"
            f"{comp:>8.2f} {ret:>8.2f} {mu:>8.2f} "
            f"{one_minus_prob:>9.2f} {one_minus_rouge:>9.2f} {truth:>10.2g} "
            f"{extr:>11.2g} {priv:>11.2g} "
            f"{r_prob:>8.2f} {r_rouge:>8.2f} {r_truth:>8.2f} "
            f"{ra_prob:>8.2f} {ra_rouge:>8.2f} {ra_truth:>8.2f} "
            f"{wf_prob:>8.2f} {wf_rouge:>8.2f} {wf_truth:>8.2f} "
            f"{r['folder']}"
        )
        lines.append(line)

        latex_vals_list = [r.get(k, float("nan")) for k in LATEX_METRICS_ORDER]
        latex_vals = " & ".join(fmt2(v) for v in latex_vals_list)
        latex_line = f"\\ours{{}} & {latex_vals} & {fmt2(mu)} \\\\"
        latex_lines.append(latex_line)

    if latex_lines:
        lines.append("")
        lines.append("LaTeX table:")
        lines.extend(latex_lines)
        lines.append("")

    return topk, lines


def format_top_muse(
    rows: List[Dict[str, Any]],
    title: str,
    top_n: int = 3,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    score_key = "muse_assessment_score"
    rows_valid = [r for r in rows if is_valid_number(r.get(score_key))]
    rows_sorted = sorted(rows_valid, key=lambda r: r[score_key], reverse=True)
    topk = rows_sorted[:top_n]

    lines: List[str] = []
    lines.append(f"=== TOP {len(topk)} — {title} ===")
    header = (
        "rank iter       loss   muse_score  extr_str    privleak   "
        "forget_know  forget_verb  retain_know   mia_auc   subfolder"
    )
    lines.append(header)

    muse_latex_lines: List[str] = []

    for i, r in enumerate(topk, 1):
        muse_score = r.get("muse_assessment_score", float("nan"))
        extr = r.get("muse_extraction_strength", float("nan"))
        priv = r.get("muse_privleak", float("nan"))
        fk = r.get("forget_knowmem_ROUGE", float("nan"))
        fv = r.get("forget_verbmem_ROUGE", float("nan"))
        rk = r.get("retain_knowmem_ROUGE", float("nan"))
        auc = r.get("mia_auc", float("nan"))

        line = (
            f"{i:<4} {r['iteration']:<10} {str(r['loss_id']):<6}"
            f"{muse_score:>10.3f} {extr:>10.3g} {priv:>10.3g} "
            f"{fk:>11.3f} {fv:>11.3f} {rk:>11.3f} {auc:>9.3f} "
            f"{r['folder']}"
        )
        lines.append(line)

        muse_latex_lines.append(
            f"\\ours{{}} & {fmt_pct2(fv)} & {fmt_pct2(fk)} & {fmt2(priv)} & {fmt_pct2(rk)} \\\\"
        )

    if muse_latex_lines:
        lines.append("")
        lines.append("LaTeX table:")
        lines.extend(muse_latex_lines)
        lines.append("")

    return topk, lines


def main() -> None:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    if len(sys.argv) > 2:
        try:
            min_model_utility = float(sys.argv[2])
        except Exception:
            print(
                f"WARNING: Could not parse min_model_utility from '{sys.argv[2]}', "
                f"falling back to DEFAULT_MIN_MODEL_UTILITY={DEFAULT_MIN_MODEL_UTILITY}",
                file=sys.stderr,
            )
            min_model_utility = DEFAULT_MIN_MODEL_UTILITY
    else:
        min_model_utility = DEFAULT_MIN_MODEL_UTILITY

    rows = scan_results(results_dir)
    if not rows:
        print("No results found.")
        sys.exit(0)

    if min_model_utility > 0.0:
        rows = [
            r
            for r in rows
            if (
                (is_valid_number(r.get("model_utility")) and r["model_utility"] >= min_model_utility)
                or is_valid_number(r.get("muse_assessment_score"))
                or r.get("model") == "MUSE"
            )
        ]
        if not rows:
            print(f"No results found after applying model_utility >= {min_model_utility} filter.")
            sys.exit(0)

    from collections import defaultdict

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r.get("model", "UNKNOWN_MODEL"), r.get("task", "UNKNOWN_TASK"))].append(r)

    lines: List[str] = []
    all_topk: List[Dict[str, Any]] = []

    first_block = True
    for (model, task) in sorted(grouped.keys()):
        group_rows = grouped[(model, task)]
        is_muse_group = (model == "MUSE") or any(is_valid_number(r.get("muse_assessment_score")) for r in group_rows)

        if is_muse_group:
            title = f"model={model} | task={task} | score=muse_assessment_score (TOP-3)"
            topk, block_lines = format_top_muse(group_rows, title=title, top_n=3)
        else:
            title = (
                f"model={model} | task={task} | "
                f"avg( model_utility, avg(1-Prob, 1-ROUGE) ) "
                f"| filter: model_utility >= {min_model_utility}"
            )
            topk, block_lines = format_top(
                group_rows,
                score_key="composite_score",
                retention_key="retention2_avg",
                title=title,
                top_n=3,
            )

        if not topk:
            continue
        if not first_block:
            lines.append("")
        first_block = False
        lines.extend(block_lines)
        all_topk.extend(topk)

    if not all_topk:
        print("No valid top results found.")
        sys.exit(0)

    lines.append("")
    lines.append("Loss function definitions (TOP results per model/task):")

    unique_keys: List[Tuple[str, str, Any]] = []
    for r in all_topk:
        key = (r["folder"], r["iteration"], r["loss_id"])
        if key not in unique_keys:
            unique_keys.append(key)

    root = Path(results_dir).expanduser().resolve()
    for folder_rel, iteration, loss_id in unique_keys:
        folder_path = root / folder_rel
        definition = find_loss_definition(folder_path, loss_id, iteration)
        lines.append("")
        lines.append(f"# loss_id {loss_id} — {folder_rel}/{iteration}.txt")
        if definition:
            lines.append(definition)
        else:
            lines.append("(definition not found in iteration*.txt under this folder)")

    output_path = root / "final.txt"
    output_path.write_text("\n".join(lines))
    print(
        f"Wrote TOP-3 per (model, task) to {output_path} "
        f"(model_utility filter: >= {min_model_utility}, MUSE unfiltered)"
    )


if __name__ == "__main__":
    main()
