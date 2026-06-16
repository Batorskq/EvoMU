# LoRA unlearning: iteration0 (initial N=10) + iteration1 (top5→5 children each = 25) + OU eval
# Extended to iteration2 and iteration3 per user request:
#   (5) After iteration1 (25 children) choose top3
#   (6) For each of those 3, generate 10 children -> 30 new losses
#   (7) Train all 30 losses (iteration2)
#   (8) Choose top5 from iteration2
#   (9) For each of those 5, generate 3 children -> 15 new losses
#  (10) Train all 15 losses (iteration3)
#
# Epochs are provided by EACH generated loss function via its docstring:
# Inside each loss function, the FIRST statement must be a docstring like: """epochs: 3"""
# We parse that and train for the specified number of epochs per loss.
# ----------------------------------------------------------------------
# MODS:
# - Per-parent logging for refinement:
#   - iteration1_{i}.txt : summary for parent i's refined children after training
#   - thinking1_{i}.txt  : refinement thinking (written by generate.py)
#   - initial1_{i}.txt   : refined functions for parent i (written by generate.py)
# - thinking_initial.txt : initial thinking (written by generate.py)
# - --debug flag: if set, force epochs=2 for ALL losses (ignores docstrings)
#
# Error handling (per user request):
# - If any error occurs during training/eval of a loss, we:
#     * omit it from top-k and all scoring/selection
#     * still write an entry in iteration*.txt with the loss definition and an "error" field
# - After everything finishes, we write summary.txt containing, for each iteration,
#   the list of entries with topk_score (if computable) and open_unlearning_summary (or error).

import os
import json
import gc
import re
import argparse
import traceback
from pathlib import Path
from contextlib import nullcontext
from typing import List, Dict, Any, Tuple, Optional, Sequence
from collections import defaultdict
from datetime import datetime
import math

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model

from utils import (
    run_generate_and_load_loss_functions,
    run_refine_and_load_loss_functions,
    get_loss_fn_definition_from_file,
    JsonlQADataset,
    CausalCollator,
    per_sample_log_probs,
    to_device,
    precompute_ref_log_probs_forget,  # reused for forget & retain
    prepare_eval_snapshot,
    run_openunlearning_eval,
    write_current_summary,
    append_top_k_metadata,
    parse_epochs_from_docstring,
    get_results_dir,   # shared results dir
)



# -----------------------
# Helpers / device & data
# -----------------------
def _ensure_device_and_dtype(device: Optional[str]):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train_device = torch.device(device)
    if train_device.type == "cuda":
        def autocast_context():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        train_dtype = torch.float16
        eval_device_idx = train_device.index if train_device.index is not None else 0
    else:
        def autocast_context():
            return nullcontext()
        train_dtype = torch.float32
        eval_device_idx = 0
    return train_device, train_dtype, eval_device_idx, autocast_context


def _build_datasets_and_collator(
    tokenizer,
    data_dir: str,
    forget_split: str,
    retain_split: str,
    max_length: int,
):
    """
    Build forget/retain datasets from the specified TOFU splits.

    Examples:
      - forget_split="forget05_train.jsonl", retain_split="retain95.jsonl"
      - forget_split="forget10_train.jsonl", retain_split="retain90.jsonl"
    """
    forget_path = os.path.join(data_dir, forget_split)
    retain_path = os.path.join(data_dir, retain_split)

    print(f"[train] Loading FORGET split from: {forget_path}")
    print(f"[train] Loading RETAIN split from: {retain_path}")

    forget_dataset = JsonlQADataset(forget_path, tokenizer, max_length=max_length)
    retain_dataset = JsonlQADataset(retain_path, tokenizer, max_length=max_length)
    collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)
    return forget_dataset, retain_dataset, collator


def _maybe_compute_ref_caches(
    model_name: str,
    device: torch.device,
    dtype_for_model: torch.dtype,
    collator: CausalCollator,
    pad_token_id: int,
    forget_dataset: JsonlQADataset,
    retain_dataset: JsonlQADataset,
) -> Dict[str, torch.Tensor]:
    """
    Compute both caches once; we will only use them for losses that request them.
    """
    print("[train] Precomputing reference log-probs (forget & retain)...")
    ref_forget = precompute_ref_log_probs_forget(
        model_name=model_name,
        forget_dataset=forget_dataset,
        pad_token_id=pad_token_id,
        device=device,
        dtype_for_model=dtype_for_model,
        collator=collator,
        batch_size=1,
    )
    ref_retain = precompute_ref_log_probs_forget(
        model_name=model_name,
        forget_dataset=retain_dataset,
        pad_token_id=pad_token_id,
        device=device,
        dtype_for_model=dtype_for_model,
        collator=collator,
        batch_size=1,
    )
    return {"forget": ref_forget, "retain": ref_retain}


# -----------------------
# Scoring / metrics utils
# -----------------------
def _extract_metric(d: Dict[str, Any], keys: Tuple[str, ...], default: float) -> float:
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return float(default)


def _safe_clamp01(x: float, default: float = 0.0) -> float:
    """
    Clamp a scalar to [0, 1]. If x is NaN / Inf / not convertible, return `default`.
    """
    try:
        xf = float(x)
    except Exception:
        return default
    if math.isnan(xf) or math.isinf(xf):
        return default
    if xf < 0.0:
        return 0.0
    if xf > 1.0:
        return 1.0
    return xf


def _compute_topk_score(metrics: Dict[str, Any]) -> float:
    """
    Top-k scoring rule:

        score = 0.5 * model_utility_scaled + 0.5 * forget_score

    where forget_score is the average over the available subset of:
        (1 - RougeL), (1 - Prob),

    All terms are scaled into [0, 1], and 1 means "better" (higher utility / better forgetting).
    """
    if not isinstance(metrics, dict):
        metrics = {}

    # Model utility (assumed ~[0,1], but we clamp for safety).
    mu_raw = _extract_metric(metrics, ("model_utility",), 0.0)
    mu = _safe_clamp01(mu_raw, 0.0)

    forget_terms: List[float] = []

    # 1 - RougeL on the forget set
    rouge_raw = _extract_metric(
        metrics,
        ("forget_Q_A_ROUGE", "forget_Q_A_Rouge", "forget_Q_A_rouge"),
        float("nan"),
    )
    if not math.isnan(rouge_raw):
        rouge = _safe_clamp01(rouge_raw, 0.0)
        forget_terms.append(1.0 - rouge)

    # 1 - Prob on the forget set
    fprob_raw = _extract_metric(
        metrics,
        ("forget_Q_A_Prob", "forget_Q_A_Probability", "forget_Q_A_prob"),
        float("nan"),
    )
    if not math.isnan(fprob_raw):
        fprob = _safe_clamp01(fprob_raw, 0.0)
        forget_terms.append(1.0 - fprob)

    if forget_terms:
        forget_score = sum(forget_terms) / float(len(forget_terms))
    else:
        forget_score = 0.0

    return 0.5 * mu + 0.5 * forget_score


# ---------------------------------------------------------------------------
# Helpers to pull detailed model-utility components from TOFU_EVAL.json
# ---------------------------------------------------------------------------
def _ou_eval_flatten_numeric_leafs(obj: Any, prefix: str = "") -> Dict[str, float]:
    """
    Flatten nested dict/list into a mapping: dotted_path -> numeric_value.

    SPECIAL HANDLING:
      - If a dict has an "agg_value" numeric field, we record ONLY that
        (as "<prefix>.agg_value") and DO NOT descend into children such as
        "value_by_index". This prevents us from ever picking per-example
        scores like index 100.0 instead of the aggregated metric.
    """
    flat: Dict[str, float] = {}

    if isinstance(obj, dict):
        # If this dict looks like a metric container with agg_value,
        # keep only the aggregate and stop.
        if "agg_value" in obj and isinstance(obj["agg_value"], (int, float)):
            key = f"{prefix}.agg_value" if prefix else "agg_value"
            flat[key] = float(obj["agg_value"])
            return flat

        # Otherwise recurse normally.
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (int, float)):
                flat[new_prefix] = float(v)
            else:
                flat.update(_ou_eval_flatten_numeric_leafs(v, new_prefix))

    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            if isinstance(v, (int, float)):
                flat[new_prefix] = float(v)
            else:
                flat.update(_ou_eval_flatten_numeric_leafs(v, new_prefix))

    return flat


def _ou_eval_key_tokens(key: str) -> set:
    """
    Tokenize a flattened key path into a bag of lowercase tokens.
    We split on '.', '_', '-', '/', '[', ']'.
    """
    separators = [".", "_", "-", "/", "[", "]"]
    tmp = key.lower()
    for sep in separators:
        tmp = tmp.replace(sep, " ")
    return set(tok for tok in tmp.split() if tok)


def _ou_eval_extract_numeric_candidate(cand: Any) -> Optional[float]:
    """
    Extract a numeric value from a typical TOFU metric leaf:
      - direct float/int
      - dict with 'agg_value', 'mean', 'value', or 'score'
    """
    if isinstance(cand, (int, float)):
        return float(cand)
    if isinstance(cand, dict):
        for agg_key in ("agg_value", "mean", "value", "score"):
            v = cand.get(agg_key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _ou_eval_extract_single_metric(obj: Dict[str, Any], names: Tuple[str, ...]) -> Optional[float]:
    """
    Try to extract a single metric by any of the provided names.
    Looks at:
      - top-level obj[name]
      - obj["metrics"][name]
      - recursive search for keys matching any of the names
    """
    # top level + "metrics" block
    for name in names:
        cand = obj.get(name)
        val = _ou_eval_extract_numeric_candidate(cand)
        if val is not None:
            return val

        metrics_block = obj.get("metrics")
        if isinstance(metrics_block, dict) and name in metrics_block:
            val = _ou_eval_extract_numeric_candidate(metrics_block[name])
            if val is not None:
                return val

    # recursive search
    def _recurse(o: Any) -> Optional[float]:
        if isinstance(o, dict):
            for k, v in o.items():
                if k in names:
                    val = _ou_eval_extract_numeric_candidate(v)
                    if val is not None:
                        return val
                nested = _recurse(v)
                if nested is not None:
                    return nested
        elif isinstance(o, list):
            for v in o:
                nested = _recurse(v)
                if nested is not None:
                    return nested
        return None

    return _recurse(obj)


def _ou_eval_extract_model_utility_components(
    eval_obj: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
    Extract model-utility component metrics for:
      - retain
      - real_authors (RA)
      - world_facts (WF)
    """
    metrics_root = eval_obj.get("metrics", eval_obj)
    flat = _ou_eval_flatten_numeric_leafs(metrics_root)

    def extract(include_token_sets: Sequence[set]) -> Optional[float]:
        best_key: Optional[str] = None
        best_score: float = -1.0

        for key, val in flat.items():
            tokens = _ou_eval_key_tokens(key)

            for inc in include_token_sets:
                if not inc.issubset(tokens):
                    continue

                # Base score: how many tokens matched
                score = float(len(inc))

                # Slight preference for aggregated values ('.agg_value')
                if key.endswith(".agg_value"):
                    score += 0.1

                if score > best_score:
                    best_score = score
                    best_key = key

        return flat.get(best_key) if best_key is not None else None

    out: Dict[str, Optional[float]] = {}

    # ---------- retain metrics ----------
    retain_prob_direct = _ou_eval_extract_single_metric(
        metrics_root,
        ("retain_Q_A_Prob", "retain_Q_A_PROB", "retain_Q_A_Probability"),
    )
    if retain_prob_direct is not None:
        out["model_utility_retain_prob"] = retain_prob_direct
    else:
        out["model_utility_retain_prob"] = extract(
            [
                {"retain", "prob"},
                {"retain", "probability"},
            ]
        )

    retain_rouge_direct = _ou_eval_extract_single_metric(
        metrics_root,
        ("retain_Q_A_ROUGE", "retain_Q_A_Rouge"),
    )
    if retain_rouge_direct is not None:
        out["model_utility_retain_rougeL"] = retain_rouge_direct
    else:
        out["model_utility_retain_rougeL"] = extract(
            [
                {"retain", "rougel"},
                {"retain", "rouge", "l"},
                {"retain", "q", "a", "rouge"},
            ]
        )

    retain_truth_direct = _ou_eval_extract_single_metric(
        metrics_root,
        ("retain_Q_A_Truth_Ratio", "retain_Truth_Ratio"),
    )
    if retain_truth_direct is not None:
        out["model_utility_retain_truth_ratio"] = retain_truth_direct
    else:
        out["model_utility_retain_truth_ratio"] = extract(
            [
                {"retain", "truth", "ratio"},
            ]
        )

    # ---------- real authors (RA) metrics ----------
    ra_prob_norm = _ou_eval_extract_single_metric(
        metrics_root,
        ("ra_Q_A_Prob_normalised", "ra_Q_A_Prob_normalized"),
    )
    if ra_prob_norm is not None:
        out["model_utility_real_authors_prob"] = ra_prob_norm
    else:
        out["model_utility_real_authors_prob"] = extract(
            [
                {"ra", "prob"},
                {"real", "authors", "prob"},
                {"ra", "probability"},
                {"real", "authors", "probability"},
            ]
        )

    ra_rouge_direct = _ou_eval_extract_single_metric(
        metrics_root,
        ("ra_Q_A_ROUGE", "ra_Q_A_Rouge"),
    )
    if ra_rouge_direct is not None:
        out["model_utility_real_authors_rougeL"] = ra_rouge_direct
    else:
        out["model_utility_real_authors_rougeL"] = extract(
            [
                {"ra", "rougel"},
                {"real", "authors", "rougel"},
                {"ra", "rouge", "l"},
                {"real", "authors", "rouge", "l"},
                {"ra", "q", "a", "rouge"},
            ]
        )

    out["model_utility_real_authors_truth_ratio"] = extract(
        [
            {"ra", "truth", "ratio"},
            {"real", "authors", "truth", "ratio"},
        ]
    )

    # ---------- world facts (WF) metrics ----------
    wf_prob_norm = _ou_eval_extract_single_metric(
        metrics_root,
        ("wf_Q_A_Prob_normalised", "wf_Q_A_Prob_normalized"),
    )
    if wf_prob_norm is not None:
        out["model_utility_world_facts_prob"] = wf_prob_norm
    else:
        out["model_utility_world_facts_prob"] = extract(
            [
                {"wf", "prob"},
                {"world", "facts", "prob"},
                {"wf", "probability"},
                {"world", "facts", "probability"},
            ]
        )

    wf_rouge_direct = _ou_eval_extract_single_metric(
        metrics_root,
        ("wf_Q_A_ROUGE", "wf_Q_A_Rouge"),
    )
    if wf_rouge_direct is not None:
        out["model_utility_world_facts_rougeL"] = wf_rouge_direct
    else:
        out["model_utility_world_facts_rougeL"] = extract(
            [
                {"wf", "rougel"},
                {"world", "facts", "rougel"},
                {"wf", "rouge", "l"},
                {"world", "facts", "rouge", "l"},
                {"wf", "q", "a", "rouge"},
            ]
        )

    out["model_utility_world_facts_truth_ratio"] = extract(
        [
            {"wf", "truth", "ratio"},
            {"world", "facts", "truth", "ratio"},
        ]
    )

    return out


def _augment_metrics_with_eval(
    open_unlearning_summary: Dict[str, Any],
    eval_json_path: str,
) -> Dict[str, Any]:
    """
    Take an open_unlearning_summary dict (TOFU_SUMMARY-based) and augment it with
    model utility component metrics extracted from TOFU_EVAL.json.
    """
    summary = dict(open_unlearning_summary) if isinstance(open_unlearning_summary, dict) else {}

    if not eval_json_path:
        return summary

    try:
        if os.path.isfile(eval_json_path):
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_obj = json.load(f)
            if isinstance(eval_obj, dict):
                components = _ou_eval_extract_model_utility_components(eval_obj)
                # Only write keys that we actually found (not None)
                for k, v in components.items():
                    if v is not None:
                        summary[k] = v
    except Exception as e:
        # Non-fatal; just log and keep whatever we already had.
        print(f"[train] Warning: failed to augment metrics from TOFU_EVAL.json: {e}")

    return summary


# -----------------------
# Training for a loss
# -----------------------
def _train_one_loss(
    base_model_name: str,
    loss_id: int,
    selected_loss_fn,
    tokenizer,
    device,
    train_dtype,
    autocast_context,
    eval_device_idx: int,
    forget_dataset: JsonlQADataset,
    retain_dataset: JsonlQADataset,
    collator: CausalCollator,
    output_dir_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    pad_token_id: int,
    ref_caches: Dict[str, torch.Tensor] | None,
    retain_logs_path: Optional[str],
    no_retain: bool = False,
) -> Dict[str, Any]:
    """
    Train LoRA for a single loss, run OU eval, and return a record with metrics.

    If no_retain=True:
      - We DO NOT compute log_probs_retain (no access to retain set forward pass).
      - We still load retain batches for:
          * stable step alignment via zip(forget_loader, retain_loader)
          * optional ref_log_probs_retain indexing (if the loss requests it)
      - We pass a dummy tensor for log_probs_retain (zeros), so loss fns must not use it.
    """

    # Inspect signature to decide if we pass reference caches
    import inspect
    sig = inspect.signature(selected_loss_fn)
    params = set(sig.parameters.keys())
    needs_ref_forget = "ref_log_probs_forget" in params
    needs_ref_retain = "ref_log_probs_retain" in params

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=train_dtype,
    )
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.to(device)
    lora_model.train()

    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]

    forget_loader = DataLoader(
        forget_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collator, pin_memory=(device.type == "cuda"),
    )
    retain_loader = DataLoader(
        retain_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collator, pin_memory=(device.type == "cuda"),
    )

    min_len = min(len(forget_loader), len(retain_loader))
    total_steps = epochs * min_len

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    epoch_final_losses: List[float] = []
    last_loss_value = None

    for epoch in range(epochs):
        print(f"\n[train] loss_id={loss_id} | Epoch {epoch+1}/{epochs} | no_retain={no_retain}")
        for step, (batch_forget, batch_retain) in enumerate(zip(forget_loader, retain_loader)):
            batch_forget_dev = to_device(batch_forget, device)
            batch_retain_dev = to_device(batch_retain, device)

            with autocast_context():
                log_probs_forget = per_sample_log_probs(
                    lora_model,
                    batch_forget_dev["input_ids"],
                    batch_forget_dev["attention_mask"],
                    pad_token_id=pad_token_id,
                )

                if no_retain:
                    # Dummy placeholder; generated losses MUST NOT use this.
                    log_probs_retain = torch.zeros_like(log_probs_forget)
                else:
                    log_probs_retain = per_sample_log_probs(
                        lora_model,
                        batch_retain_dev["input_ids"],
                        batch_retain_dev["attention_mask"],
                        pad_token_id=pad_token_id,
                    )

                kwargs = {
                    "log_probs_forget": log_probs_forget,
                    "log_probs_retain": log_probs_retain,
                }

                if needs_ref_forget and ref_caches is not None:
                    idx_f = batch_forget_dev["idx"].long()
                    kwargs["ref_log_probs_forget"] = ref_caches["forget"][idx_f].to(log_probs_forget.dtype)

                if needs_ref_retain and ref_caches is not None:
                    # Even in no_retain mode, we can index ref retain log-probs via retain batch idx.
                    idx_r = batch_retain_dev["idx"].long()
                    kwargs["ref_log_probs_retain"] = ref_caches["retain"][idx_r].to(log_probs_forget.dtype)

                loss = selected_loss_fn(**kwargs)

            # --- ensure differentiable scalar ---
            if not isinstance(loss, torch.Tensor):
                loss = torch.as_tensor(loss, device=device, dtype=log_probs_forget.dtype)
            if loss.ndim != 0:
                loss = loss.mean()
            if not loss.requires_grad:
                # ensure grad path at least through forget (and retain if enabled)
                if no_retain:
                    loss = loss + 0.0 * (log_probs_forget.mean())
                else:
                    loss = loss + 0.0 * (log_probs_forget.mean() + log_probs_retain.mean())
            # ------------------------------------

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_((trainable_params), grad_clip_norm)
            optimizer.step()
            scheduler.step()
            last_loss_value = float(loss.item())

            if (step + 1) % 10 == 0 or (step + 1) == min_len:
                print(f"[train] loss_id={loss_id} step {step+1}/{min_len} loss={last_loss_value:.4f}")

        epoch_final_losses.append(last_loss_value if last_loss_value is not None else float("nan"))

    # Save final adapters
    final_dir = os.path.join(output_dir_root, f"loss_{loss_id}")
    os.makedirs(final_dir, exist_ok=True)
    lora_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train] Completed LoRA training for loss_id={loss_id}. Saved to {final_dir}")

    # Free training model before merge/eval
    try:
        lora_model.to("cpu")
    except Exception:
        pass
    del lora_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Merge and eval
    eval_snapshot_parent = os.path.join(final_dir, "merged_for_eval")
    os.makedirs(eval_snapshot_parent, exist_ok=True)
    eval_snapshot_dir = prepare_eval_snapshot(
        base_model_name=base_model_name,
        lora_dir=final_dir,
        save_dir=eval_snapshot_parent,
        dtype=train_dtype,
        device=device,
    )
    eval_snapshot_dir_abs = os.path.abspath(eval_snapshot_dir)
    print(f"[train] Merged model for eval saved to {eval_snapshot_dir_abs}")

    print("[train] Launching Open-Unlearning evaluation ...")
    run_info = run_openunlearning_eval(
        ckpt_dir=eval_snapshot_dir_abs,
        device_idx=eval_device_idx,
        task_name=None,
        retain_logs_path=retain_logs_path,
        base_model_name=base_model_name,
    )
    print("[train] Open-Unlearning eval finished.")
    print(json.dumps(run_info, indent=4))

    # Load/assemble metrics
    tofu_summary_json_path = run_info.get("tofu_summary_json", "")
    tofu_eval_json_path = run_info.get("tofu_eval_json", "")
    try:
        if tofu_summary_json_path and os.path.isfile(tofu_summary_json_path):
            with open(tofu_summary_json_path, "r", encoding="utf-8") as f:
                metrics_block: Dict[str, Any] = json.load(f)
        else:
            metrics_block = {"error": "TOFU_SUMMARY.json not found"}
    except Exception as e:
        metrics_block = {"error": str(e)}

    # Add detailed model-utility components from TOFU_EVAL.json
    metrics_block = _augment_metrics_with_eval(metrics_block, tofu_eval_json_path)

    # Compute top-k score
    if isinstance(metrics_block, dict) and "error" in metrics_block:
        topk_score = float("-inf")
    else:
        topk_score = _compute_topk_score(metrics_block)

    return {
        "loss_id": loss_id,
        "loss_history": epoch_final_losses,
        "tofu_summary_json": tofu_summary_json_path,
        "tofu_eval_json": tofu_eval_json_path,
        "metrics_block": metrics_block,
        "topk_score": topk_score,
        "eval_run": run_info,
    }



# -----------------------
# Error recording helpers
# -----------------------
def _record_error_entry(
    loss_id: int,
    loss_fn_definition: str,
    filename: str,
    error: BaseException,
) -> None:
    """
    Write an entry to iteration*.txt with the function definition and an 'error' field.
    """
    err_obj = {
        "error": str(error),
        "exception_type": type(error).__name__,
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }
    write_current_summary(
        loss_id=loss_id,
        loss_fn_definition=loss_fn_definition,
        loss_history=[],
        tofu_summary_json_path="",
        tofu_eval_json_path="",
        precomputed_metrics=err_obj,
        filename=filename,
    )


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _build_iteration_summary_dict(iter_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    From an iteration file object, build a list of {loss_id, topk_score, open_unlearning_summary or error}.
    We recompute topk_score from stored metrics when available.
    """
    out: List[Dict[str, Any]] = []
    for k, v in iter_json.items():
        if k == "__top_k__":
            continue
        try:
            loss_id = int(k)
        except Exception:
            continue
        metrics = v.get("open_unlearning_summary", {})
        if isinstance(metrics, dict) and "error" in metrics:
            out.append({
                "loss_id": loss_id,
                "topk_score": None,
                "error": metrics.get("error"),
                "open_unlearning_summary": metrics,
            })
        else:
            score = _compute_topk_score(metrics if isinstance(metrics, dict) else {})
            score = score if (isinstance(score, (int, float)) and math.isfinite(score)) else None
            out.append({
                "loss_id": loss_id,
                "topk_score": score,
                "open_unlearning_summary": metrics,
            })

    # sort: valid scores desc, then errors at the end
    def _sort_key(d):
        sc = d.get("topk_score")
        return (0, -sc) if isinstance(sc, (int, float)) else (1, 0.0)

    out.sort(key=_sort_key)
    return out


def _write_overall_summary(results_dir: Path) -> None:
    """
    Create summary.txt with per-iteration summaries containing (loss_id, topk_score, open_unlearning_summary or error).
    Includes iteration0..iteration3.
    """
    it0 = _load_json_if_exists(results_dir / "iteration0.txt")
    it1 = _load_json_if_exists(results_dir / "iteration1.txt")
    it2 = _load_json_if_exists(results_dir / "iteration2.txt")
    it3 = _load_json_if_exists(results_dir / "iteration3.txt")

    summary = {
        "iteration0": _build_iteration_summary_dict(it0) if it0 else [],
        "iteration1": _build_iteration_summary_dict(it1) if it1 else [],
        "iteration2": _build_iteration_summary_dict(it2) if it2 else [],
        "iteration3": _build_iteration_summary_dict(it3) if it3 else [],
    }
    out_path = results_dir / "summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Wrote overall summary to {out_path}")


# -----------------------
# Slug helper for results dirs
# -----------------------
def _slugify_name(val: str) -> str:
    """
    Turn a model name or filename into a safe path component.
    Example: 'open-unlearning/tofu_Llama-3.2-1B-Instruct_full'
         -> 'tofu_Llama-3.2-1B-Instruct_full'
    """
    base = os.path.basename(val)
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '-', base)
    slug = slug.strip("-_.")
    return slug or "default"


# -----------------------
# Main
# -----------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA TOFU unlearning: multi-iteration (0..3). Epochs per loss come from function docstrings."
    )

    # Base model: can be Llama-2 TOFU or Llama-3.2 TOFU, etc.
    parser.add_argument("--model_name", type=str, default="locuslab/tofu_ft_llama2-7b")

    # TOFU data location and splits
    parser.add_argument("--data_dir", type=str, default="tofu_data")
    parser.add_argument(
        "--forget_split",
        type=str,
        default="forget05_train.jsonl",
        help="Forget split filename inside data_dir (e.g. forget05_train.jsonl, forget10_train.jsonl)",
    )
    parser.add_argument(
        "--retain_split",
        type=str,
        default="retain95.jsonl",
        help="Retain split filename inside data_dir (e.g. retain95.jsonl, retain90.jsonl)",
    )

    parser.add_argument("--output_dir", type=str, default="unlearned_lora_models")
    parser.add_argument("--output_dir_iter1", type=str, default="unlearned_lora_models_iter1")
    parser.add_argument("--output_dir_iter2", type=str, default="unlearned_lora_models_iter2")
    parser.add_argument("--output_dir_iter3", type=str, default="unlearned_lora_models_iter3")

    # epochs removed: now each loss function specifies its own epochs via docstring
    parser.add_argument(
    "--no_retain",
    action="store_true",
    help=(
        "If set, generate loss functions that MUST NOT use log_probs_retain. "
        "Training will skip computing log_probs_retain forward pass (retain batches still loaded for ref indexing)."
    ),
)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Iteration knobs
    parser.add_argument("--iter0_top_k", type=int, default=5, help="number of parents to keep for refinement (iteration1)")
    parser.add_argument("--children_per_parent", type=int, default=5, help="children per parent for iteration1")
    parser.add_argument("--iter1_top_k", type=int, default=3, help="top results to seed iteration2")

    # New knobs for iteration2/3
    parser.add_argument("--iter2_children_per_parent", type=int, default=10, help="children per parent for iteration2")
    parser.add_argument("--iter2_top_k", type=int, default=5, help="top results from iteration2 to seed iteration3")
    parser.add_argument("--iter3_children_per_parent", type=int, default=3, help="children per parent for iteration3")
    parser.add_argument("--iter3_top_k", type=int, default=5, help="(optional) how many to record as top in iteration3.txt")

    # Debug knob
    parser.add_argument("--limit_losses", type=int, default=0, help="if >0, limit initial losses trained")
    parser.add_argument("--debug", action="store_true",
                        help="If set, override all loss docstrings and train exactly 2 epochs for every loss.")

    # retain logs so forget_quality is computed
    parser.add_argument(
        "--retain_logs_path",
        type=str,
        default="",
        help=(
            "Path to retain model TOFU_EVAL.json to enable reference-based metrics like "
            "forget_quality/privleak. If empty, we will try standard OpenUnlearning defaults "
            "based on the inferred OpenUnlearning model key."
        ),
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    print(f"[train] Base model: {args.model_name}")
    print(f"[train] Data dir: {args.data_dir}")
    print(f"[train] TOFU splits: forget={args.forget_split} | retain={args.retain_split}")

    # -----------------------
    # Establish a config-scoped results directory and export it
    # -----------------------
    model_slug = _slugify_name(args.model_name)
    forget_slug = _slugify_name(os.path.splitext(os.path.basename(args.forget_split))[0])
    retain_slug = _slugify_name(os.path.splitext(os.path.basename(args.retain_split))[0])

    # Layout: results/<model_slug>/<forget_slug>__<retain_slug>/<timestamp>/
    base_results = repo_root / "results" / model_slug / f"{forget_slug}__{retain_slug}"
    base_results.mkdir(parents=True, exist_ok=True)
    run_dir = base_results / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Make available to generate.py and utils.py (including subprocess)
    os.environ["UNLEARN_RESULTS_DIR"] = str(run_dir)
    print(f"[train] Using results directory: {run_dir.resolve()}")

    # -----------------------
    # Initial generation (generate.py default N=10)
    # -----------------------
    loss_functions = run_generate_and_load_loss_functions(repo_root, no_retain=args.no_retain)

    if not loss_functions:
        raise ValueError("No loss functions were generated.")

    # Optionally limit how many we train in this run
    all_loss_ids = sorted(loss_functions.keys())
    if args.limit_losses and args.limit_losses > 0:
        all_loss_ids = all_loss_ids[: args.limit_losses]

    # Device / dtype
    train_device, train_dtype, eval_device_idx, autocast_context = _ensure_device_and_dtype(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # Data once (supports forget05/retain95, forget10/retain90, etc.)
    forget_dataset, retain_dataset, collator = _build_datasets_and_collator(
        tokenizer,
        args.data_dir,
        args.forget_split,
        args.retain_split,
        args.max_length,
    )

    # Precompute both reference caches up front (used only when a loss needs them)
    ref_caches = _maybe_compute_ref_caches(
        model_name=args.model_name,
        device=train_device,
        dtype_for_model=train_dtype,
        collator=collator,
        pad_token_id=pad_token_id,
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Train/Eval initial losses and write iteration0.txt per loss
    results_iter0: List[Dict[str, Any]] = []
    detailed_for_topk_iter0: List[Dict[str, Any]] = []
    loss_file_iter0 = get_results_dir() / "generated_unlearning_losses.txt"

    for loss_id in all_loss_ids:
        selected_loss_fn = loss_functions[loss_id]
        loss_def = get_loss_fn_definition_from_file(loss_file_iter0, loss_id)
        # Read epochs from the loss function's docstring (override if debug)
        parsed_epochs = parse_epochs_from_docstring(selected_loss_fn, default=1)
        epochs_this = 2 if args.debug else parsed_epochs
        if args.debug:
            print(f"[train] loss_id={loss_id} DEBUG mode: overriding epochs to 2 (docstring was {parsed_epochs})")
        else:
            print(f"[train] loss_id={loss_id} will train for epochs={epochs_this} (from docstring)")

        try:
            record = _train_one_loss(
                base_model_name=args.model_name,
                loss_id=loss_id,
                selected_loss_fn=selected_loss_fn,
                tokenizer=tokenizer,
                device=train_device,
                train_dtype=train_dtype,
                autocast_context=autocast_context,
                eval_device_idx=eval_device_idx,
                forget_dataset=forget_dataset,
                retain_dataset=retain_dataset,
                collator=collator,
                output_dir_root=args.output_dir,
                epochs=epochs_this,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                pad_token_id=pad_token_id,
                ref_caches=ref_caches,
                retain_logs_path=args.retain_logs_path,
                no_retain=args.no_retain,
            )

            results_iter0.append(record)

            # Merge into iteration0.txt
            write_current_summary(
                loss_id=loss_id,
                loss_fn_definition=loss_def,
                loss_history=record["loss_history"],
                tofu_summary_json_path=record.get("tofu_summary_json", ""),
                tofu_eval_json_path=record.get("tofu_eval_json", ""),
                precomputed_metrics=record.get("metrics_block", {}),
                filename="iteration0.txt",
            )

            # For top-k selection (full entries)
            detailed_for_topk_iter0.append({
                "loss_id": loss_id,
                "loss_fn_definition": loss_def,
                "loss_history": record["loss_history"],
                "open_unlearning_summary": record.get("metrics_block", {}),
                "topk_score": record.get("topk_score", float("-inf")),
            })

        except Exception as e:
            print(f"[train] ERROR while training loss_id={loss_id}: {e}")
            _record_error_entry(loss_id, loss_def, filename="iteration0.txt", error=e)

        # Free CUDA between losses
        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    # Select top-K parents (by custom score) — failed ones were omitted
    sorted_iter0 = sorted(detailed_for_topk_iter0, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
    parents = sorted_iter0[: max(1, args.iter0_top_k)]

    append_top_k_metadata(parents, filename="iteration0.txt")
    print("\n[train] Iteration-0 Top parents:")
    for t in parents:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    # -----------------------
    # Iteration 1: each parent -> children_per_parent children
    # -----------------------
    seeds: List[Dict[str, Any]] = []
    for p in parents:
        seeds.append({
            "parent_loss_id": p["loss_id"],
            "loss_fn_definition": p["loss_fn_definition"],
            "loss_history": p["loss_history"],
            "open_unlearning_summary": p["open_unlearning_summary"],
        })

    refined_functions, refine_meta = run_refine_and_load_loss_functions(
        repo_root=repo_root,
        seeds=seeds,
        children_per_parent=args.children_per_parent,  # 5 by default
        out_file="generated_unlearning_losses_iter1.txt",
        snapshot_file="initial_iter1.txt",
        no_retain=args.no_retain
    )
    refined_loss_ids = sorted(refined_functions.keys())

    # Train/Eval refined losses and write iteration1*.txt
    os.makedirs(args.output_dir_iter1, exist_ok=True)
    results_iter1: List[Dict[str, Any]] = []
    detailed_for_topk_iter1: List[Dict[str, Any]] = []
    per_parent_details: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    loss_file_iter1 = get_results_dir() / "generated_unlearning_losses_iter1.txt"

    # helper to get integer parent index from mapping (keys may be str)
    def _parent_idx_for(loss_id_int: int, mapping: Dict[int, Dict[str, int]]) -> int:
        meta = mapping.get(loss_id_int) or mapping.get(str(loss_id_int))
        if isinstance(meta, dict):
            return int(meta.get("parent_index", 0))
        return 0

    for loss_id in refined_loss_ids:
        selected_loss_fn = refined_functions[loss_id]
        loss_def = get_loss_fn_definition_from_file(loss_file_iter1, loss_id)
        parsed_epochs = parse_epochs_from_docstring(selected_loss_fn, default=1)
        epochs_this = 2 if args.debug else parsed_epochs
        if args.debug:
            print(f"[train] [refine] loss_id={loss_id} DEBUG mode: overriding epochs to 2 (docstring was {parsed_epochs})")
        else:
            print(f"[train] [refine] loss_id={loss_id} will train for epochs={epochs_this} (from docstring)")

        try:
            record = _train_one_loss(
                base_model_name=args.model_name,
                loss_id=loss_id,
                selected_loss_fn=selected_loss_fn,
                tokenizer=tokenizer,
                device=train_device,
                train_dtype=train_dtype,
                autocast_context=autocast_context,
                eval_device_idx=eval_device_idx,
                forget_dataset=forget_dataset,
                retain_dataset=retain_dataset,
                collator=collator,
                output_dir_root=args.output_dir_iter1,
                epochs=epochs_this,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                pad_token_id=pad_token_id,
                ref_caches=ref_caches,
                retain_logs_path=args.retain_logs_path,
                no_retain=args.no_retain,
            )
            results_iter1.append(record)

            # Write to global iteration1.txt
            write_current_summary(
                loss_id=loss_id,
                loss_fn_definition=loss_def,
                loss_history=record["loss_history"],
                tofu_summary_json_path=record.get("tofu_summary_json", ""),
                tofu_eval_json_path=record.get("tofu_eval_json", ""),
                precomputed_metrics=record.get("metrics_block", {}),
                filename="iteration1.txt",
            )

            # Collect detail used for overall top-k
            detail = {
                "loss_id": loss_id,
                "loss_fn_definition": loss_def,
                "loss_history": record["loss_history"],
                "open_unlearning_summary": record.get("metrics_block", {}),
                "topk_score": record.get("topk_score", float("-inf")),
            }
            detailed_for_topk_iter1.append(detail)

            # Per-parent summary file: iteration1_{i}.txt
            pidx = _parent_idx_for(loss_id, refine_meta)
            write_current_summary(
                loss_id=loss_id,
                loss_fn_definition=loss_def,
                loss_history=record["loss_history"],
                tofu_summary_json_path=record.get("tofu_summary_json", ""),
                tofu_eval_json_path=record.get("tofu_eval_json", ""),
                precomputed_metrics=record.get("metrics_block", {}),
                filename=f"iteration1_{pidx}.txt",
            )
            per_parent_details[pidx].append(detail)

        except Exception as e:
            print(f"[train] ERROR while training refined loss_id={loss_id}: {e}")
            # Write error to iteration1.txt and the appropriate per-parent file
            _record_error_entry(loss_id, loss_def, filename="iteration1.txt", error=e)
            pidx = _parent_idx_for(loss_id, refine_meta)
            _record_error_entry(loss_id, loss_def, filename=f"iteration1_{pidx}.txt", error=e)

        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    # Top among children (overall) — failed ones were omitted
    sorted_iter1 = sorted(detailed_for_topk_iter1, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
    top_children = sorted_iter1[: max(1, args.iter1_top_k)]
    append_top_k_metadata(top_children, filename="iteration1.txt")

    print("\n[train] Iteration-1 Top children:")
    for t in top_children:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    # Also append per-parent top-k metadata
    for pidx, lst in per_parent_details.items():
        lst_sorted = sorted(lst, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
        append_top_k_metadata(lst_sorted, filename=f"iteration1_{pidx}.txt")

    # -----------------------
    # Iteration 2: top-3 from iteration1 -> 10 children each (30 total)
    # -----------------------
    seeds2: List[Dict[str, Any]] = []
    for c in top_children:
        seeds2.append({
            "parent_loss_id": c["loss_id"],
            "loss_fn_definition": c["loss_fn_definition"],
            "loss_history": c["loss_history"],
            "open_unlearning_summary": c["open_unlearning_summary"],
        })

    refined_functions2, refine_meta2 = run_refine_and_load_loss_functions(
        repo_root=repo_root,
        seeds=seeds2,
        children_per_parent=args.iter2_children_per_parent,  # 10 by default
        out_file="generated_unlearning_losses_iter2.txt",
        snapshot_file="initial_iter2.txt",
        no_retain=args.no_retain,
    )
    refined_loss_ids2 = sorted(refined_functions2.keys())

    os.makedirs(args.output_dir_iter2, exist_ok=True)
    detailed_for_topk_iter2: List[Dict[str, Any]] = []
    per_parent_details2: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    loss_file_iter2 = get_results_dir() / "generated_unlearning_losses_iter2.txt"

    for loss_id in refined_loss_ids2:
        selected_loss_fn = refined_functions2[loss_id]
        loss_def = get_loss_fn_definition_from_file(loss_file_iter2, loss_id)
        parsed_epochs = parse_epochs_from_docstring(selected_loss_fn, default=1)
        epochs_this = 2 if args.debug else parsed_epochs
        if args.debug:
            print(f"[train] [refine2] loss_id={loss_id} DEBUG mode: overriding epochs to 2 (docstring was {parsed_epochs})")
        else:
            print(f"[train] [refine2] loss_id={loss_id} will train for epochs={epochs_this} (from docstring)")

        try:
            record = _train_one_loss(
                base_model_name=args.model_name,
                loss_id=loss_id,
                selected_loss_fn=selected_loss_fn,
                tokenizer=tokenizer,
                device=train_device,
                train_dtype=train_dtype,
                autocast_context=autocast_context,
                eval_device_idx=eval_device_idx,
                forget_dataset=forget_dataset,
                retain_dataset=retain_dataset,
                collator=collator,
                output_dir_root=args.output_dir_iter2,
                epochs=epochs_this,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                pad_token_id=pad_token_id,
                ref_caches=ref_caches,
                retain_logs_path=args.retain_logs_path,
                no_retain=args.no_retain,
            )

            write_current_summary(
                loss_id=loss_id,
                loss_fn_definition=loss_def,
                loss_history=record["loss_history"],
                tofu_summary_json_path=record.get("tofu_summary_json", ""),
                tofu_eval_json_path=record.get("tofu_eval_json", ""),
                precomputed_metrics=record.get("metrics_block", {}),
                filename="iteration2.txt",
            )

            detail = {
                "loss_id": loss_id,
                "loss_fn_definition": loss_def,
                "loss_history": record["loss_history"],
                "open_unlearning_summary": record.get("metrics_block", {}),
                "topk_score": record.get("topk_score", float("-inf")),
            }
            detailed_for_topk_iter2.append(detail)

            pidx = _parent_idx_for(loss_id, refine_meta2)
            write_current_summary(
                loss_id=loss_id,
                loss_fn_definition=loss_def,
                loss_history=record["loss_history"],
                tofu_summary_json_path=record.get("tofu_summary_json", ""),
                tofu_eval_json_path=record.get("tofu_eval_json", ""),
                precomputed_metrics=record.get("metrics_block", {}),
                filename=f"iteration2_{pidx}.txt",
            )
            per_parent_details2[pidx].append(detail)

        except Exception as e:
            print(f"[train] ERROR while training refined loss_id={loss_id} (iter2): {e}")
            _record_error_entry(loss_id, loss_def, filename="iteration2.txt", error=e)
            pidx = _parent_idx_for(loss_id, refine_meta2)
            _record_error_entry(loss_id, loss_def, filename=f"iteration2_{pidx}.txt", error=e)

        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    sorted_iter2 = sorted(detailed_for_topk_iter2, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
    top_children2 = sorted_iter2[: max(1, args.iter2_top_k)]
    append_top_k_metadata(top_children2, filename="iteration2.txt")

    print("\n[train] Iteration-2 Top children:")
    for t in top_children2:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    for pidx, lst in per_parent_details2.items():
        lst_sorted = sorted(lst, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
        append_top_k_metadata(lst_sorted, filename=f"iteration2_{pidx}.txt")

    # -----------------------
    # Iteration 3: top-5 from iteration2 -> 3 children each (15 total)
    # -----------------------
    seeds3: List[Dict[str, Any]] = []
    for c in top_children2:
        seeds3.append({
            "parent_loss_id": c["loss_id"],
            "loss_fn_definition": c["loss_fn_definition"],
            "loss_history": c["loss_history"],
            "open_unlearning_summary": c["open_unlearning_summary"],
        })

    refined_functions3, refine_meta3 = run_refine_and_load_loss_functions(
        repo_root=repo_root,
        seeds=seeds3,
        children_per_parent=args.iter3_children_per_parent,  # 3 by default
        out_file="generated_unlearning_losses_iter3.txt",
        snapshot_file="initial_iter3.txt",
        no_retain=args.no_retain,
    )
    refined_loss_ids3 = sorted(refined_functions3.keys())

    os.makedirs(args.output_dir_iter3, exist_ok=True)
    detailed_for_topk_iter3: List[Dict[str, Any]] = []
    per_parent_details3: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    loss_file_iter3 = get_results_dir() / "generated_unlearning_losses_iter3.txt"

    for loss_id in refined_loss_ids3:
        selected_loss_fn = refined_functions3[loss_id]
        loss_def = get_loss_fn_definition_from_file(loss_file_iter3, loss_id)
        parsed_epochs = parse_epochs_from_docstring(selected_loss_fn, default=1)
        epochs_this = 2 if args.debug else parsed_epochs
        if args.debug:
            print(f"[train] [refine3] loss_id={loss_id} DEBUG mode: overriding epochs to 2 (docstring was {parsed_epochs})")
        else:
            print(f"[train] [refine3] loss_id={loss_id} will train for epochs={epochs_this} (from docstring)")

        try:
            record = _train_one_loss(
                base_model_name=args.model_name,
                loss_id=loss_id,
                selected_loss_fn=selected_loss_fn,
                tokenizer=tokenizer,
                device=train_device,
                train_dtype=train_dtype,
                autocast_context=autocast_context,
                eval_device_idx=eval_device_idx,
                forget_dataset=forget_dataset,
                retain_dataset=retain_dataset,
                collator=collator,
                output_dir_root=args.output_dir_iter3,
                epochs=epochs_this,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                pad_token_id=pad_token_id,
                ref_caches=ref_caches,
                retain_logs_path=args.retain_logs_path,
                no_retain=args.no_retain,
            )

            write_current_summary(
                loss_id=loss_id,
                loss_fn_definition=loss_def,
                loss_history=record["loss_history"],
                tofu_summary_json_path=record.get("tofu_summary_json", ""),
                tofu_eval_json_path=record.get("tofu_eval_json", ""),
                precomputed_metrics=record.get("metrics_block", {}),
                filename="iteration3.txt",
            )

            detail = {
                "loss_id": loss_id,
                "loss_fn_definition": loss_def,
                "loss_history": record["loss_history"],
                "open_unlearning_summary": record.get("metrics_block", {}),
                "topk_score": record.get("topk_score", float("-inf")),
            }
            detailed_for_topk_iter3.append(detail)

            pidx = _parent_idx_for(loss_id, refine_meta3)
            write_current_summary(
                loss_id=loss_id,
                loss_fn_definition=loss_def,
                loss_history=record["loss_history"],
                tofu_summary_json_path=record.get("tofu_summary_json", ""),
                tofu_eval_json_path=record.get("tofu_eval_json", ""),
                precomputed_metrics=record.get("metrics_block", {}),
                filename=f"iteration3_{pidx}.txt",
            )
            per_parent_details3[pidx].append(detail)

        except Exception as e:
            print(f"[train] ERROR while training refined loss_id={loss_id} (iter3): {e}")
            _record_error_entry(loss_id, loss_def, filename="iteration3.txt", error=e)
            pidx = _parent_idx_for(loss_id, refine_meta3)
            _record_error_entry(loss_id, loss_def, filename=f"iteration3_{pidx}.txt", error=e)

        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    # Top list for iteration3 (for convenience)
    sorted_iter3 = sorted(detailed_for_topk_iter3, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
    top_children3 = sorted_iter3[: max(1, args.iter3_top_k)]
    append_top_k_metadata(top_children3, filename="iteration3.txt")

    print("\n[train] Iteration-3 Top children:")
    for t in top_children3:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    for pidx, lst in per_parent_details3.items():
        lst_sorted = sorted(lst, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
        append_top_k_metadata(lst_sorted, filename=f"iteration3_{pidx}.txt")

    _write_overall_summary(get_results_dir())


if __name__ == "__main__":
    main()
