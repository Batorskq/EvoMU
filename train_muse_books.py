# train_muse_books.py
#
# LoRA unlearning search with Open-Unlearning MUSE Books scoring.
#
# NOTE (Option A implemented via utils_books.py):
#   per_sample_log_probs() now returns PER-TOKEN MEAN log-probs (not sums),
#   and reference caches are computed in the same normalized space.
#   This fixes the "silently dominated by sequence length scale" issue for long sequences.

import os
import sys
import json
import gc
import re
import argparse
import traceback
import math
import time
import subprocess
from pathlib import Path
from contextlib import nullcontext
from typing import List, Dict, Any, Optional, Sequence
from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import LoraConfig, get_peft_model

# CHANGED: import from utils_books instead of utils
from utils_books import (
    run_generate_and_load_loss_functions,
    run_refine_and_load_loss_functions,
    get_loss_fn_definition_from_file,
    JsonlQADataset,
    CausalCollator,
    per_sample_log_probs,
    to_device,
    precompute_ref_log_probs_forget,
    prepare_eval_snapshot,
    write_current_summary,
    append_top_k_metadata,
    parse_epochs_from_docstring,
    get_results_dir,
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
    forget_path = os.path.join(data_dir, forget_split)
    retain_path = os.path.join(data_dir, retain_split)

    print(f"[train-muse-books] Loading FORGET split from: {forget_path}")
    print(f"[train-muse-books] Loading RETAIN split from: {retain_path}")

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
    print("[train-muse-books] Precomputing reference log-probs (forget & retain)...")
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

    ref_forget = ref_forget.cpu()
    ref_retain = ref_retain.cpu()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"forget": ref_forget, "retain": ref_retain}


# -----------------------
# Eval-tokenizer stabilization helpers
# -----------------------

_TOKENIZER_ARTIFACTS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "merges.txt",
]


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _wipe_tokenizer_artifacts(dir_path: Path) -> None:
    for fname in _TOKENIZER_ARTIFACTS:
        _safe_unlink(dir_path / fname)
    try:
        for p in dir_path.glob("tokenizer.*"):
            _safe_unlink(p)
    except Exception:
        pass


def _load_tokenizer_best_effort(
    primary_name: str,
    fallback_name: str,
    secondary_fallback_name: str,
) -> AutoTokenizer:
    last_err: Optional[BaseException] = None
    for name in [primary_name, fallback_name, secondary_fallback_name]:
        if not name:
            continue
        for use_fast in (True, False):
            try:
                tok = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
                return tok
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Failed to load tokenizer from any candidate. Last error: {last_err}")


def _overwrite_eval_tokenizer_in_snapshot(
    snapshot_dir: str,
    base_model_name: str,
    eval_tokenizer_name: str,
    secondary_fallback_tokenizer_name: str,
) -> None:
    snap = Path(snapshot_dir)
    if not snap.exists():
        print(f"[train-muse-books] WARNING: snapshot_dir does not exist: {snapshot_dir}")
        return

    try:
        tok = _load_tokenizer_best_effort(
            primary_name=base_model_name,  # likely fails for muse-bench targets (no tokenizer on hub)
            fallback_name=eval_tokenizer_name,
            secondary_fallback_name=secondary_fallback_tokenizer_name,
        )
    except Exception as e:
        print(f"[train-muse-books] WARNING: could not load any tokenizer to overwrite snapshot: {e}")
        return

    if getattr(tok, "pad_token", None) is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token

    _wipe_tokenizer_artifacts(snap)

    try:
        tok.save_pretrained(str(snap))
        print("[train-muse-books] Overwrote eval snapshot tokenizer with:", getattr(tok, "name_or_path", "<unknown>"))
        ct = getattr(tok, "chat_template", None)
        print("[train-muse-books] Snapshot tokenizer chat_template present:", ct is not None)
    except Exception as e:
        print(f"[train-muse-books] WARNING: failed to save tokenizer into snapshot_dir: {e}")


# -----------------------
# MUSE metric utilities
# -----------------------


def _safe_clamp01(x: float, default: float = 0.0) -> float:
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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        xf = float(x)
    except Exception:
        return default
    if math.isnan(xf) or math.isinf(xf):
        return default
    return xf


def _privleak_good_from_raw(priv_raw: Any) -> float:
    """
    MUSE privleak is BEST when close to 0. Large positive OR large negative magnitude is bad.
    The raw value is often reported on a ~[-100, +100] scale (percent-like), but sometimes it can
    be already in [-1, 1]. We detect scale by magnitude:
      - if |x| > 2 -> treat as percent-like scale=100
      - else scale=1
    We then map |x| to [0,1] with a smooth function that never yields NaN:
        good = 1 / (1 + |x|/scale)
    """
    x = _safe_float(priv_raw, default=0.0)
    a = abs(x)
    scale = 100.0 if a > 2.0 else 1.0
    good = 1.0 / (1.0 + (a / scale))
    return _safe_clamp01(good, default=0.0)


def _muse_flatten_numeric_leafs(obj: Any, prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}

    if isinstance(obj, dict):
        if "agg_value" in obj and isinstance(obj["agg_value"], (int, float)):
            key = f"{prefix}.agg_value" if prefix else "agg_value"
            flat[key] = float(obj["agg_value"])
            return flat

        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (int, float)):
                flat[new_prefix] = float(v)
            else:
                flat.update(_muse_flatten_numeric_leafs(v, new_prefix))

    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            if isinstance(v, (int, float)):
                flat[new_prefix] = float(v)
            else:
                flat.update(_muse_flatten_numeric_leafs(v, new_prefix))

    return flat


def _strip_per_sample_details(obj: Any) -> Any:
    PER_SAMPLE_KEYS = {
        "value_by_index",
        "examples",
        "per_example",
        "per_sample",
        "per_sample_details",
    }

    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in PER_SAMPLE_KEYS:
                continue
            cleaned[k] = _strip_per_sample_details(v)
        return cleaned
    if isinstance(obj, list):
        return [_strip_per_sample_details(v) for v in obj]
    return obj


def _muse_key_tokens(key: str) -> set:
    separators = [".", "_", "-", "/", "[", "]"]
    tmp = key.lower()
    for sep in separators:
        tmp = tmp.replace(sep, " ")
    return set(tok for tok in tmp.split() if tok)


def _muse_find_metric(
    flat: Dict[str, float],
    include_token_sets: Sequence[Sequence[str]],
    description: str,
) -> float:
    best_key: Optional[str] = None
    best_score: float = -1.0

    for key, val in flat.items():
        tokens = _muse_key_tokens(key)

        for inc in include_token_sets:
            inc_set = set(tok.lower() for tok in inc)
            if not inc_set.issubset(tokens):
                continue

            score = float(len(inc_set))
            if key.endswith(".agg_value") or "agg_value" in tokens:
                score += 0.1

            if score > best_score:
                best_score = score
                best_key = key

    if best_key is None:
        raise ValueError(
            f"[MUSE] Could not find metric for '{description}'. "
            f"Tried token patterns: {include_token_sets}. "
            "Please inspect MUSE_EVAL.json and adjust the extractor if needed."
        )

    return flat[best_key]


def _extract_muse_core_metrics(eval_obj: Dict[str, Any]) -> Dict[str, float]:
    metrics_root = eval_obj.get("metrics", eval_obj)
    flat = _muse_flatten_numeric_leafs(metrics_root)

    verb_forget = _muse_find_metric(flat, include_token_sets=[["verbmem", "forget"]], description="VerbMem on D_forget")
    know_forget = _muse_find_metric(flat, include_token_sets=[["knowmem", "forget"]], description="KnowMem on D_forget")
    priv_forget = _muse_find_metric(
        flat,
        include_token_sets=[
            ["privleak", "forget"],
            ["priv", "leak", "forget"],
            ["privleak"],
            ["priv", "leak"],
        ],
        description="PrivLeak on D_forget",
    )
    know_retain = _muse_find_metric(flat, include_token_sets=[["knowmem", "retain"]], description="KnowMem on D_retain")

    return {
        "muse_verbmem_forget": float(verb_forget),
        "muse_knowmem_forget": float(know_forget),
        "muse_privleak_forget": float(priv_forget),
        "muse_knowmem_retain": float(know_retain),
    }


def _compute_muse_good_metrics(muse_core: Dict[str, float]) -> Dict[str, float]:
    v_f_raw = muse_core.get("muse_verbmem_forget", 0.0)
    k_f_raw = muse_core.get("muse_knowmem_forget", 0.0)
    p_f_raw = muse_core.get("muse_privleak_forget", 0.0)
    k_r_raw = muse_core.get("muse_knowmem_retain", 0.0)

    v_f_good = 1.0 - _safe_clamp01(v_f_raw)
    k_f_good = 1.0 - _safe_clamp01(k_f_raw)

    # PrivLeak is best near 0 (large magnitude is bad). Do NOT treat negative as "good".
    p_f_good = _privleak_good_from_raw(p_f_raw)

    k_r_good = _safe_clamp01(k_r_raw)

    return {
        "muse_verbmem_forget_good": v_f_good,
        "muse_knowmem_forget_good": k_f_good,
        "muse_privleak_forget_good": p_f_good,
        "muse_knowmem_retain_good": k_r_good,
    }


def _compute_muse_topk_score(muse_metrics: Dict[str, Any]) -> float:
    if "muse_verbmem_forget_good" in muse_metrics:
        v_f_good = _safe_clamp01(muse_metrics.get("muse_verbmem_forget_good", 0.0))
        k_f_good = _safe_clamp01(muse_metrics.get("muse_knowmem_forget_good", 0.0))
        p_f_good = _safe_clamp01(muse_metrics.get("muse_privleak_forget_good", 0.0))
        k_r_good = _safe_clamp01(muse_metrics.get("muse_knowmem_retain_good", 0.0))
    else:
        v_f_raw = muse_metrics.get("muse_verbmem_forget", 0.0)
        k_f_raw = muse_metrics.get("muse_knowmem_forget", 0.0)
        p_f_raw = muse_metrics.get("muse_privleak_forget", 0.0)
        k_r_raw = muse_metrics.get("muse_knowmem_retain", 0.0)

        v_f_good = 1.0 - _safe_clamp01(v_f_raw)
        k_f_good = 1.0 - _safe_clamp01(k_f_raw)
        p_f_good = _privleak_good_from_raw(p_f_raw)
        k_r_good = _safe_clamp01(k_r_raw)

    forget_group = (v_f_good + k_f_good + p_f_good) / 3.0
    return 0.5 * forget_group + 0.5 * k_r_good


# -----------------------
# Open-Unlearning MUSE launcher
# -----------------------

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _pick_model_key_from_path(path_in: str) -> str:
    if not path_in:
        return "Llama-2-7b-hf"

    low = path_in.lower()

    if ("muse-books_target" in low) or ("muse_books_target" in low) or ("books_target" in low):
        return "Llama-2-7b-hf"

    if ("7b-chat" in low) or ("llama-2-7b-chat" in low) or ("llama2-7b-chat" in low):
        return "Llama-2-7b-chat-hf"

    if ("llama-2-13b" in low) or ("13b" in low and "llama" in low):
        return "Llama-2-13b-hf"

    if ("llama-2-7b" in low) or ("llama2-7b" in low) or ("7b" in low and "llama" in low):
        return "Llama-2-7b-hf"

    return "Llama-2-7b-hf"


def run_muse_eval(
    ckpt_dir: str,
    device_idx: int = 0,
    task_name: Optional[str] = None,
    experiment: str = "eval/muse/books",
    base_model_name: Optional[str] = None,
    ou_model_key: Optional[str] = None,
) -> Dict[str, Any]:
    ou_root = os.path.join(REPO_DIR, "ou")
    eval_py = os.path.join(ou_root, "src", "eval.py")
    if not os.path.exists(eval_py):
        raise FileNotFoundError(f"[run_muse_eval] eval.py not found at {eval_py}")

    model_key = ou_model_key or _pick_model_key_from_path(base_model_name or ckpt_dir)

    if task_name is None:
        task_name = f"MUSE_BOOKS_{int(time.time() * 1000)}"

    eval_dir = os.path.join(ou_root, "saves", "eval", task_name)

    args = [
        sys.executable,
        eval_py,
        "--config-name=eval.yaml",
        f"experiment={experiment}",
        f"model={model_key}",
        f"model.model_args.pretrained_model_name_or_path={ckpt_dir}",
        "model.model_args.attn_implementation=sdpa",
        f"task_name={task_name}",
    ]

    print("[run_muse_eval] Launching Open-Unlearning MUSE Books eval with args:")
    for a in args:
        print("   ", a)

    env = os.environ.copy()
    current = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not current:
        env["CUDA_VISIBLE_DEVICES"] = str(int(device_idx) if device_idx is not None else 0)

    proc = subprocess.Popen(args, cwd=ou_root, env=env)
    ret = proc.wait()

    muse_eval_json = os.path.join(eval_dir, "MUSE_EVAL.json")
    muse_summary_json = os.path.join(eval_dir, "MUSE_SUMMARY.json")

    print("[run_muse_eval] Return code:", ret)
    if os.path.exists(eval_dir):
        print(f"[run_muse_eval] Results dir: {eval_dir}")
        if os.path.exists(muse_eval_json):
            print(f"[run_muse_eval] MUSE_EVAL.json: {muse_eval_json}")
        else:
            print("[run_muse_eval] WARNING: MUSE_EVAL.json not found in eval dir.")
        if os.path.exists(muse_summary_json):
            print(f"[run_muse_eval] MUSE_SUMMARY.json: {muse_summary_json}")
    else:
        print("[run_muse_eval] WARNING: expected eval dir not found:", eval_dir)

    return {
        "retcode": ret,
        "experiment": experiment,
        "model_key": model_key,
        "eval_dir": eval_dir,
        "muse_eval_json": muse_eval_json,
        "muse_summary_json": muse_summary_json,
        "task_name": task_name,
    }


# -----------------------
# Training for a loss (MUSE-scored)
# -----------------------

def _train_one_loss_muse(
    base_model_name: str,
    loss_id: int,
    selected_loss_fn,
    tokenizer,
    device: torch.device,
    train_dtype: torch.dtype,
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
    ref_caches: Optional[Dict[str, torch.Tensor]],
    muse_experiment: str,
    eval_tokenizer_name: str,
    eval_tokenizer_fallback: str,
    ou_model_key: Optional[str],
) -> Dict[str, Any]:
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

    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.to(device)
    lora_model.train()

    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]

    # -----------------------------
    # Equal batch sizes + cycling
    # -----------------------------
    forget_n = len(forget_dataset)
    retain_n = len(retain_dataset)
    if forget_n == 0 or retain_n == 0:
        raise ValueError("[train-muse-books] forget or retain dataset is empty.")

    effective_bs = min(batch_size, forget_n, retain_n)
    if effective_bs <= 0:
        raise ValueError("[train-muse-books] effective batch size computed as <= 0 (unexpected).")

    if effective_bs != batch_size:
        print(f"[train-muse-books] Adjusting batch_size {batch_size} -> {effective_bs} (dataset too small).")

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=effective_bs,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=effective_bs,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"[train-muse-books] Data sizes: forget_dataset={forget_n} retain_dataset={retain_n} "
        f"| effective_batch_size={effective_bs} | forget_batches={len(forget_loader)} retain_batches={len(retain_loader)}"
    )

    steps_per_epoch = max(len(forget_loader), len(retain_loader))
    if steps_per_epoch == 0:
        raise ValueError(
            "[train-muse-books] No training steps (a DataLoader has length 0). "
            "Check dataset sizes and effective batch size."
        )

    total_steps = epochs * steps_per_epoch

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    epoch_final_losses: List[float] = []
    last_loss_value: Optional[float] = None

    for epoch in range(epochs):
        print(f"\n[train-muse-books] loss_id={loss_id} | Epoch {epoch + 1}/{epochs}")

        forget_it = iter(forget_loader)
        retain_it = iter(retain_loader)

        for step in range(steps_per_epoch):
            try:
                batch_forget = next(forget_it)
            except StopIteration:
                forget_it = iter(forget_loader)
                batch_forget = next(forget_it)

            try:
                batch_retain = next(retain_it)
            except StopIteration:
                retain_it = iter(retain_loader)
                batch_retain = next(retain_it)

            batch_forget_dev = to_device(batch_forget, device)
            batch_retain_dev = to_device(batch_retain, device)

            with autocast_context():
                # per_sample_log_probs() is already PER-TOKEN MEAN in utils_books.py
                log_probs_forget = per_sample_log_probs(
                    lora_model,
                    batch_forget_dev["input_ids"],
                    batch_forget_dev["attention_mask"],
                    pad_token_id=pad_token_id,
                )
                log_probs_retain = per_sample_log_probs(
                    lora_model,
                    batch_retain_dev["input_ids"],
                    batch_retain_dev["attention_mask"],
                    pad_token_id=pad_token_id,
                )

                kwargs: Dict[str, Any] = {
                    "log_probs_forget": log_probs_forget,
                    "log_probs_retain": log_probs_retain,
                }

                if needs_ref_forget and ref_caches is not None:
                    idx_f = torch.as_tensor(batch_forget["idx"], dtype=torch.long, device=torch.device("cpu"))
                    ref_f = ref_caches["forget"][idx_f]
                    kwargs["ref_log_probs_forget"] = ref_f.to(device=device, dtype=log_probs_forget.dtype)

                if needs_ref_retain and ref_caches is not None:
                    idx_r = torch.as_tensor(batch_retain["idx"], dtype=torch.long, device=torch.device("cpu"))
                    ref_r = ref_caches["retain"][idx_r]
                    kwargs["ref_log_probs_retain"] = ref_r.to(device=device, dtype=log_probs_retain.dtype)

                # Safety: truncate batch-wise tensors to a common n if something upstream changes.
                def _batch_dim(x: Any) -> Optional[int]:
                    if isinstance(x, torch.Tensor) and x.ndim >= 1:
                        return int(x.shape[0])
                    return None

                dims = []
                for k in ("log_probs_forget", "log_probs_retain", "ref_log_probs_forget", "ref_log_probs_retain"):
                    if k in kwargs:
                        d = _batch_dim(kwargs[k])
                        if d is not None:
                            dims.append(d)
                if dims:
                    n = min(dims)
                    for k in ("log_probs_forget", "log_probs_retain", "ref_log_probs_forget", "ref_log_probs_retain"):
                        if k in kwargs and isinstance(kwargs[k], torch.Tensor) and kwargs[k].ndim >= 1:
                            if int(kwargs[k].shape[0]) != n:
                                kwargs[k] = kwargs[k][:n]

                loss = selected_loss_fn(**kwargs)

            if not isinstance(loss, torch.Tensor):
                loss = torch.as_tensor(loss, device=device, dtype=log_probs_forget.dtype)
            if loss.ndim != 0:
                loss = loss.mean()
            if not loss.requires_grad:
                loss = loss + 0.0 * (log_probs_forget.mean() + log_probs_retain.mean())

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(trainable_params, grad_clip_norm)

            optimizer.step()
            scheduler.step()
            last_loss_value = float(loss.item())

            if (step + 1) % 10 == 0 or (step + 1) == steps_per_epoch:
                print(f"[train-muse-books] loss_id={loss_id} step {step + 1}/{steps_per_epoch} loss={last_loss_value:.6f}")

        epoch_final_losses.append(last_loss_value if last_loss_value is not None else float("nan"))

    final_dir = os.path.join(output_dir_root, f"loss_{loss_id}")
    os.makedirs(final_dir, exist_ok=True)
    lora_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train-muse-books] Completed LoRA training for loss_id={loss_id}. Saved to {final_dir}")

    try:
        lora_model.to("cpu")
    except Exception:
        pass
    del lora_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    print(f"[train-muse-books] Merged model for eval saved to {eval_snapshot_dir_abs}")

    # --- CRITICAL: stabilize tokenizer inside the eval snapshot ---
    _overwrite_eval_tokenizer_in_snapshot(
        snapshot_dir=eval_snapshot_dir_abs,
        base_model_name=base_model_name,
        eval_tokenizer_name=eval_tokenizer_name,
        secondary_fallback_tokenizer_name=eval_tokenizer_fallback,
    )

    print("[train-muse-books] Launching MUSE Books evaluation ...")
    run_info = run_muse_eval(
        ckpt_dir=eval_snapshot_dir_abs,
        device_idx=eval_device_idx,
        experiment=muse_experiment,
        base_model_name=base_model_name,
        ou_model_key=ou_model_key,
    )
    print("[train-muse-books] MUSE eval finished.")
    print(json.dumps(run_info, indent=4))

    muse_eval_json_path = run_info.get("muse_eval_json", "")
    try:
        if muse_eval_json_path and os.path.isfile(muse_eval_json_path):
            with open(muse_eval_json_path, "r", encoding="utf-8") as f:
                eval_obj: Dict[str, Any] = json.load(f)
        else:
            raise FileNotFoundError(f"MUSE_EVAL.json not found at {muse_eval_json_path}")
    except Exception as e:
        raise RuntimeError(f"[train-muse-books] Failed to load/parse MUSE_EVAL.json: {e}") from e

    muse_core = _extract_muse_core_metrics(eval_obj)
    muse_good = _compute_muse_good_metrics(muse_core)

    metrics_block: Dict[str, Any] = _strip_per_sample_details(eval_obj) if isinstance(eval_obj, dict) else {}

    for k, v in muse_core.items():
        metrics_block[k] = float(v)
    for k, v in muse_good.items():
        metrics_block[k] = float(v)

    topk_score = _compute_muse_topk_score({**muse_core, **muse_good})
    metrics_block["muse_assessment_score"] = float(topk_score)

    return {
        "loss_id": loss_id,
        "loss_history": epoch_final_losses,
        "muse_eval_json": muse_eval_json_path,
        "muse_summary_json": run_info.get("muse_summary_json", ""),
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
            out.append(
                {"loss_id": loss_id, "topk_score": None, "error": metrics.get("error"), "open_unlearning_summary": metrics}
            )
        else:
            score = _compute_muse_topk_score(metrics if isinstance(metrics, dict) else {})
            score = score if (isinstance(score, (int, float)) and math.isfinite(score)) else None
            out.append({"loss_id": loss_id, "topk_score": score, "open_unlearning_summary": metrics})

    def _sort_key(d):
        sc = d.get("topk_score")
        return (0, -(sc if isinstance(sc, (int, float)) else 0.0)) if sc is not None else (1, 0.0)

    out.sort(key=_sort_key)
    return out


def _write_overall_summary(results_dir: Path) -> None:
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
    print(f"[train-muse-books] Wrote overall summary to {out_path}")


# -----------------------
# Slug helper for results dirs
# -----------------------

def _slugify_name(val: str) -> str:
    base = os.path.basename(val)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", base)
    slug = slug.strip("-_.")
    return slug or "default"


# -----------------------
# Stage runner helpers (to avoid duplicated code)
# -----------------------

def _parent_idx_for(loss_id_int: int, mapping: Dict[Any, Dict[str, Any]]) -> int:
    meta = mapping.get(loss_id_int) or mapping.get(str(loss_id_int))
    if isinstance(meta, dict):
        return int(meta.get("parent_index", 0))
    return 0


def _train_loss_set(
    *,
    functions: Dict[int, Any],
    loss_file: Path,
    stage_filename: str,
    output_dir_root: str,
    args,
    tokenizer,
    train_device,
    train_dtype,
    autocast_context,
    eval_device_idx,
    forget_dataset,
    retain_dataset,
    collator,
    pad_token_id,
    ref_caches,
    refine_meta: Optional[Dict[Any, Dict[str, Any]]] = None,
    write_per_parent_files: bool = False,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir_root, exist_ok=True)

    detailed: List[Dict[str, Any]] = []
    per_parent_details: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    loss_ids = sorted(functions.keys())
    if args.limit_losses and args.limit_losses > 0:
        loss_ids = loss_ids[: args.limit_losses]

    for loss_id in loss_ids:
        selected_loss_fn = functions[loss_id]
        loss_def = get_loss_fn_definition_from_file(loss_file, loss_id)

        parsed_epochs = parse_epochs_from_docstring(selected_loss_fn, default=1)
        epochs_this = parsed_epochs

        print(f"[train-muse-books] [{stage_filename}] loss_id={loss_id} epochs={epochs_this}")

        try:
            record = _train_one_loss_muse(
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
                output_dir_root=output_dir_root,
                epochs=(1 if args.debug else epochs_this),
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                pad_token_id=pad_token_id,
                ref_caches=ref_caches,
                muse_experiment=args.muse_experiment,
                eval_tokenizer_name=args.eval_tokenizer_name,
                eval_tokenizer_fallback=args.eval_tokenizer_fallback,
                ou_model_key=args.ou_model_key,
            )

            write_current_summary(
                loss_id=loss_id,
                loss_fn_definition=loss_def,
                loss_history=record["loss_history"],
                tofu_summary_json_path="",
                tofu_eval_json_path="",
                precomputed_metrics=record.get("metrics_block", {}),
                filename=stage_filename,
            )

            detail = {
                "loss_id": loss_id,
                "loss_fn_definition": loss_def,
                "loss_history": record["loss_history"],
                "open_unlearning_summary": record.get("metrics_block", {}),
                "topk_score": record.get("topk_score", float("-inf")),
            }
            detailed.append(detail)

            if write_per_parent_files and refine_meta is not None:
                pidx = _parent_idx_for(loss_id, refine_meta)
                write_current_summary(
                    loss_id=loss_id,
                    loss_fn_definition=loss_def,
                    loss_history=record["loss_history"],
                    tofu_summary_json_path="",
                    tofu_eval_json_path="",
                    precomputed_metrics=record.get("metrics_block", {}),
                    filename=f"{os.path.splitext(stage_filename)[0]}_{pidx}.txt",
                )
                per_parent_details[pidx].append(detail)

        except Exception as e:
            print(f"[train-muse-books] ERROR while training loss_id={loss_id}: {e}")
            _record_error_entry(loss_id, loss_def, filename=stage_filename, error=e)

            if write_per_parent_files and refine_meta is not None:
                pidx = _parent_idx_for(loss_id, refine_meta)
                _record_error_entry(loss_id, loss_def, filename=f"{os.path.splitext(stage_filename)[0]}_{pidx}.txt", error=e)

        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    if write_per_parent_files:
        for pidx, lst in per_parent_details.items():
            lst_sorted = sorted(lst, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
            append_top_k_metadata(lst_sorted, filename=f"{os.path.splitext(stage_filename)[0]}_{pidx}.txt")

    return detailed


def _select_top_k(detailed: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    sorted_list = sorted(detailed, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
    return sorted_list[: max(1, k)]


# -----------------------
# Main
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA unlearning search with Open-Unlearning MUSE Books scoring.\n"
            "Top-k selection metric:\n"
            "  mean( mean(VerbMem, KnowMem, PrivLeak on D_forget) [after inversion/centering], KnowMem on D_retain ).\n"
        )
    )

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--model_name",
        type=str,
        default="muse-bench/MUSE-books_target",
        help="Base model to unlearn from (default is the MUSE Books target weights).",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer used for training data tokenization (Llama vocab 32k).",
    )
    parser.add_argument(
        "--eval_tokenizer_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer to write into merged_for_eval snapshots (stabilizes eval).",
    )
    parser.add_argument(
        "--eval_tokenizer_fallback",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Secondary fallback tokenizer if eval_tokenizer_name cannot be loaded.",
    )

    parser.add_argument("--ou_model_key", type=str, default="Llama-2-7b-hf")

    parser.add_argument("--data_dir", type=str, default="muse_books_data")
    parser.add_argument("--forget_split", type=str, default="forget_train.jsonl")
    parser.add_argument("--retain_split", type=str, default="retain_train.jsonl")

    parser.add_argument("--output_dir", type=str, default="unlearned_lora_models_muse_books")
    parser.add_argument("--output_dir_iter1", type=str, default="unlearned_lora_models_muse_books_iter1")
    parser.add_argument("--output_dir_iter2", type=str, default="unlearned_lora_models_muse_books_iter2")
    parser.add_argument("--output_dir_iter3", type=str, default="unlearned_lora_models_muse_books_iter3")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--iter0_top_k", type=int, default=10)
    parser.add_argument("--children_per_parent", type=int, default=10)
    parser.add_argument("--iter1_top_k", type=int, default=3)

    parser.add_argument("--iter2_children_per_parent", type=int, default=10)
    parser.add_argument("--iter2_top_k", type=int, default=5)
    parser.add_argument("--iter3_children_per_parent", type=int, default=3)
    parser.add_argument("--iter3_top_k", type=int, default=5)

    parser.add_argument("--limit_losses", type=int, default=0)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--muse_experiment",
        type=str,
        default="eval/muse/books",
        help="Hydra experiment key for Open-Unlearning MUSE Books evaluation.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    repo_root = Path(__file__).resolve().parent

    print(f"[train-muse-books] Base model: {args.model_name}")
    print(f"[train-muse-books] Train tokenizer: {args.tokenizer_name}")
    print(f"[train-muse-books] Eval snapshot tokenizer: {args.eval_tokenizer_name}")
    print(f"[train-muse-books] Eval snapshot tokenizer fallback: {args.eval_tokenizer_fallback}")
    print(f"[train-muse-books] Open-Unlearning model key: {args.ou_model_key}")
    print(f"[train-muse-books] Data dir: {args.data_dir}")
    print(f"[train-muse-books] MUSE-Books splits: forget={args.forget_split} | retain={args.retain_split}")
    print(f"[train-muse-books] MUSE experiment key: {args.muse_experiment}")

    model_slug = _slugify_name(args.model_name)
    forget_slug = _slugify_name(os.path.splitext(os.path.basename(args.forget_split))[0])
    retain_slug = _slugify_name(os.path.splitext(os.path.basename(args.retain_split))[0])

    base_results = repo_root / "results" / model_slug / f"{forget_slug}__{retain_slug}" / "muse_books"
    base_results.mkdir(parents=True, exist_ok=True)
    run_dir = base_results / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ["UNLEARN_RESULTS_DIR"] = str(run_dir)
    print(f"[train-muse-books] Using results directory: {run_dir.resolve()}")

    loss_functions = run_generate_and_load_loss_functions(repo_root)
    if not loss_functions:
        raise ValueError("No loss functions were generated.")

    train_device, train_dtype, eval_device_idx, autocast_context = _ensure_device_and_dtype(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    forget_dataset, retain_dataset, collator = _build_datasets_and_collator(
        tokenizer,
        args.data_dir,
        args.forget_split,
        args.retain_split,
        args.max_length,
    )

    ref_caches = _maybe_compute_ref_caches(
        model_name=args.model_name,
        device=train_device,
        dtype_for_model=train_dtype,
        collator=collator,
        pad_token_id=pad_token_id,
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
    )

    # -----------------------
    # Iteration 0
    # -----------------------
    loss_file_iter0 = get_results_dir() / "generated_unlearning_losses.txt"
    detailed_iter0 = _train_loss_set(
        functions=loss_functions,
        loss_file=loss_file_iter0,
        stage_filename="iteration0.txt",
        output_dir_root=args.output_dir,
        args=args,
        tokenizer=tokenizer,
        train_device=train_device,
        train_dtype=train_dtype,
        autocast_context=autocast_context,
        eval_device_idx=eval_device_idx,
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        collator=collator,
        pad_token_id=pad_token_id,
        ref_caches=ref_caches,
        refine_meta=None,
        write_per_parent_files=False,
    )

    parents = _select_top_k(detailed_iter0, args.iter0_top_k)
    append_top_k_metadata(parents, filename="iteration0.txt")

    print("\n[train-muse-books] Iteration-0 Top parents:")
    for t in parents:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    seeds: List[Dict[str, Any]] = []
    for p in parents:
        seeds.append(
            {
                "parent_loss_id": p["loss_id"],
                "loss_fn_definition": p["loss_fn_definition"],
                "loss_history": p["loss_history"],
                "open_unlearning_summary": p["open_unlearning_summary"],
            }
        )

    refined_functions1, refine_meta1 = run_refine_and_load_loss_functions(
        repo_root=repo_root,
        seeds=seeds,
        children_per_parent=args.children_per_parent,
        out_file="generated_unlearning_losses_iter1.txt",
        snapshot_file="initial_iter1.txt",
    )

    # -----------------------
    # Iteration 1
    # -----------------------
    loss_file_iter1 = get_results_dir() / "generated_unlearning_losses_iter1.txt"
    detailed_iter1 = _train_loss_set(
        functions=refined_functions1,
        loss_file=loss_file_iter1,
        stage_filename="iteration1.txt",
        output_dir_root=args.output_dir_iter1,
        args=args,
        tokenizer=tokenizer,
        train_device=train_device,
        train_dtype=train_dtype,
        autocast_context=autocast_context,
        eval_device_idx=eval_device_idx,
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        collator=collator,
        pad_token_id=pad_token_id,
        ref_caches=ref_caches,
        refine_meta=refine_meta1,
        write_per_parent_files=True,
    )

    top_children1 = _select_top_k(detailed_iter1, args.iter1_top_k)
    append_top_k_metadata(top_children1, filename="iteration1.txt")

    print("\n[train-muse-books] Iteration-1 Top children:")
    for t in top_children1:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    seeds2: List[Dict[str, Any]] = []
    for c in top_children1:
        seeds2.append(
            {
                "parent_loss_id": c["loss_id"],
                "loss_fn_definition": c["loss_fn_definition"],
                "loss_history": c["loss_history"],
                "open_unlearning_summary": c["open_unlearning_summary"],
            }
        )

    refined_functions2, refine_meta2 = run_refine_and_load_loss_functions(
        repo_root=repo_root,
        seeds=seeds2,
        children_per_parent=args.iter2_children_per_parent,
        out_file="generated_unlearning_losses_iter2.txt",
        snapshot_file="initial_iter2.txt",
    )

    # -----------------------
    # Iteration 2
    # -----------------------
    loss_file_iter2 = get_results_dir() / "generated_unlearning_losses_iter2.txt"
    detailed_iter2 = _train_loss_set(
        functions=refined_functions2,
        loss_file=loss_file_iter2,
        stage_filename="iteration2.txt",
        output_dir_root=args.output_dir_iter2,
        args=args,
        tokenizer=tokenizer,
        train_device=train_device,
        train_dtype=train_dtype,
        autocast_context=autocast_context,
        eval_device_idx=eval_device_idx,
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        collator=collator,
        pad_token_id=pad_token_id,
        ref_caches=ref_caches,
        refine_meta=refine_meta2,
        write_per_parent_files=True,
    )

    top_children2 = _select_top_k(detailed_iter2, args.iter2_top_k)
    append_top_k_metadata(top_children2, filename="iteration2.txt")

    print("\n[train-muse-books] Iteration-2 Top children:")
    for t in top_children2:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    seeds3: List[Dict[str, Any]] = []
    for c in top_children2:
        seeds3.append(
            {
                "parent_loss_id": c["loss_id"],
                "loss_fn_definition": c["loss_fn_definition"],
                "loss_history": c["loss_history"],
                "open_unlearning_summary": c["open_unlearning_summary"],
            }
        )

    refined_functions3, refine_meta3 = run_refine_and_load_loss_functions(
        repo_root=repo_root,
        seeds=seeds3,
        children_per_parent=args.iter3_children_per_parent,
        out_file="generated_unlearning_losses_iter3.txt",
        snapshot_file="initial_iter3.txt",
    )

    # -----------------------
    # Iteration 3
    # -----------------------
    loss_file_iter3 = get_results_dir() / "generated_unlearning_losses_iter3.txt"
    detailed_iter3 = _train_loss_set(
        functions=refined_functions3,
        loss_file=loss_file_iter3,
        stage_filename="iteration3.txt",
        output_dir_root=args.output_dir_iter3,
        args=args,
        tokenizer=tokenizer,
        train_device=train_device,
        train_dtype=train_dtype,
        autocast_context=autocast_context,
        eval_device_idx=eval_device_idx,
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        collator=collator,
        pad_token_id=pad_token_id,
        ref_caches=ref_caches,
        refine_meta=refine_meta3,
        write_per_parent_files=True,
    )

    top_children3 = _select_top_k(detailed_iter3, args.iter3_top_k)
    append_top_k_metadata(top_children3, filename="iteration3.txt")

    print("\n[train-muse-books] Iteration-3 Top children:")
    for t in top_children3:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    _write_overall_summary(get_results_dir())


if __name__ == "__main__":
    main()
