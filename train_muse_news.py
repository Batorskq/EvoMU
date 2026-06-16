# train_muse_news.py
#
# LoRA unlearning search with Open-Unlearning MUSE News scoring (NON-CHAT: Llama-2-7b-hf).
#
# Key difference vs your chat version:
#   - Open-Unlearning "model" key is Llama-2-7b-hf (NOT Llama-2-7b-chat-hf)
#   - The tokenizer baked into merged_for_eval snapshots is meta-llama/Llama-2-7b-hf
#   - The OU model-key picker defaults to NON-CHAT for MUSE-news_target
#
# IMPORTANT NOTE (tokenizer/eval determinism):
#   If you evaluate a merged checkpoint directory that contains tokenizer files,
#   eval pipelines often load the tokenizer from that directory. If those
#   tokenizer files differ from what your Open-Unlearning config expects
#   (chat template / BOS/EOS behavior), MUSE metrics can change dramatically
#   even when the *weights are identical*.
#
# This script therefore overwrites the tokenizer inside each merged_for_eval
# snapshot with a "known-good" eval tokenizer (default: meta-llama/Llama-2-7b-hf),
# so that running with optimizer.step() commented out yields the same scores as
# evaluating the original base model via the same Open-Unlearning config.
#
# - Uses MUSE News benchmark (via experiment=eval/muse/default by default).
# - Default base model (to unlearn from) is the official MUSE News target:
#       model_name="muse-bench/MUSE-news_target"
# - Training loop is structurally identical to your previous script (iterations 0..3).
#
# Top-k selection metric:
#   forget_group_good = mean(VerbMem_forget_good, KnowMem_forget_good, PrivLeak_forget_good)
#   retain_good       = KnowMem_retain_good
#   score             = 0.5 * forget_group_good + 0.5 * retain_good
#f
#   where:
#       VerbMem_forget_good   = 1 - clamp01(VerbMem_forget_raw)
#       KnowMem_forget_good   = 1 - clamp01(KnowMem_forget_raw)
#       PrivLeak_forget_good  = f(|PrivLeak_forget_raw|)  (BEST when PrivLeak is near 0; large magnitude is bad)
#       KnowMem_retain_good   = clamp01(KnowMem_retain_raw)
#
# - All four raw metrics are extracted from MUSE_EVAL.json produced by Open-Unlearning eval.
# - Data defaults point to a MUSE-News-style dataset:
#       data_dir="muse_news_data"
#       forget_split="forget_train.jsonl"
#       retain_split="retain_train.jsonl"

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

from utils import (
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

    print(f"[train-muse] Loading FORGET split from: {forget_path}")
    print(f"[train-muse] Loading RETAIN split from: {retain_path}")

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
    print("[train-muse] Precomputing reference log-probs (forget & retain)...")
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
        print(f"[train-muse] WARNING: snapshot_dir does not exist: {snapshot_dir}")
        return

    try:
        tok = _load_tokenizer_best_effort(
            primary_name=base_model_name,  # likely fails for muse-bench/MUSE-news_target (no tokenizer on hub)
            fallback_name=eval_tokenizer_name,
            secondary_fallback_name=secondary_fallback_tokenizer_name,
        )
    except Exception as e:
        print(f"[train-muse] WARNING: could not load any tokenizer to overwrite snapshot: {e}")
        return

    # Ensure pad token exists
    if getattr(tok, "pad_token", None) is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token

    _wipe_tokenizer_artifacts(snap)

    try:
        tok.save_pretrained(str(snap))
        print("[train-muse] Overwrote eval snapshot tokenizer with:", getattr(tok, "name_or_path", "<unknown>"))
        ct = getattr(tok, "chat_template", None)
        print("[train-muse] Snapshot tokenizer chat_template present:", ct is not None)
    except Exception as e:
        print(f"[train-muse] WARNING: failed to save tokenizer into snapshot_dir: {e}")


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

    # IMPORTANT FIX:
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
    """
    IMPORTANT: This NON-CHAT version defaults to Llama-2-7b-hf.

    The single biggest cause of your mismatch earlier was selecting the chat model key
    while comparing against non-chat target numbers.
    """
    if not path_in:
        return "Llama-2-7b-hf"

    low = path_in.lower()
    base = os.path.basename(path_in).lower()

    # MUSE news target is a Llama-2-7B derivative; target numbers you want are non-chat.
    if ("muse-news_target" in low) or ("muse-news_target" in base) or ("muse-news_target" in low.replace("-", "_")):
        return "Llama-2-7b-hf"

    # If user points explicitly to a chat checkpoint path
    if ("7b-chat" in low) or ("llama-2-7b-chat" in low) or ("llama2-7b-chat" in low):
        return "Llama-2-7b-chat-hf"

    # Llama 3 family shortcuts (kept for compatibility)
    if ("llama-3.2-1b-instruct" in low) or ("llama-3.2-1b-instruct" in base):
        return "Llama-3.2-1B-Instruct"
    if ("llama-3.2-3b-instruct" in low) or ("llama-3.2-3b-instruct" in base):
        return "Llama-3.2-3B-Instruct"
    if ("llama-3.1-8b-instruct" in low) or ("llama-3.1-8b-instruct" in base):
        return "Llama-3.1-8B-Instruct"

    # Default for "llama-2-7b" (non-chat)
    if ("llama-2-7b" in low) or ("llama2-7b" in low) or ("7b" in low and "llama" in low):
        return "Llama-2-7b-hf"

    return "Llama-2-7b-hf"


def run_muse_eval(
    ckpt_dir: str,
    device_idx: int = 0,
    task_name: Optional[str] = None,
    experiment: str = "eval/muse/default",
    base_model_name: Optional[str] = None,
    ou_model_key: Optional[str] = None,
) -> Dict[str, Any]:
    ou_root = os.path.join(REPO_DIR, "ou")
    eval_py = os.path.join(ou_root, "src", "eval.py")
    if not os.path.exists(eval_py):
        raise FileNotFoundError(f"[run_muse_eval] eval.py not found at {eval_py}")

    model_key = ou_model_key or _pick_model_key_from_path(base_model_name or ckpt_dir)

    if task_name is None:
        task_name = f"MUSE_NEWS_{int(time.time() * 1000)}"

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

    print("[run_muse_eval] Launching Open-Unlearning MUSE News eval with args:")
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

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
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
        print(f"\n[train-muse] loss_id={loss_id} | Epoch {epoch + 1}/{epochs}")
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
                    idx_f = torch.as_tensor(batch_forget["idx"], dtype=torch.long, device=torch.device("cpu"))
                    ref_f = ref_caches["forget"][idx_f]
                    kwargs["ref_log_probs_forget"] = ref_f.to(device=device, dtype=log_probs_forget.dtype)

                if needs_ref_retain and ref_caches is not None:
                    idx_r = torch.as_tensor(batch_retain["idx"], dtype=torch.long, device=torch.device("cpu"))
                    ref_r = ref_caches["retain"][idx_r]
                    kwargs["ref_log_probs_retain"] = ref_r.to(device=device, dtype=log_probs_retain.dtype)

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

            if (step + 1) % 10 == 0 or (step + 1) == min_len:
                print(f"[train-muse] loss_id={loss_id} step {step + 1}/{min_len} loss={last_loss_value:.4f}")

        epoch_final_losses.append(last_loss_value if last_loss_value is not None else float("nan"))

    final_dir = os.path.join(output_dir_root, f"loss_{loss_id}")
    os.makedirs(final_dir, exist_ok=True)
    lora_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train-muse] Completed LoRA training for loss_id={loss_id}. Saved to {final_dir}")

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
    print(f"[train-muse] Merged model for eval saved to {eval_snapshot_dir_abs}")

    # --- CRITICAL: stabilize tokenizer inside the eval snapshot ---
    _overwrite_eval_tokenizer_in_snapshot(
        snapshot_dir=eval_snapshot_dir_abs,
        base_model_name=base_model_name,
        eval_tokenizer_name=eval_tokenizer_name,
        secondary_fallback_tokenizer_name=eval_tokenizer_fallback,
    )
    # ------------------------------------------------------------

    print("[train-muse] Launching MUSE News evaluation ...")
    run_info = run_muse_eval(
        ckpt_dir=eval_snapshot_dir_abs,
        device_idx=eval_device_idx,
        experiment=muse_experiment,
        base_model_name=base_model_name,
        ou_model_key=ou_model_key,
    )
    print("[train-muse] MUSE eval finished.")
    print(json.dumps(run_info, indent=4))

    muse_eval_json_path = run_info.get("muse_eval_json", "")
    try:
        if muse_eval_json_path and os.path.isfile(muse_eval_json_path):
            with open(muse_eval_json_path, "r", encoding="utf-8") as f:
                eval_obj: Dict[str, Any] = json.load(f)
        else:
            raise FileNotFoundError(f"MUSE_EVAL.json not found at {muse_eval_json_path}")
    except Exception as e:
        raise RuntimeError(f"[train-muse] Failed to load/parse MUSE_EVAL.json: {e}") from e

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
            out.append({"loss_id": loss_id, "topk_score": None, "error": metrics.get("error"), "open_unlearning_summary": metrics})
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
    print(f"[train-muse] Wrote overall summary to {out_path}")


# -----------------------
# Slug helper for results dirs
# -----------------------


def _slugify_name(val: str) -> str:
    base = os.path.basename(val)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", base)
    slug = slug.strip("-_.")
    return slug or "default"


# -----------------------
# Main
# -----------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA unlearning search with Open-Unlearning MUSE News scoring (NON-CHAT: Llama-2-7b-hf).\n"
            "Top-k selection metric:\n"
            "  mean( mean(VerbMem, KnowMem, PrivLeak on D_forget) [after inversion/centering], KnowMem on D_retain ).\n"
            "NOTE: optimizer.step() is commented out."
        )
    )

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--model_name",
        type=str,
        default="muse-bench/MUSE-news_target",
        help="Base model to unlearn from (default is the MUSE News target weights).",
    )

    # Non-chat defaults
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer used for training data tokenization (NON-CHAT).",
    )
    parser.add_argument(
        "--eval_tokenizer_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer to write into merged_for_eval snapshots (must match OU Llama-2-7b-hf config).",
    )
    parser.add_argument(
        "--eval_tokenizer_fallback",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Secondary fallback tokenizer if eval_tokenizer_name cannot be loaded.",
    )

    # Force OU model key if you want absolute control (recommended)
    parser.add_argument(
        "--ou_model_key",
        type=str,
        default="Llama-2-7b-hf",
        help="Open-Unlearning model key to use for eval (set to Llama-2-7b-hf for non-chat).",
    )

    parser.add_argument("--data_dir", type=str, default="muse_news_data")
    parser.add_argument("--forget_split", type=str, default="forget_train.jsonl")
    parser.add_argument("--retain_split", type=str, default="retain_train.jsonl")

    parser.add_argument("--output_dir", type=str, default="unlearned_lora_models_muse_news")
    parser.add_argument("--output_dir_iter1", type=str, default="unlearned_lora_models_muse_news_iter1")
    parser.add_argument("--output_dir_iter2", type=str, default="unlearned_lora_models_muse_news_iter2")
    parser.add_argument("--output_dir_iter3", type=str, default="unlearned_lora_models_muse_news_iter3")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--iter0_top_k", type=int, default=5)
    parser.add_argument("--children_per_parent", type=int, default=5)
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
        default="eval/muse/default",
        help="Hydra experiment key for Open-Unlearning MUSE evaluation.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    repo_root = Path(__file__).resolve().parent

    print(f"[train-muse] Base model: {args.model_name}")
    print(f"[train-muse] Train tokenizer: {args.tokenizer_name}")
    print(f"[train-muse] Eval snapshot tokenizer: {args.eval_tokenizer_name}")
    print(f"[train-muse] Eval snapshot tokenizer fallback: {args.eval_tokenizer_fallback}")
    print(f"[train-muse] Open-Unlearning model key: {args.ou_model_key}")
    print(f"[train-muse] Data dir: {args.data_dir}")
    print(f"[train-muse] MUSE-News splits: forget={args.forget_split} | retain={args.retain_split}")
    print(f"[train-muse] MUSE experiment key: {args.muse_experiment}")

    model_slug = _slugify_name(args.model_name)
    forget_slug = _slugify_name(os.path.splitext(os.path.basename(args.forget_split))[0])
    retain_slug = _slugify_name(os.path.splitext(os.path.basename(args.retain_split))[0])

    base_results = repo_root / "results" / model_slug / f"{forget_slug}__{retain_slug}" / "muse_news"
    base_results.mkdir(parents=True, exist_ok=True)
    run_dir = base_results / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ["UNLEARN_RESULTS_DIR"] = str(run_dir)
    print(f"[train-muse] Using results directory: {run_dir.resolve()}")

    loss_functions = run_generate_and_load_loss_functions(repo_root)
    if not loss_functions:
        raise ValueError("No loss functions were generated.")

    all_loss_ids = sorted(loss_functions.keys())
    if args.limit_losses and args.limit_losses > 0:
        all_loss_ids = all_loss_ids[: args.limit_losses]

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

    os.makedirs(args.output_dir, exist_ok=True)

    detailed_for_topk_iter0: List[Dict[str, Any]] = []
    loss_file_iter0 = get_results_dir() / "generated_unlearning_losses.txt"

    for loss_id in all_loss_ids:
        selected_loss_fn = loss_functions[loss_id]
        loss_def = get_loss_fn_definition_from_file(loss_file_iter0, loss_id)

        parsed_epochs = parse_epochs_from_docstring(selected_loss_fn, default=1)
        epochs_this = 1 if args.debug else parsed_epochs
        print(f"[train-muse] loss_id={loss_id} will train for epochs={epochs_this} (docstring={parsed_epochs}, debug={args.debug})")

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
                output_dir_root=args.output_dir,
                epochs=epochs_this,
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
                filename="iteration0.txt",
            )

            detailed_for_topk_iter0.append(
                {
                    "loss_id": loss_id,
                    "loss_fn_definition": loss_def,
                    "loss_history": record["loss_history"],
                    "open_unlearning_summary": record.get("metrics_block", {}),
                    "topk_score": record.get("topk_score", float("-inf")),
                }
            )

        except Exception as e:
            print(f"[train-muse] ERROR while training loss_id={loss_id}: {e}")
            _record_error_entry(loss_id, loss_def, filename="iteration0.txt", error=e)

        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    sorted_iter0 = sorted(detailed_for_topk_iter0, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
    parents = sorted_iter0[: max(1, args.iter0_top_k)]

    append_top_k_metadata(parents, filename="iteration0.txt")
    print("\n[train-muse] Iteration-0 Top parents:")
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

    refined_functions, refine_meta = run_refine_and_load_loss_functions(
        repo_root=repo_root,
        seeds=seeds,
        children_per_parent=args.children_per_parent,
        out_file="generated_unlearning_losses_iter1.txt",
        snapshot_file="initial_iter1.txt",
    )
    refined_loss_ids = sorted(refined_functions.keys())

    os.makedirs(args.output_dir_iter1, exist_ok=True)
    detailed_for_topk_iter1: List[Dict[str, Any]] = []
    per_parent_details: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    loss_file_iter1 = get_results_dir() / "generated_unlearning_losses_iter1.txt"

    def _parent_idx_for(loss_id_int: int, mapping: Dict[Any, Dict[str, Any]]) -> int:
        meta = mapping.get(loss_id_int) or mapping.get(str(loss_id_int))
        if isinstance(meta, dict):
            return int(meta.get("parent_index", 0))
        return 0

    for loss_id in refined_loss_ids:
        selected_loss_fn = refined_functions[loss_id]
        loss_def = get_loss_fn_definition_from_file(loss_file_iter1, loss_id)
        parsed_epochs = parse_epochs_from_docstring(selected_loss_fn, default=1)
        epochs_this = 2 if args.debug else parsed_epochs

        print(f"[train-muse] [refine] loss_id={loss_id} epochs={epochs_this} (docstring={parsed_epochs}, debug={args.debug})")

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
                output_dir_root=args.output_dir_iter1,
                epochs=epochs_this,
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
                filename="iteration1.txt",
            )

            detail = {
                "loss_id": loss_id,
                "loss_fn_definition": loss_def,
                "loss_history": record["loss_history"],
                "open_unlearning_summary": record.get("metrics_block", {}),
                "topk_score": record.get("topk_score", float("-inf")),
            }
            detailed_for_topk_iter1.append(detail)

            pidx = _parent_idx_for(loss_id, refine_meta)
            write_current_summary(
                loss_id=loss_id,
                loss_fn_definition=loss_def,
                loss_history=record["loss_history"],
                tofu_summary_json_path="",
                tofu_eval_json_path="",
                precomputed_metrics=record.get("metrics_block", {}),
                filename=f"iteration1_{pidx}.txt",
            )
            per_parent_details[pidx].append(detail)

        except Exception as e:
            print(f"[train-muse] ERROR while training refined loss_id={loss_id}: {e}")
            _record_error_entry(loss_id, loss_def, filename="iteration1.txt", error=e)
            pidx = _parent_idx_for(loss_id, refine_meta)
            _record_error_entry(loss_id, loss_def, filename=f"iteration1_{pidx}.txt", error=e)

        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    sorted_iter1 = sorted(detailed_for_topk_iter1, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
    top_children = sorted_iter1[: max(1, args.iter1_top_k)]
    append_top_k_metadata(top_children, filename="iteration1.txt")

    print("\n[train-muse] Iteration-1 Top children:")
    for t in top_children:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    for pidx, lst in per_parent_details.items():
        lst_sorted = sorted(lst, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
        append_top_k_metadata(lst_sorted, filename=f"iteration1_{pidx}.txt")

    seeds2: List[Dict[str, Any]] = []
    for c in top_children:
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

        print(f"[train-muse] [refine2] loss_id={loss_id} epochs={epochs_this} (docstring={parsed_epochs}, debug={args.debug})")

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
                output_dir_root=args.output_dir_iter2,
                epochs=epochs_this,
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
                tofu_summary_json_path="",
                tofu_eval_json_path="",
                precomputed_metrics=record.get("metrics_block", {}),
                filename=f"iteration2_{pidx}.txt",
            )
            per_parent_details2[pidx].append(detail)

        except Exception as e:
            print(f"[train-muse] ERROR while training refined loss_id={loss_id} (iter2): {e}")
            _record_error_entry(loss_id, loss_def, filename="iteration2.txt", error=e)
            pidx = _parent_idx_for(loss_id, refine_meta2)
            _record_error_entry(loss_id, loss_def, filename=f"iteration2_{pidx}.txt", error=e)

        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    sorted_iter2 = sorted(detailed_for_topk_iter2, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
    top_children2 = sorted_iter2[: max(1, args.iter2_top_k)]
    append_top_k_metadata(top_children2, filename="iteration2.txt")

    print("\n[train-muse] Iteration-2 Top children:")
    for t in top_children2:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    for pidx, lst in per_parent_details2.items():
        lst_sorted = sorted(lst, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
        append_top_k_metadata(lst_sorted, filename=f"iteration2_{pidx}.txt")

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

        print(f"[train-muse] [refine3] loss_id={loss_id} epochs={epochs_this} (docstring={parsed_epochs}, debug={args.debug})")

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
                output_dir_root=args.output_dir_iter3,
                epochs=epochs_this,
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
                tofu_summary_json_path="",
                tofu_eval_json_path="",
                precomputed_metrics=record.get("metrics_block", {}),
                filename=f"iteration3_{pidx}.txt",
            )
            per_parent_details3[pidx].append(detail)

        except Exception as e:
            print(f"[train-muse] ERROR while training refined loss_id={loss_id} (iter3): {e}")
            _record_error_entry(loss_id, loss_def, filename="iteration3.txt", error=e)
            pidx = _parent_idx_for(loss_id, refine_meta3)
            _record_error_entry(loss_id, loss_def, filename=f"iteration3_{pidx}.txt", error=e)

        gc.collect()
        if train_device.type == "cuda":
            torch.cuda.empty_cache()

    sorted_iter3 = sorted(detailed_for_topk_iter3, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
    top_children3 = sorted_iter3[: max(1, args.iter3_top_k)]
    append_top_k_metadata(top_children3, filename="iteration3.txt")

    print("\n[train-muse] Iteration-3 Top children:")
    for t in top_children3:
        print({"loss_id": t["loss_id"], "topk_score": t["topk_score"]})

    for pidx, lst in per_parent_details3.items():
        lst_sorted = sorted(lst, key=lambda r: r.get("topk_score", float("-inf")), reverse=True)
        append_top_k_metadata(lst_sorted, filename=f"iteration3_{pidx}.txt")

    _write_overall_summary(get_results_dir())


if __name__ == "__main__":
    main()
