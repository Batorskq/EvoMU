# utils_books.py
#
# Utilities for MUSE Books LoRA unlearning search.
#
# KEY CHANGE (Option A):
#   per_sample_log_probs() returns PER-TOKEN MEAN log-probability (not sum),
#   and precompute_ref_log_probs_forget() caches the same normalized value.
#
# Additional change (requested now):
#   Use generate_books.py (per-token-mean-aware prompts) instead of generate.py.

import os
import sys
import re
import json
import math
import time
import gc
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# -----------------------
# Results directory handling (must match generate_books.py behavior)
# -----------------------

def _make_timestamp_dir(base: Path) -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = base / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_results_dir() -> Path:
    """
    Resolve the run directory for artifacts:
    - If UNLEARN_RESULTS_DIR is set, use it.
    - Otherwise create results/<timestamp> under repo root.
    """
    repo_root = Path(__file__).resolve().parent
    env = os.environ.get("UNLEARN_RESULTS_DIR", "").strip()
    if env:
        rd = Path(env)
        rd.mkdir(parents=True, exist_ok=True)
        return rd
    base = repo_root / "results"
    base.mkdir(parents=True, exist_ok=True)
    return _make_timestamp_dir(base)


# -----------------------
# Data: JSONL QA dataset + causal collator
# -----------------------

def _coalesce_text(obj: Dict[str, Any]) -> str:
    if "prompt" in obj and "completion" in obj:
        return f"{obj.get('prompt','')}{obj.get('completion','')}"
    if "instruction" in obj and "response" in obj:
        return f"{obj.get('instruction','')}\n{obj.get('response','')}"
    if "question" in obj and "answer" in obj:
        q = obj.get("question", "")
        a = obj.get("answer", "")
        return f"Q: {q}\nA: {a}"
    if "input" in obj and "output" in obj:
        return f"{obj.get('input','')}\n{obj.get('output','')}"
    if "text" in obj:
        return str(obj.get("text", ""))

    parts = []
    for k, v in obj.items():
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    if parts:
        return "\n".join(parts)
    return json.dumps(obj, ensure_ascii=False)


class JsonlQADataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.examples: List[Dict[str, Any]] = []

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"JSONL not found: {path}")

        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
                text = _coalesce_text(obj)

                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_attention_mask=True,
                    add_special_tokens=True,
                )
                self.examples.append(
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "idx": len(self.examples),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.examples[i]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "idx": int(ex["idx"]),
        }


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch passed to collator.")

        input_ids = [b["input_ids"] for b in batch]
        attn = [b["attention_mask"] for b in batch]
        idxs = [int(b["idx"]) for b in batch]

        max_len = max(int(x.shape[0]) for x in input_ids)

        padded_ids = []
        padded_attn = []
        for ids, m in zip(input_ids, attn):
            pad_len = max_len - int(ids.shape[0])
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)], dim=0)
                m = torch.cat([m, torch.zeros((pad_len,), dtype=torch.long)], dim=0)
            padded_ids.append(ids)
            padded_attn.append(m)

        return {
            "input_ids": torch.stack(padded_ids, dim=0),
            "attention_mask": torch.stack(padded_attn, dim=0),
            "idx": torch.tensor(idxs, dtype=torch.long),
        }


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# -----------------------
# Log-prob computation (Option A fix lives here)
# -----------------------

def _token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    IMPORTANT: this MUST allow gradients during training.
    Do NOT decorate with @torch.no_grad().
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B, T, V]

    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_mask = attention_mask[:, 1:].to(dtype=torch.bool)

    # compute in fp32 for stability; cast back to logits dtype
    logp = torch.log_softmax(logits.float(), dim=-1)
    gathered = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    return gathered.to(dtype=logits.dtype), token_mask



def per_sample_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    _ = pad_token_id
    token_logp, token_mask = _token_logprobs(model, input_ids, attention_mask)
    lengths = token_mask.sum(dim=1).clamp(min=1).to(token_logp.dtype)
    summed = (token_logp * token_mask.to(token_logp.dtype)).sum(dim=1)
    mean_logp = summed / lengths
    return mean_logp


@torch.no_grad()
def precompute_ref_log_probs_forget(
    model_name: str,
    forget_dataset: JsonlQADataset,
    pad_token_id: int,
    device: torch.device,
    dtype_for_model: torch.dtype,
    collator: CausalCollator,
    batch_size: int = 1,
) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype_for_model)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.eval()

    loader = DataLoader(
        forget_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    out = torch.empty((len(forget_dataset),), dtype=torch.float32, device="cpu")

    for batch in loader:
        idx = batch["idx"].clone().cpu().long()
        batch_dev = to_device(batch, device)
        scores = per_sample_log_probs(
            model,
            batch_dev["input_ids"],
            batch_dev["attention_mask"],
            pad_token_id=pad_token_id,
        ).detach().float().cpu()
        out[idx] = scores

    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out


# -----------------------
# Loss function generation / refinement integration
# -----------------------

def _ensure_import_torch_in_text(code: str) -> str:
    code_strip = code.lstrip()
    if code_strip.startswith("import torch"):
        return code
    return "import torch\n\n" + code


def _rewrite_common_invalid_calls(code: str) -> str:
    code = re.sub(r"\btorch\.softplus\s*\(", "torch.nn.functional.softplus(", code)
    return code


def _load_loss_functions_from_file(path: Path) -> Dict[int, Callable]:
    txt = path.read_text(encoding="utf-8")
    txt = _ensure_import_torch_in_text(txt)
    txt = _rewrite_common_invalid_calls(txt)

    g: Dict[str, Any] = {"torch": torch}
    l: Dict[str, Any] = {}

    try:
        exec(compile(txt, str(path), "exec"), g, l)
    except Exception as e:
        raise RuntimeError(f"Failed to exec loss file {path}: {e}") from e

    merged = {}
    merged.update(g)
    merged.update(l)

    out: Dict[int, Callable] = {}
    for k, v in merged.items():
        if not callable(v):
            continue
        m = re.fullmatch(r"loss_fn_(\d+)", str(k))
        if not m:
            continue
        out[int(m.group(1))] = v

    if not out:
        raise RuntimeError(f"No loss_fn_* functions found in {path}")
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def run_generate_and_load_loss_functions(repo_root: Path, n_losses: int = 20) -> Dict[int, Callable]:
    """
    Calls generate_books.py --mode initial to create generated_unlearning_losses.txt in the run dir,
    then loads and returns {loss_id: fn}.
    """
    results_dir = get_results_dir()
    out_file = "generated_unlearning_losses.txt"
    snapshot_file = "initial.txt"

    gen_py = Path(repo_root) / "generate_books.py"
    if not gen_py.exists():
        raise FileNotFoundError(f"generate_books.py not found at: {gen_py}")

    out_path = results_dir / out_file
    if not out_path.exists():
        cmd = [
            sys.executable,
            str(gen_py),
            "--mode", "initial",
            "--N", str(int(n_losses)),
            "--out_file", out_file,
            "--snapshot_file", snapshot_file,
        ]
        print("[utils_books] Running initial loss generation (generate_books.py):")
        print("  ", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(repo_root), env=os.environ.copy())
        if proc.returncode != 0:
            raise RuntimeError(f"generate_books.py initial failed with return code {proc.returncode}")

    return _load_loss_functions_from_file(out_path)


def run_refine_and_load_loss_functions(
    repo_root: Path,
    seeds: List[Dict[str, Any]],
    children_per_parent: int,
    out_file: str,
    snapshot_file: str,
) -> Tuple[Dict[int, Callable], Dict[Any, Dict[str, Any]]]:
    """
    Calls generate_books.py --mode refine with a seeds JSON written into the run dir.
    Returns:
      - refined_functions: {global_loss_id: fn}
      - refine_meta: mapping {global_loss_id: {"parent_index": int, "child_index": int}}
    """
    results_dir = get_results_dir()
    gen_py = Path(repo_root) / "generate_books.py"
    if not gen_py.exists():
        raise FileNotFoundError(f"generate_books.py not found at: {gen_py}")

    seeds_path = results_dir / "seeds_tmp.json"
    seeds_path.write_text(json.dumps(seeds, indent=2, ensure_ascii=False), encoding="utf-8")

    cmd = [
        sys.executable,
        str(gen_py),
        "--mode", "refine",
        "--seeds_file", str(seeds_path),
        "--children_per_parent", str(int(children_per_parent)),
        "--out_file", out_file,
        "--snapshot_file", snapshot_file,
    ]
    print("[utils_books] Running refine generation (generate_books.py):")
    print("  ", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"generate_books.py refine failed with return code {proc.returncode}")

    out_path = results_dir / out_file
    refined = _load_loss_functions_from_file(out_path)

    mapping_path = results_dir / "refine_mapping.json"
    refine_meta: Dict[Any, Dict[str, Any]] = {}
    if mapping_path.exists():
        try:
            refine_meta = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to parse refine_mapping.json: {e}") from e

    return refined, refine_meta


def get_loss_fn_definition_from_file(loss_file: Path, loss_id: int) -> str:
    txt = loss_file.read_text(encoding="utf-8")

    pat = re.compile(rf"(?m)^(def\s+loss_fn_{int(loss_id)}\s*\(.*?\)\s*:\s*)")
    m = pat.search(txt)
    if not m:
        return ""

    start = m.start(1)

    pat_next = re.compile(r"(?m)^def\s+loss_fn_\d+\s*\(")
    m2 = pat_next.search(txt, pos=m.end(1))
    end = m2.start() if m2 else len(txt)

    return txt[start:end].rstrip()


def parse_epochs_from_docstring(fn: Callable, default: int = 1) -> int:
    doc = getattr(fn, "__doc__", None) or ""
    m = re.search(r"epochs\s*:\s*(\d+)", doc)
    if not m:
        return int(default)
    k = int(m.group(1))
    if 1 <= k <= 10:
        return k
    return int(default)


# -----------------------
# Eval snapshot merging
# -----------------------

def prepare_eval_snapshot(
    base_model_name: str,
    lora_dir: str,
    save_dir: str,
    dtype: torch.dtype,
    device: torch.device,
) -> str:
    if PeftModel is None:
        raise RuntimeError("peft is not available; cannot merge LoRA adapters.")

    save_dir = str(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    merged_dir = os.path.join(save_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.eval()

    peft_model = PeftModel.from_pretrained(model, lora_dir)
    peft_model.to(device)
    peft_model.eval()

    merged = peft_model.merge_and_unload()
    merged.eval()

    merged.save_pretrained(merged_dir, safe_serialization=True)

    try:
        merged.to("cpu")
    except Exception:
        pass
    del merged
    del peft_model
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return merged_dir


# -----------------------
# Iteration summary writers
# -----------------------

def _load_summary_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def write_current_summary(
    loss_id: int,
    loss_fn_definition: str,
    loss_history: List[float],
    tofu_summary_json_path: str,
    tofu_eval_json_path: str,
    precomputed_metrics: Dict[str, Any],
    filename: str,
) -> None:
    results_dir = get_results_dir()
    out_path = results_dir / filename

    obj = _load_summary_file(out_path)
    obj[str(int(loss_id))] = {
        "loss_fn_definition": loss_fn_definition,
        "loss_history": loss_history,
        "open_unlearning_summary": precomputed_metrics,
        "tofu_summary_json_path": tofu_summary_json_path,
        "tofu_eval_json_path": tofu_eval_json_path,
    }

    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_top_k_metadata(top_k_list: List[Dict[str, Any]], filename: str) -> None:
    results_dir = get_results_dir()
    out_path = results_dir / filename
    obj = _load_summary_file(out_path)

    cleaned = []
    for r in top_k_list:
        cleaned.append(
            {
                "loss_id": int(r.get("loss_id")),
                "loss_fn_definition": r.get("loss_fn_definition", ""),
                "loss_history": r.get("loss_history", []),
                "open_unlearning_summary": r.get("open_unlearning_summary", {}),
                "topk_score": r.get("topk_score", None),
            }
        )

    obj["__top_k__"] = cleaned
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
