import os
import sys
import json
import gc
import time
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,  # re-export convenience
)

from peft import PeftModel

# Repo root (used by OU eval launcher and defaults)
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# ================================
# Results directory utilities
# ================================
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
    This is evaluated at call time so train.py can set the env before calls.
    """
    env = os.environ.get("UNLEARN_RESULTS_DIR", "").strip()
    if env:
        rd = Path(env)
        rd.mkdir(parents=True, exist_ok=True)
        return rd
    base = Path(REPO_DIR) / "results"
    base.mkdir(parents=True, exist_ok=True)
    return _make_timestamp_dir(base)


# ================================
# 0) Common helpers
# ================================
_EPOCHS_DOC_RE = re.compile(r'(?im)^\s*epochs?\s*:\s*(\d+)\s*$')

# Minimal parser/sanitizer (loss_fn_* blocks only)
_DEF_HEADER_START_RE = re.compile(r'(?m)^(\s*)def\s+(loss_fn_\d+)\s*\(')
_DEF_ANY_START_RE = re.compile(r'(?m)^(\s*)def\s+([A-Za-z_]\w*)\s*\(')


def parse_epochs_from_docstring(fn: Callable, default: int = 1) -> int:
    try:
        doc = fn.__doc__ or ""
    except Exception:
        doc = ""
    for line in (doc.splitlines() if isinstance(doc, str) else []):
        m = _EPOCHS_DOC_RE.match(line.strip())
        if m:
            try:
                val = int(m.group(1))
                return max(1, val)
            except Exception:
                break
    return max(1, int(default))


def _attach_epochs_attr(namespace: Dict[str, Any]) -> None:
    for name, obj in namespace.items():
        if callable(obj) and isinstance(name, str) and name.startswith("loss_fn_"):
            try:
                setattr(obj, "_epochs", parse_epochs_from_docstring(obj, default=1))
            except Exception:
                setattr(obj, "_epochs", 1)


def _extract_loss_fn_blocks_robust(code_block: str) -> List[Tuple[str, str, int]]:
    """
    Robustly extract top-level loss_fn_* definitions.
    Body collection stops on any dedent (indent <= header indent), not only on the next 'def'.
    This prevents stray top-level text from being included in a function body.
    """
    lines = code_block.splitlines()
    n = len(lines)
    i = 0
    blocks: List[Tuple[str, str, int]] = []
    while i < n:
        m = _DEF_HEADER_START_RE.match(lines[i])
        if not m:
            i += 1
            continue
        header_indent = len(m.group(1) or "")
        func_name = m.group(2)

        # Find end of header (line that closes signature with '):')
        header_end = i
        header_closed = False
        while header_end < n:
            if re.search(r'\):\s*(?:#.*)?$', lines[header_end]):
                header_closed = True
                break
            header_end += 1
        if not header_closed:
            i += 1
            continue

        # Collect body: only lines strictly more indented than header (or blank)
        j = header_end + 1
        body_lines: List[str] = []
        while j < n:
            line = lines[j]
            next_def = _DEF_ANY_START_RE.match(line)
            if next_def:
                indent_line = len(next_def.group(1) or "")
                if indent_line <= header_indent:
                    break

            stripped = line.lstrip()
            indent_len = len(line) - len(stripped)

            if stripped == "":
                body_lines.append(line)
            elif indent_len <= header_indent:
                break
            else:
                body_lines.append(line)

            j += 1

        body_text = "\n".join(body_lines).rstrip()
        blocks.append((func_name, body_text, header_indent))
        i = j
    return blocks


def _rebuild_with_standard_signature(blocks: List[Tuple[str, str, int]]) -> str:
    rebuilt = []
    for name, body, indent in blocks:
        indent_spaces = " " * indent
        header = (
            f"{indent_spaces}def {name}(log_probs_forget, log_probs_retain, "
            f"ref_log_probs_forget=None, ref_log_probs_retain=None):"
        )
        if body.strip():
            rebuilt.append(header + "\n" + body)
        else:
            # Mirror generate.py: ensure syntactically valid + returns a tensor-like value.
            rebuilt.append(header + "\n" + "    return (log_probs_forget.mean() - log_probs_retain.mean())")
    return "\n\n".join(rebuilt).strip()


def _keep_only_loss_def_blocks(code_block: str) -> str:
    blocks = _extract_loss_fn_blocks_robust(code_block)
    if not blocks:
        return ""
    return _rebuild_with_standard_signature(blocks)


def _normalize_numeric_names_in_order(code_block: str) -> str:
    blocks = _extract_loss_fn_blocks_robust(code_block)
    if not blocks:
        return code_block
    renumbered: List[Tuple[str, str, int]] = []
    for idx, (_old_name, body, indent) in enumerate(blocks, start=1):
        renumbered.append((f"loss_fn_{idx}", body, indent))
    return _rebuild_with_standard_signature(renumbered)


def _ensure_import_torch(code_block: str) -> str:
    header = "import torch\n\n"
    trimmed = code_block.lstrip()
    if trimmed.startswith("import torch"):
        return code_block
    return header + code_block


def _rewrite_scalar_max_min_to_torch(code: str) -> str:
    """
    Rewrite unsafe scalar max/min patterns into tensor-safe clamp calls.
    This catches both Python max/min and torch.max/min where one arg is a scalar 0/0.0.
    """
    code = re.sub(r'\bmax\s*\(\s*0+(?:\.0+)?\s*,\s*(.*?)\)', r'torch.clamp(\1, min=0.0)', code)
    code = re.sub(r'\bmax\s*\(\s*(.*?)\s*,\s*0+(?:\.0+)?\s*\)', r'torch.clamp(\1, min=0.0)', code)
    code = re.sub(r'\btorch\.max\s*\(\s*0+(?:\.0+)?\s*,\s*(.*?)\)', r'torch.clamp(\1, min=0.0)', code)
    code = re.sub(r'\btorch\.max\s*\(\s*(.*?)\s*,\s*0+(?:\.0+)?\s*\)', r'torch.clamp(\1, min=0.0)', code)
    code = re.sub(r'\bmin\s*\(\s*0+(?:\.0+)?\s*,\s*(.*?)\)', r'torch.clamp(\1, max=0.0)', code)
    code = re.sub(r'\bmin\s*\(\s*(.*?)\s*,\s*0+(?:\.0+)?\s*\)', r'torch.clamp(\1, max=0.0)', code)
    code = re.sub(r'\btorch\.min\s*\(\s*0+(?:\.0+)?\s*,\s*(.*?)\)', r'torch.clamp(\1, max=0.0)', code)
    code = re.sub(r'\btorch\.min\s*\(\s*(.*?)\s*,\s*0+(?:\.0+)?\s*\)', r'torch.clamp(\1, max=0.0)', code)
    return code


def _sanitize_code_for_exec(code_str: str) -> str:
    """
    Keep only top-level loss_fn_* definitions and ensure import torch.
    Compile once to catch any residual syntax errors before exec().
    """
    only_defs = _keep_only_loss_def_blocks(code_str)
    if not only_defs.strip():
        return "import torch\n"
    only_defs = _normalize_numeric_names_in_order(only_defs)
    only_defs = _rewrite_scalar_max_min_to_torch(only_defs)
    only_defs = _ensure_import_torch(only_defs)
    try:
        compile(only_defs, "<generated_unlearning_losses>", "exec")
    except SyntaxError as e:
        print("[sanitize] SyntaxError after sanitization:", e)
        return "import torch\n"
    return only_defs


def _wrap_to_scalar_loss(fn: Callable) -> Callable:
    """Ensure the loss is a differentiable scalar; preserve name & docstring."""
    import torch as _torch

    def wrapped(log_probs_forget, log_probs_retain,
                ref_log_probs_forget=None, ref_log_probs_retain=None):
        out = fn(log_probs_forget, log_probs_retain,
                 ref_log_probs_forget, ref_log_probs_retain)
        if not isinstance(out, _torch.Tensor):
            out = _torch.as_tensor(out, device=log_probs_forget.device, dtype=log_probs_forget.dtype)
        if out.ndim != 0:
            out = out.mean()
        if not out.requires_grad:
            out = out + 0.0 * (log_probs_forget.mean() + log_probs_retain.mean())
        return out

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


# ================================
# 1) Loss generation (generate.py)
# ================================
def run_generate_and_load_loss_functions(repo_root: Path, no_retain: bool = False) -> Dict[int, Callable]:
    results_dir = get_results_dir()
    loss_file = results_dir / "generated_unlearning_losses.txt"

    env = os.environ.copy()
    env["UNLEARN_RESULTS_DIR"] = str(results_dir)

    cmd = [sys.executable, str(repo_root / "generate.py"), "--mode", "initial"]
    if no_retain:
        cmd.append("--no_retain")

    subprocess.run(
        cmd,
        check=True,
        env=env,
    )

    code_str = loss_file.read_text(encoding="utf-8")
    code_str = _sanitize_code_for_exec(code_str)  # robust safety
    namespace = {"torch": torch}
    exec(code_str, namespace)

    _attach_epochs_attr(namespace)

    loss_fns: Dict[int, Callable] = {}
    for name, obj in namespace.items():
        if callable(obj) and name.startswith("loss_fn_"):
            suffix = name.split("loss_fn_")[1]
            if suffix.isdigit():
                loss_fns[int(suffix)] = _wrap_to_scalar_loss(obj)  # ensure scalar & grad
    return dict(sorted(loss_fns.items(), key=lambda kv: kv[0]))



def run_refine_and_load_loss_functions(
    repo_root: Path,
    seeds: List[Dict[str, Any]],
    children_per_parent: int = 5,
    out_file: str = "generated_unlearning_losses_iter1.txt",
    snapshot_file: str = "initial_iter1.txt",
    no_retain: bool = False,
) -> Tuple[Dict[int, Callable], Dict[int, Dict[str, int]]]:
    """
    Returns:
        (loss_functions_dict, mapping_dict)
    mapping_dict: { global_loss_id (int) : {"parent_index": int, "child_index": int} }
    """
    results_dir = get_results_dir()
    seeds_path = results_dir / ".refine_seeds.json"
    seeds_path.write_text(json.dumps(seeds, indent=2), encoding="utf-8")

    out_path = results_dir / out_file
    snap_path = results_dir / snapshot_file

    args = [
        sys.executable,
        str(repo_root / "generate.py"),
        "--mode", "refine",
        "--seeds_file", str(seeds_path),
        "--children_per_parent", str(int(children_per_parent)),
        "--out_file", str(out_path.name),
        "--snapshot_file", str(snap_path.name),
    ]
    if no_retain:
        args.append("--no_retain")

    env = os.environ.copy()
    env["UNLEARN_RESULTS_DIR"] = str(results_dir)

    subprocess.run(args, check=True, env=env)

    code_str = out_path.read_text(encoding="utf-8")
    code_str = _sanitize_code_for_exec(code_str)  # robust safety
    namespace = {"torch": torch}
    exec(code_str, namespace)

    _attach_epochs_attr(namespace)

    loss_fns: Dict[int, Callable] = {}
    for name, obj in namespace.items():
        if callable(obj) and name.startswith("loss_fn_"):
            suffix = name.split("loss_fn_")[1]
            if suffix.isdigit():
                loss_fns[int(suffix)] = _wrap_to_scalar_loss(obj)  # ensure scalar & grad
    loss_fns = dict(sorted(loss_fns.items(), key=lambda kv: kv[0]))

    # NEW: read mapping saved by generate.py (in results dir)
    mapping_path = results_dir / "refine_mapping.json"
    mapping_raw: Dict[str, Dict[str, int]] = {}
    if mapping_path.exists():
        try:
            mapping_raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception as e:
            print("[utils] WARNING: failed to read refine_mapping.json:", e)
            mapping_raw = {}

    # Normalize keys to int
    mapping: Dict[int, Dict[str, int]] = {}
    for k, v in mapping_raw.items():
        try:
            mapping[int(k)] = {"parent_index": int(v.get("parent_index", 0)), "child_index": int(v.get("child_index", 0))}
        except Exception:
            continue

    return loss_fns, mapping



def get_loss_fn_definition_from_file(loss_file: Path, loss_id: int) -> str:
    """
    Return ONLY the definition of def loss_fn_{loss_id}(...): with its body.
    Uses the same robust block extractor to avoid regex over-capture across functions.
    """
    text = loss_file.read_text(encoding="utf-8")
    blocks = _extract_loss_fn_blocks_robust(text)
    target = f"loss_fn_{loss_id}"
    for name, body, indent in blocks:
        if name == target:
            # Rebuild just this single function with the canonical signature
            return _rebuild_with_standard_signature([(name, body, indent)])
    # Fallback: minimal, explicit scan until next top-level def
    start_tok = f"def {target}"
    start = text.find(start_tok)
    if start == -1:
        return f"# ERROR: {target} not found in {loss_file}"
    # Find the next top-level def (column 0) after start
    m = re.search(r"(?m)^\s*def\s+[A-Za-z_]\w*\s*\(", text[start + 1:])
    end = (start + 1 + m.start()) if m else len(text)
    snippet = text[start:end].rstrip()
    # Ensure we don't accidentally include imports or extra text; keep as-is.
    return snippet


# ================================
# 2) Dataset + Collator
# ================================
class JsonlQADataset(Dataset):
    """
    .jsonl with keys like {"prompt": "...", "response": "..."} (with fallbacks).
    Each item returns input_ids, attention_mask, and a stable 'idx'.
    """
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, torch.Tensor]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)

                prompt = (
                    obj.get("prompt")
                    or obj.get("instruction")
                    or obj.get("question")
                    or ""
                )
                response = (
                    obj.get("response")
                    or obj.get("output")
                    or obj.get("answer")
                    or ""
                )

                text = (prompt.strip() + "\n" + response.strip()).strip()

                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                self.samples.append(
                    {
                        "input_ids": enc["input_ids"][0],
                        "attention_mask": enc["attention_mask"][0],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "idx": torch.tensor(idx, dtype=torch.long),
        }


class CausalCollator:
    """
    Pads batch; returns input_ids, attention_mask, labels (pads -> -100), and idx.
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_indices = []

        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            idx_val = item["idx"]

            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat(
                    [ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)]
                )
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])

            labels = ids.clone()
            labels[mask == 0] = -100  # ignore padded positions

            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)
            batch_labels.append(labels)
            batch_indices.append(idx_val)

        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
            "idx": torch.stack(batch_indices, dim=0),
        }


# ================================
# 3) Log-prob utilities
# ================================
@torch.no_grad()
def _per_sample_log_probs_nograd(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    return per_sample_log_probs(model, input_ids, attention_mask, pad_token_id)


def per_sample_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Average per-sequence log p(y|x) via teacher-forced next-token scoring.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)

    logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:]

    target_mask = target_mask * (target_ids != pad_token_id)

    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    token_counts = target_mask.sum(dim=-1).clamp(min=1)
    seq_log_probs = (gathered * target_mask).sum(dim=-1) / token_counts
    return seq_log_probs


# ================================
# 4) Helpers
# ================================
def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


# ================================
# 5) Precompute reference log-probs
# ================================
def precompute_ref_log_probs_forget(
    model_name: str,
    forget_dataset: 'JsonlQADataset',
    pad_token_id: int,
    device: torch.device,
    dtype_for_model: torch.dtype,
    collator: 'CausalCollator',
    batch_size: int = 1,
) -> torch.Tensor:
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_for_model,
    )
    ref_model.to(device)

    # If multiple GPUs are available, use DataParallel to speed up precomputation
    if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(
            f"[utils] Using DataParallel for reference model on "
            f"{torch.cuda.device_count()} GPUs"
        )
        ref_model = torch.nn.DataParallel(ref_model)

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    precompute_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )

    num_forget = len(forget_dataset)
    ref_log_probs_cache = torch.empty(
        num_forget,
        dtype=dtype_for_model,
        device=device,
    )

    with torch.no_grad():
        for batch in precompute_loader:
            batch_dev = to_device(batch, device)
            seq_lp = _per_sample_log_probs_nograd(
                ref_model,
                batch_dev["input_ids"],
                batch_dev["attention_mask"],
                pad_token_id=pad_token_id,
            )
            idxs = batch_dev["idx"].long()
            ref_log_probs_cache[idxs] = seq_lp.detach()

    del ref_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ref_log_probs_cache


# ================================
# 6) Merge LoRA -> full weights
# ================================
def prepare_eval_snapshot(
    base_model_name: str,
    lora_dir: str,
    save_dir: str,
    dtype: torch.dtype,
    device: torch.device,
) -> str:
    """
    Merge LoRA into base weights and write a CLEAN, ATOMIC eval snapshot.

    Critical changes:
      - Write to a temp directory then rename (atomic) to avoid partial snapshots.
      - Remove any existing save_dir first (prevents mixing shards across runs).
      - Save tokenizer from *base_model_name* (not lora_dir) to avoid weird tokenizer drift.
    """
    import shutil
    import time

    save_dir = str(save_dir)
    tmp_dir = save_dir + f".tmp_{int(time.time())}"

    # Always start clean (prevents stale shard/index contamination)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
    )

    peft_model = PeftModel.from_pretrained(base_model, lora_dir)
    peft_model.to(device)

    merged_model = peft_model.merge_and_unload()
    merged_model.to("cpu")

    # Write into tmp_dir first
    merged_model.save_pretrained(tmp_dir)

    # IMPORTANT: tokenizer should come from the *base model*
    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if getattr(tok, "pad_token", None) is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token
    tok.save_pretrained(tmp_dir)

    # Atomic promote tmp_dir -> save_dir
    os.rename(tmp_dir, save_dir)

    del merged_model, peft_model, base_model, tok
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return save_dir



# ================================
# 7) Open-Unlearning eval launcher
# ================================
def run_openunlearning_eval(
    ckpt_dir: str,
    device_idx: int = 0,
    task_name: Optional[str] = None,
    retain_logs_path: Optional[str] = "",
    base_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Launch OpenUnlearning's eval.py on a merged checkpoint, and robustly resolve
    where Hydra wrote TOFU_EVAL.json / TOFU_SUMMARY.json.

    Key fixes vs. earlier version:
      1) Infer correct Hydra `model=` from ckpt_dir/config.json (avoids Llama2 vs Llama3 mismatch).
      2) Do NOT fall back to unrelated "newest" summary when eval fails or task_name doesn't match.
      3) Only accept outputs created after this eval starts.

    Returns a dict with:
      - retcode
      - eval_dir
      - tofu_eval_json
      - tofu_summary_json
      - task_name
      - retain_logs_path_used
      - model_key
      - model_inference (vocab_size/hidden_size/model_type)
    """
    import os
    import sys
    import re
    import json
    import time
    import subprocess
    from glob import glob
    from typing import Any, Dict, Optional, Tuple, List

    ou_root = os.path.join(REPO_DIR, "ou")
    eval_py = os.path.join(ou_root, "src", "eval.py")
    if not os.path.exists(eval_py):
        raise FileNotFoundError(f"[run_openunlearning_eval] eval.py not found at {eval_py}")

    if task_name is None:
        task_name = f"EVAL_{int(time.time())}"

    # --- read ckpt config to infer model family ---
    model_inference: Dict[str, Any] = {}
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            vocab_size = cfg.get("vocab_size", None)
            hidden_size = cfg.get("hidden_size", cfg.get("n_embd", None))
            model_type = cfg.get("model_type", None)
            architectures = cfg.get("architectures", None)
            model_inference = {
                "config_path": cfg_path,
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "model_type": model_type,
                "architectures": architectures,
            }
        except Exception as e:
            print(f"[run_openunlearning_eval] WARNING: failed to parse {cfg_path}: {e}")

    def _list_hydra_model_keys() -> List[str]:
        # Common Hydra group locations
        candidate_dirs = [
            os.path.join(ou_root, "conf", "model"),
            os.path.join(ou_root, "configs", "model"),
            os.path.join(ou_root, "config", "model"),
            os.path.join(ou_root, "src", "conf", "model"),
        ]
        keys: List[str] = []
        for d in candidate_dirs:
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.endswith(".yaml") or fn.endswith(".yml"):
                        keys.append(os.path.splitext(fn)[0])
        # de-dup while preserving order
        seen = set()
        out = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _pick_model_key(
        ckpt_dir: str,
        base_model_name: Optional[str],
        model_inference: Dict[str, Any],
    ) -> str:
        """
        Choose a Hydra `model=` key that matches the checkpoint.
        Preference order:
          1) If base_model_name string matches an existing key (fuzzy), use it.
          2) If config.json indicates Llama-3 vocab (128256), choose a Llama-3.* key (1B/3B/8B by hidden_size).
          3) Else default to Llama-2-7b-chat-hf.
        """
        keys = _list_hydra_model_keys()

        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

        # 1) try base_model_name fuzzy match against keys
        if base_model_name and keys:
            nb = norm(base_model_name)
            # exact-ish
            for k in keys:
                if norm(k) == nb:
                    return k
            # substring-ish
            for k in keys:
                nk = norm(k)
                if nb in nk or nk in nb:
                    return k

        vocab_size = model_inference.get("vocab_size")
        hidden_size = model_inference.get("hidden_size")

        # 2) heuristic by vocab/hidden dims
        want_patterns: List[str] = []
        if vocab_size == 128256:
            # Llama-3 family
            if hidden_size == 2048:
                want_patterns = ["llama-3.2-1b", "llama3.2-1b", "3.2-1b", "llama-3-1b", "llama-3.2-1B"]
            elif hidden_size == 3072:
                want_patterns = ["llama-3.2-3b", "llama3.2-3b", "3.2-3b", "llama-3-3b", "llama-3.2-3B"]
            elif hidden_size == 4096:
                want_patterns = ["llama-3.1-8b", "llama3.1-8b", "3.1-8b", "llama-3-8b", "llama-3.1-8B"]
            else:
                want_patterns = ["llama-3", "llama3"]
        elif vocab_size == 32000:
            want_patterns = ["llama-2-7b-chat", "llama2-7b-chat", "llama-2-7b", "llama2-7b", "Llama-2-7b-chat-hf"]

        if keys and want_patterns:
            pats = [norm(p) for p in want_patterns]
            # prefer keys that contain "instruct" if checkpoint is likely instruct
            for k in keys:
                nk = norm(k)
                if any(p in nk for p in pats) and ("instruct" in nk):
                    return k
            for k in keys:
                nk = norm(k)
                if any(p in nk for p in pats):
                    return k

        # 3) last resort default
        return "Llama-2-7b-chat-hf"

    model_key = _pick_model_key(ckpt_dir, base_model_name, model_inference)

    exp_key = "eval/tofu/default"
    saves_eval_root = os.path.join(ou_root, "saves", "eval")
    expected_eval_dir = os.path.join(saves_eval_root, task_name)
    expected_tofu_eval_json = os.path.join(expected_eval_dir, "TOFU_EVAL.json")
    expected_tofu_summary_json = os.path.join(expected_eval_dir, "TOFU_SUMMARY.json")

    args = [
        sys.executable,
        eval_py,
        "--config-name=eval.yaml",
        f"experiment={exp_key}",
        f"model={model_key}",
        f"model.model_args.pretrained_model_name_or_path={ckpt_dir}",
        f"model.model_args.attn_implementation=sdpa",
        f"task_name={task_name}",
    ]

    retain_logs_override = (retain_logs_path or "").strip()
    if retain_logs_override:
        args.append(f"retain_logs_path={retain_logs_override}")
    else:
        print(
            "[run_openunlearning_eval] WARNING: retain_logs_path not provided. "
            "Some metrics (e.g., forget_quality/privleak/model_utility) may be missing or default."
        )

    env = os.environ.copy()
    if not env.get("CUDA_VISIBLE_DEVICES", "").strip():
        env["CUDA_VISIBLE_DEVICES"] = str(int(device_idx) if device_idx is not None else 0)

    print("[run_openunlearning_eval] Using model key:", model_key)
    if model_inference:
        print("[run_openunlearning_eval] Checkpoint config inference:", model_inference)
    print("[run_openunlearning_eval] Launching Open-Unlearning eval with args:")
    for a in args:
        print("   ", a)

    start_ts = time.time()
    proc = subprocess.Popen(args, cwd=ou_root, env=env)
    ret = proc.wait()
    print("[run_openunlearning_eval] Return code:", ret)

    # -----------------------
    # Output discovery (safe)
    # -----------------------
    resolved_eval_dir = expected_eval_dir
    tofu_eval_json = expected_tofu_eval_json
    tofu_summary_json = expected_tofu_summary_json

    def _norm(p: str) -> str:
        return os.path.normpath(p).replace("\\", "/")

    def _is_recent(p: str, since: float) -> bool:
        try:
            return os.path.getmtime(p) >= (since - 2.0)  # small clock/FS skew tolerance
        except Exception:
            return False

    # Search roots where Hydra might write
    search_roots = [
        os.path.join(ou_root, "saves", "eval"),
        os.path.join(ou_root, "outputs"),
        os.path.join(ou_root, "multirun"),
    ]

    # Only consider summaries from *this* run window
    candidates: List[str] = []
    for r in search_roots:
        if os.path.isdir(r):
            candidates.extend(glob(os.path.join(r, "**", "TOFU_SUMMARY.json"), recursive=True))

    # Filter by "recent" + task_name in path (strong requirement)
    recent = [p for p in candidates if _is_recent(p, start_ts)]
    matching = [p for p in recent if f"/{task_name}/" in _norm(p) or _norm(p).endswith(f"/{task_name}/TOFU_SUMMARY.json")]

    if matching:
        # pick newest among matching
        best = max(matching, key=lambda p: os.path.getmtime(p))
        tofu_summary_json = best
        resolved_eval_dir = os.path.dirname(best)
        tofu_eval_json = os.path.join(resolved_eval_dir, "TOFU_EVAL.json")
    else:
        # If eval succeeded but output dir naming differs, allow a softer match:
        # "task_name appears somewhere in path"
        soft = [p for p in recent if task_name in _norm(p)]
        if soft:
            best = max(soft, key=lambda p: os.path.getmtime(p))
            tofu_summary_json = best
            resolved_eval_dir = os.path.dirname(best)
            tofu_eval_json = os.path.join(resolved_eval_dir, "TOFU_EVAL.json")
        else:
            # No valid outputs attributable to this run. Do NOT fall back to unrelated summaries.
            print(
                "[run_openunlearning_eval] WARNING: Could not locate TOFU_SUMMARY.json produced by this run "
                f"(task_name={task_name})."
            )

    if os.path.exists(tofu_summary_json):
        print(f"[run_openunlearning_eval] TOFU_SUMMARY.json: {tofu_summary_json}")
    else:
        print(f"[run_openunlearning_eval] WARNING: TOFU_SUMMARY.json not found at {tofu_summary_json}")

    if os.path.exists(tofu_eval_json):
        print(f"[run_openunlearning_eval] TOFU_EVAL.json: {tofu_eval_json}")
    else:
        print(f"[run_openunlearning_eval] WARNING: TOFU_EVAL.json not found at {tofu_eval_json}")

    return {
        "retcode": ret,
        "experiment": exp_key,
        "eval_dir": resolved_eval_dir,
        "tofu_eval_json": tofu_eval_json,
        "tofu_summary_json": tofu_summary_json,
        "task_name": task_name,
        "retain_logs_path_used": retain_logs_override,
        "model_key": model_key,
        "model_inference": model_inference,
    }




# ================================
# 8) Iteration summaries
# ================================
def write_current_summary(
    loss_id: int,
    loss_fn_definition: str,
    loss_history: List[float],
    tofu_summary_json_path: str,
    filename: str = "iteration0.txt",
    tofu_eval_json_path: Optional[str] = None,
    precomputed_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    results_dir = get_results_dir()
    out_path = results_dir / filename

    # 1) Build metrics dict
    if isinstance(precomputed_metrics, dict):
        metrics = dict(precomputed_metrics)
    else:
        # Fallback: load from TOFU_SUMMARY / TOFU_EVAL directly
        try:
            if tofu_summary_json_path and os.path.isfile(tofu_summary_json_path):
                with open(tofu_summary_json_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                metrics = {"error": "TOFU_SUMMARY.json not found"}
        except Exception as e:
            metrics = {"error": str(e)}

        # Best-effort: also pick up extraction_strength / privleak from TOFU_EVAL.json
        try:
            if tofu_eval_json_path and os.path.isfile(tofu_eval_json_path):
                with open(tofu_eval_json_path, "r", encoding="utf-8") as f:
                    eval_obj = json.load(f)

                def _maybe_extract_metric(obj: Dict[str, Any], names: Tuple[str, ...]) -> Optional[float]:
                    for name in names:
                        cand = obj.get(name)
                        if isinstance(cand, dict):
                            for key in ("agg_value", "value", "mean"):
                                v = cand.get(key)
                                if isinstance(v, (int, float)):
                                    return float(v)
                        elif isinstance(cand, (int, float)):
                            return float(cand)

                        metrics_block = obj.get("metrics")
                        if isinstance(metrics_block, dict):
                            sub = metrics_block.get(name)
                            if isinstance(sub, dict):
                                for key in ("agg_value", "value", "mean"):
                                    v = sub.get(key)
                                    if isinstance(v, (int, float)):
                                        return float(v)
                            elif isinstance(sub, (int, float)):
                                return float(sub)
                    return None

                es_val = _maybe_extract_metric(eval_obj, ("extraction_strength", "EXTRACTION_STRENGTH", "Extraction_Strength"))
                if es_val is not None:
                    metrics["extraction_strength"] = es_val

                pl_val = _maybe_extract_metric(eval_obj, ("privleak", "PRIVLEAK", "priv_leak", "PRIV_LEAK"))
                if pl_val is not None:
                    metrics["privleak"] = pl_val
        except Exception:
            pass

    # 2) In all cases, drop forget_truth_ratio so we never use/expose it.
    if isinstance(metrics, dict):
        for key in ("forget_truth_ratio", "FORGET_TRUTH_RATIO"):
            metrics.pop(key, None)
        sub = metrics.get("metrics")
        if isinstance(sub, dict):
            for key in ("forget_truth_ratio", "FORGET_TRUTH_RATIO"):
                sub.pop(key, None)

    # 3) Merge into the iteration file
    existing: Dict[str, Any] = {}
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, dict):
                    existing = {}
        except Exception:
            existing = {}

    existing[str(loss_id)] = {
        "loss_fn_definition": loss_fn_definition,
        "loss_history": [float(x) for x in loss_history],
        "open_unlearning_summary": metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"[utils] Wrote/merged summary for loss_id={loss_id} to {out_path}")
    return str(out_path)


def append_top_k_metadata(top_k_entries: List[Dict[str, Any]], filename: str = "iteration0.txt") -> None:
    results_dir = get_results_dir()
    out_path = results_dir / filename
    data: Dict[str, Any] = {}
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

    data["__top_k__"] = top_k_entries

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[utils] Appended __top_k__ metadata to {out_path}")
