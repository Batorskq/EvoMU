# generate_books.py
#
# Loss-function generator for MUSE Books (Option A: per-token mean log-probs).
#
# IMPORTANT ASSUMPTION (matches utils_books.py / per_sample_log_probs):
#   log_probs_forget and log_probs_retain are per-sample scalars of shape [B]
#   representing the MEAN per-token log-probability over non-pad predicted tokens.
#   They are NOT sums, and are already length-normalized.
#
# Therefore:
#   - Do NOT add any extra "divide by length" behavior (length is not provided anyway).
#   - Use coefficients/margins appropriate for per-token mean log-probs (typical range ~[-15, 0]).

import os
import json
import re
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# -----------------------
# Results directory handling (compatible with train/utils)
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
# Prompt builders (TOFU-style, updated for per-token means)
# -----------------------
def build_prompts(n_losses: int) -> Tuple[str, str]:
    """
    Returns (SYSTEM_PROMPT, USER_PROMPT) for initial generation.
    """
    system_prompt = (
        "You are an expert ML researcher specializing in machine unlearning for large language models.\n"
        "Your job is to DESIGN NOVEL LOSS FUNCTIONS for fine-tuning a model to forget specific data\n"
        "without catastrophically hurting performance on the retain set.\n\n"
        "CRITICAL SCALAR DEFINITION (READ CAREFULLY):\n"
        "- You will receive two tensors: log_probs_forget and log_probs_retain.\n"
        "- Each is a 1D tensor of shape [B].\n"
        "- Each element is the MEAN PER-TOKEN log-probability over non-pad predicted tokens for that sample.\n"
        "- These are NOT sums. They are already length-normalized.\n"
        "- Typical values are negative (e.g., ~[-15, 0]).\n\n"
        "CRITICAL OUTPUT FORMAT RULES:\n"
        "1) First output a <think> block with your private reasoning.\n"
        "2) Then output an <answer> block containing ONLY Python code: EXACTLY "
        f"{n_losses} loss function definitions. No prose, no Markdown fences.\n\n"
        "Each function MUST:\n"
        f"- Be named EXACTLY: loss_fn_1, loss_fn_2, ..., loss_fn_{n_losses}.\n"
        "- Use THIS EXACT SIGNATURE (do NOT change names or order; no *args/**kwargs):\n"
        "    def loss_fn_K(log_probs_forget, log_probs_retain,\n"
        "                   ref_log_probs_forget=None, ref_log_probs_retain=None):\n"
        "- Return a single scalar (use .mean() or equivalent reduction).\n"
        "- Include at least one fixed numeric tradeoff constant INSIDE the function body (e.g., alpha = 0.7).\n"
        "- Assume PyTorch tensors; use ONLY the four inputs in the signature; use only torch ops.\n"
        "- DO NOT use `lambda` (use normal `def`).\n"
        "- IMPORTANT: Inside <answer>, EVERY line must belong to a Python function definition\n"
        "  starting with `def loss_fn_` or be blank. ANY other text is forbidden.\n"
        "- At the very top of each function body, include an inline one-line docstring specifying epochs EXACTLY like:\n"
        "  \"\"\"epochs: K\"\"\"  where K is a positive integer in [1, 10].\n"
        "- Prefer putting the entire function header on ONE line (helps tooling), but multiline is allowed.\n\n"
        "OBJECTIVE DIRECTION (CRITICAL):\n"
        "- You are minimizing the loss during training.\n"
        "- Higher FORGET mean log-probabilities should INCREASE the loss (push the model to lower them).\n"
        "- Higher RETAIN mean log-probabilities should DECREASE the loss (push the model to raise them).\n"
        "- In other words, holding other terms fixed: the loss should be monotonically INCREASING in log_probs_forget\n"
        "  and monotonically DECREASING in log_probs_retain.\n\n"
        "REFERENCE USAGE (OPTIONAL, ENCOURAGED):\n"
        "- ref_log_probs_forget / ref_log_probs_retain are in the SAME SCALE (per-token means).\n"
        "- If you use references, penalize positive (log_probs_forget - ref_log_probs_forget) and/or penalize\n"
        "  negative (log_probs_retain - ref_log_probs_retain). Reference deltas should influence the loss non-trivially.\n\n"
        "SCALE GUIDANCE (VERY IMPORTANT FOR PER-TOKEN MEANS):\n"
        "- Use coefficients and margins that make sense for mean log-probs.\n"
        "- Example margins: 0.05 to 1.0 (not 50).\n"
        "- Example coefficients: 0.1 to 10 (not 1e6).\n"
        "- Avoid exploding exponentials; clamp inputs if you exponentiate.\n\n"
        "DIVERSITY GUIDANCE:\n"
        "- Vary mechanisms where possible (e.g., margin/difference; hinge/clamp/softplus; squared/absolute; exponential/logistic;\n"
        "  ratio/normalization; reference-based deltas; multiplicative/geometric forms; robust/Huber; focal/curriculum).\n\n"
        "NUMERICAL STABILITY:\n"
        "- Use stable transforms where needed (e.g., softplus instead of log(1+exp(x)), small eps for divisions).\n"
        "- Avoid operations that can easily produce NaNs/Inf for typical mean log-prob ranges.\n"
        "- NEVER use Python max/min or torch.max/min with scalar constants (e.g., 0 or 0.0) for hinge-like behavior;\n"
        "  use torch.clamp(x, min=0.0) or torch.relu(x) instead.\n"
    )

    user_prompt = (
        "We are doing machine unlearning on a dataset with FORGET and RETAIN splits.\n\n"
        "We optimize a loss over batches where:\n"
        "- log_probs_forget is a [B] tensor of per-sample MEAN per-token log-prob on FORGET samples.\n"
        "- log_probs_retain is a [B] tensor of per-sample MEAN per-token log-prob on RETAIN samples.\n\n"
        f"Propose {n_losses} candidate unlearning losses meeting the rules.\n"
        "VERY IMPORTANT:\n"
        f"- In <answer>, output ONLY the function definitions named loss_fn_1..loss_fn_{n_losses},\n"
        "  back-to-back. No Markdown fences. No prose. No prints.\n"
        "- After </answer>, output NOTHING else.\n"
    )
    return system_prompt, user_prompt


def build_refine_prompts(parent: Dict[str, Any], n_children: int) -> Tuple[str, str]:
    """
    Returns (SYSTEM_PROMPT, USER_PROMPT) for refinement (TOFU-style metrics).
    This prompt is updated to reflect per-token mean log-probs.
    """
    parent_def = parent.get("loss_fn_definition", "")
    loss_history = parent.get("loss_history", [])
    metrics = parent.get("open_unlearning_summary", {})

    metrics_json = json.dumps(metrics, ensure_ascii=False)

    system_prompt = (
        "You are an expert ML researcher specializing in machine unlearning for large language models.\n"
        "You will IMPROVE a given parent loss by proposing refined variants.\n\n"
        "CRITICAL SCALAR DEFINITION (READ CAREFULLY):\n"
        "- log_probs_forget and log_probs_retain are 1D tensors of shape [B].\n"
        "- Each element is the MEAN PER-TOKEN log-probability (already length-normalized, not a sum).\n"
        "- Typical values are negative (e.g., ~[-15, 0]).\n\n"
        "CRITICAL OUTPUT FORMAT RULES:\n"
        "1) First output a <think> block with your private reasoning.\n"
        "2) Then output an <answer> block containing ONLY Python code: EXACTLY "
        f"{n_children} loss function definitions. No prose, no Markdown fences.\n\n"
        "Each function MUST:\n"
        f"- Be named EXACTLY: loss_fn_1, loss_fn_2, ..., loss_fn_{n_children}.\n"
        "- Use THIS EXACT SIGNATURE (do NOT change names or order; no *args/**kwargs):\n"
        "    def loss_fn_K(log_probs_forget, log_probs_retain,\n"
        "                   ref_log_probs_forget=None, ref_log_probs_retain=None):\n"
        "- Return a single scalar (use .mean() or equivalent reduction).\n"
        "- Include at least one fixed numeric tradeoff constant INSIDE the function body (e.g., alpha = 0.7).\n"
        "- Assume PyTorch tensors; use ONLY the four inputs in the signature; use only torch ops.\n"
        "- DO NOT use `lambda` (use normal `def`).\n"
        "- IMPORTANT: Inside <answer>, EVERY line must belong to a Python function definition\n"
        "  starting with `def loss_fn_` or be blank. ANY other text is forbidden.\n"
        "- At the very top of each function body, include an inline one-line docstring specifying epochs EXACTLY like:\n"
        "  \"\"\"epochs: K\"\"\"  where K is a positive integer in [1, 10].\n\n"
        "OBJECTIVE DIRECTION (CRITICAL):\n"
        "- You are minimizing the loss during training.\n"
        "- Higher FORGET mean log-probabilities should INCREASE the loss.\n"
        "- Higher RETAIN mean log-probabilities should DECREASE the loss.\n\n"
        "REFERENCE USAGE (OPTIONAL, ENCOURAGED):\n"
        "- References are in the SAME SCALE (per-token means).\n"
        "- Penalize positive (log_probs_forget - ref_log_probs_forget) and/or penalize\n"
        "  negative (log_probs_retain - ref_log_probs_retain).\n\n"
        "METRIC-GUIDED REFINEMENT (GENERIC):\n"
        "- If forgetting metrics are poor: strengthen penalties tied to log_probs_forget\n"
        "  (higher coefficients, margins with softplus/hinge, or ref-delta penalties).\n"
        "- If utility/retention metrics are poor: strengthen rewards tied to log_probs_retain\n"
        "  and/or soften forgetting pressure.\n\n"
        "SCALE GUIDANCE (PER-TOKEN MEANS):\n"
        "- Use margins like 0.05..1.0 and coefficients like 0.1..10.\n"
        "- Clamp inputs to exp/logistic-like functions.\n\n"
        "NUMERICAL STABILITY:\n"
        "- Use stable transforms (softplus, eps), avoid NaNs/Inf.\n"
        "- Avoid torch.max/min with scalars; use torch.clamp or torch.relu.\n"
    )

    user_prompt = (
        "Here is the PARENT loss (Python):\n\n"
        f"{parent_def}\n\n"
        "Parent training loss history (final loss per epoch):\n"
        f"{loss_history}\n\n"
        "Parent final evaluation metrics:\n"
        f"{metrics_json}\n\n"
        f"Now produce {n_children} refined candidate loss functions that address weaknesses implied by the history/metrics.\n"
        "IMPORTANT:\n"
        f"- In <answer>, output ONLY the function definitions named loss_fn_1..loss_fn_{n_children}, back-to-back.\n"
        "- No Markdown fences. No prose. No comments outside functions.\n"
        "- After </answer>, output NOTHING else.\n"
    )
    return system_prompt, user_prompt


# -----------------------
# Prompt builders (MUSE - per-token mean aware)
# -----------------------
_MUSE_METRIC_KEYS = (
    "muse_verbmem_forget",
    "muse_knowmem_forget",
    "muse_privleak_forget",
    "muse_knowmem_retain",
)


def _looks_like_muse_metrics(metrics: Any) -> bool:
    if not isinstance(metrics, dict):
        return False
    return any(k in metrics for k in _MUSE_METRIC_KEYS)


def _extract_muse_metrics_only(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return ONLY the 4 raw MUSE metrics we want to show the refinement LLM.
    Missing keys are included with value None.
    """
    out: Dict[str, Any] = {}
    for k in _MUSE_METRIC_KEYS:
        v = metrics.get(k, None)
        out[k] = float(v) if isinstance(v, (int, float)) else None
    return out


def build_refine_prompts_muse(parent: Dict[str, Any], n_children: int) -> Tuple[str, str]:
    """
    MUSE-Books refinement prompt (updated for per-token mean log-probs).
    Shows ONLY:
      - muse_verbmem_forget
      - muse_knowmem_forget
      - muse_privleak_forget
      - muse_knowmem_retain
    and explains directionality.
    """
    parent_def = parent.get("loss_fn_definition", "")
    loss_history = parent.get("loss_history", [])
    metrics = parent.get("open_unlearning_summary", {}) or {}

    muse_metrics_only = _extract_muse_metrics_only(metrics)
    metrics_json = json.dumps(muse_metrics_only, ensure_ascii=False)

    system_prompt = (
        "You are an expert ML researcher specializing in machine unlearning for large language models.\n"
        "You will IMPROVE a given parent loss by proposing refined variants for MUSE Books.\n\n"
        "CRITICAL SCALAR DEFINITION (READ CAREFULLY):\n"
        "- log_probs_forget and log_probs_retain are 1D tensors of shape [B].\n"
        "- Each element is the MEAN PER-TOKEN log-probability (already length-normalized, not a sum).\n"
        "- Typical values are negative (e.g., ~[-15, 0]).\n\n"
        "CRITICAL OUTPUT FORMAT RULES:\n"
        "1) First output a <think> block with your private reasoning.\n"
        "2) Then output an <answer> block containing ONLY Python code: EXACTLY "
        f"{n_children} loss function definitions. No prose, no Markdown fences.\n\n"
        "Each function MUST:\n"
        f"- Be named EXACTLY: loss_fn_1, loss_fn_2, ..., loss_fn_{n_children}.\n"
        "- Use THIS EXACT SIGNATURE (do NOT change names or order; no *args/**kwargs):\n"
        "    def loss_fn_K(log_probs_forget, log_probs_retain,\n"
        "                   ref_log_probs_forget=None, ref_log_probs_retain=None):\n"
        "- Return a single scalar (use .mean() or equivalent reduction).\n"
        "- Include at least one fixed numeric tradeoff constant INSIDE the function body (e.g., alpha = 0.7).\n"
        "- Assume PyTorch tensors; use ONLY the four inputs in the signature; use only torch ops.\n"
        "- DO NOT use `lambda` (use normal `def`).\n"
        "- IMPORTANT: Inside <answer>, EVERY line must belong to a Python function definition\n"
        "  starting with `def loss_fn_` or be blank. ANY other text is forbidden.\n"
        "- At the very top of each function body, include an inline one-line docstring specifying epochs EXACTLY like:\n"
        "  \"\"\"epochs: K\"\"\"  where K is a positive integer in [1, 10].\n\n"
        "OBJECTIVE DIRECTION (CRITICAL):\n"
        "- You are minimizing the loss during training.\n"
        "- Higher FORGET mean log-probabilities should INCREASE the loss (push down forget).\n"
        "- Higher RETAIN mean log-probabilities should DECREASE the loss (push up retain).\n\n"
        "MUSE METRICS YOU WILL SEE (ONLY THESE 4):\n"
        "- muse_verbmem_forget: verbatim memorization on the FORGET set. LOWER is better.\n"
        "- muse_knowmem_forget: knowledge memorization on the FORGET set. LOWER is better.\n"
        "- muse_privleak_forget: privacy leakage relative to a retrained baseline.\n"
        "  BEST is close to 0. Large positive OR large negative magnitude is bad.\n"
        "  Sign meaning: positive => over-unlearning; negative => under-unlearning.\n"
        "  (Often reported scaled by 100; do NOT assume it lies in [0,1].)\n"
        "- muse_knowmem_retain: knowledge retention on the RETAIN set. HIGHER is better.\n\n"
        "METRIC-GUIDED REFINEMENT (MUSE):\n"
        "- If muse_verbmem_forget or muse_knowmem_forget are HIGH: strengthen forgetting pressure\n"
        "  (increase weight/margins on log_probs_forget; optionally add ref-based penalties when\n"
        "  (log_probs_forget - ref_log_probs_forget) is positive).\n"
        "- If muse_knowmem_retain is LOW: protect retention\n"
        "  (increase reward on log_probs_retain; optionally penalize drops when\n"
        "  (log_probs_retain - ref_log_probs_retain) is negative).\n"
        "- If |muse_privleak_forget| is LARGE: reduce “extremeness” and bring leakage toward 0\n"
        "  (rebalance forget vs retain pressure; prefer smoother penalties like softplus/Huber;\n"
        "  avoid overly aggressive forgetting margins/coefficients).\n\n"
        "SCALE GUIDANCE (PER-TOKEN MEANS):\n"
        "- Use margins like 0.05..1.0 and coefficients like 0.1..10.\n"
        "- If you exponentiate, clamp the exponent input (e.g., max=10).\n\n"
        "NUMERICAL STABILITY:\n"
        "- Use stable transforms where needed (softplus, eps).\n"
        "- Avoid NaNs/Inf; avoid torch.max/min with scalars; use torch.clamp or torch.relu.\n"
    )

    user_prompt = (
        "Here is the PARENT loss (Python):\n\n"
        f"{parent_def}\n\n"
        "Parent training loss history (final loss per epoch):\n"
        f"{loss_history}\n\n"
        "Parent final MUSE metrics (ONLY these 4):\n"
        f"{metrics_json}\n\n"
        "Note: for muse_privleak_forget, the goal is to move it toward 0 (not necessarily “lower”).\n\n"
        f"Now produce {n_children} refined candidate loss functions that improve these metrics.\n"
        "IMPORTANT:\n"
        f"- In <answer>, output ONLY the function definitions named loss_fn_1..loss_fn_{n_children}, back-to-back.\n"
        "- No Markdown fences. No prose. No comments outside functions.\n"
        "- After </answer>, output NOTHING else.\n"
    )

    return system_prompt, user_prompt


# -----------------------
# vLLM generation (two-phase)
# -----------------------
def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_two_phase(
    llm: LLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
) -> Tuple[str, str]:
    """
    Returns (think_block, answer_code_only).
    """
    base_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Phase 1: free-form reasoning until "<answer>"
    sampling_think = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=4096,
        stop=["<answer>"],
    )
    phase1_outputs = llm.generate([base_prompt], sampling_think)
    phase1_text = phase1_outputs[0].outputs[0].text
    think_block = phase1_text.strip()
    if "<think>" not in think_block:
        think_block = "<think>\n" + think_block
    if "</think>" not in think_block:
        think_block = think_block + "\n</think>"

    # Phase 2: code only, stop at "</answer>"
    prefix_for_answer = base_prompt + think_block + "<answer>"
    sampling_answer = SamplingParams(
        temperature=0.2,
        top_p=1.0,
        max_tokens=4096,
        stop=["</answer>"],
    )
    phase2_outputs = llm.generate([prefix_for_answer], sampling_answer)
    answer_block_raw = phase2_outputs[0].outputs[0].text.strip()
    if answer_block_raw.lstrip().startswith("<answer>"):
        answer_block_raw = answer_block_raw.split("<answer>", 1)[1].lstrip()

    return think_block, answer_block_raw


# -----------------------
# Post-processing (robust)
# -----------------------
_LAMBDA_ASSIGN_RE = re.compile(
    r'(?m)^\s*(loss_fn_(\d+))\s*=\s*lambda\s*(?:\((?P<params>[^)]*)\))?\s*:\s*(?P<body>.+)$'
)

_DEF_HEADER_START_RE = re.compile(r'(?m)^(\s*)def\s+(loss_fn_\d+)\s*\(')
_DEF_ANY_START_RE = re.compile(r'(?m)^(\s*)def\s+([A-Za-z_]\w*)\s*\(')


def _sanitize_lambdas_and_empty_params(code_block: str) -> str:
    code = code_block

    def _lambda_repl(m):
        fname = m.group(1)
        body = m.group("body").strip()
        sig = "(log_probs_forget, log_probs_retain, ref_log_probs_forget=None, ref_log_probs_retain=None)"
        return f"def {fname}{sig}:\n    return {body}"

    code = re.sub(_LAMBDA_ASSIGN_RE, _lambda_repl, code)

    empty_def_pat = re.compile(r'(?m)^\s*def\s+(loss_fn_\d+)\s*\(\s*\)\s*:\s*$')
    code = re.sub(
        empty_def_pat,
        r"def \1(log_probs_forget, log_probs_retain, ref_log_probs_forget=None, ref_log_probs_retain=None):",
        code
    )
    return code


def _extract_loss_fn_blocks_robust(code_block: str) -> List[Tuple[str, str, int]]:
    """
    Robustly extract top-level loss_fn_* definitions.
    Body collection stops on any dedent (indent <= header indent), not only on the next 'def'.
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


def _truncate_to_n_functions(code_block: str, n: int) -> str:
    blocks = _extract_loss_fn_blocks_robust(code_block)
    if not blocks:
        return code_block
    blocks = blocks[:n]
    return _rebuild_with_standard_signature(blocks)


def _ensure_import_torch(code_block: str) -> str:
    header = "import torch\n\n"
    trimmed = code_block.lstrip()
    if trimmed.startswith("import torch"):
        return code_block
    return header + code_block


def _rewrite_scalar_max_min_to_torch(code: str) -> str:
    code = re.sub(r'\bmax\s*\(\s*0+(?:\.0+)?\s*,\s*(.*?)\)', r'torch.clamp(\1, min=0.0)', code)
    code = re.sub(r'\bmax\s*\(\s*(.*?)\s*,\s*0+(?:\.0+)?\s*\)', r'torch.clamp(\1, min=0.0)', code)
    code = re.sub(r'\btorch\.max\s*\(\s*0+(?:\.0+)?\s*,\s*(.*?)\)', r'torch.clamp(\1, min=0.0)', code)
    code = re.sub(r'\btorch\.max\s*\(\s*(.*?)\s*,\s*0+(?:\.0+)?\s*\)', r'torch.clamp(\1, min=0.0)', code)

    code = re.sub(r'\bmin\s*\(\s*0+(?:\.0+)?\s*,\s*(.*?)\)', r'torch.clamp(\1, max=0.0)', code)
    code = re.sub(r'\bmin\s*\(\s*(.*?)\s*,\s*0+(?:\.0+)?\s*\)', r'torch.clamp(\1, max=0.0)', code)
    code = re.sub(r'\btorch\.min\s*\(\s*0+(?:\.0+)?\s*,\s*(.*?)\)', r'torch.clamp(\1, max=0.0)', code)
    code = re.sub(r'\btorch\.min\s*\(\s*(.*?)\s*,\s*0+(?:\.0+)?\s*\)', r'torch.clamp(\1, max=0.0)', code)
    return code


def _finalize_code_block(raw_code: str, n_expected: int) -> str:
    """
    Finalize the raw <answer> code for refine mode.
    If the model fails to produce any valid `loss_fn_*` blocks, synthesize fallbacks.
    """
    code = _sanitize_lambdas_and_empty_params(raw_code)
    code = _keep_only_loss_def_blocks(code)

    if not code.strip():
        print("[generate_books] WARNING: refine mode: model produced no valid loss_fn_* "
              "definitions; synthesizing fallback functions instead.")
        synth_blocks = _synthesize_fallback_blocks(n_expected, start_index=1)
        combined = _rebuild_with_standard_signature(synth_blocks)
        combined = _normalize_numeric_names_in_order(combined)
        combined = _truncate_to_n_functions(combined, n_expected)
        combined = _rewrite_scalar_max_min_to_torch(combined)
        combined = _ensure_import_torch(combined)
        return combined

    code = _normalize_numeric_names_in_order(code)
    code = _truncate_to_n_functions(code, n_expected)
    code = _rewrite_scalar_max_min_to_torch(code)
    code = _ensure_import_torch(code)
    return code


def save_results(code_block: str, out_path: Path, snapshot_path: Path) -> None:
    out_path.write_text(code_block, encoding="utf-8")
    snapshot_path.write_text(code_block, encoding="utf-8")
    print(f"Saved generated loss functions to {out_path.resolve()}")
    print(f"Saved snapshot to {snapshot_path.resolve()}")


# -----------------------
# Helpers: dedup + fallback
# -----------------------
def _canon_body(body: str) -> str:
    s = re.sub(r'\s+', ' ', body).strip()
    return s


def _synthesize_fallback_blocks(count: int, start_index: int) -> List[Tuple[str, str, int]]:
    """
    Create 'count' valid loss_fn_* blocks.
    All satisfy: increasing in log_probs_forget, decreasing in log_probs_retain.
    Scales are chosen for per-token mean log-probs.
    """
    blocks: List[Tuple[str, str, int]] = []
    for i in range(count):
        idx = start_index + i
        ep = 1 + (idx % 10)
        t = idx % 5

        if t == 0:
            body = (
                f'    """epochs: {ep}"""\n'
                f'    alpha = {0.8 + 0.1 * (idx % 4):.2f}\n'
                f'    return (alpha * log_probs_forget - log_probs_retain).mean()'
            )
        elif t == 1:
            body = (
                f'    """epochs: {ep}"""\n'
                f'    lam = {1.0 + 0.5 * (idx % 3):.2f}\n'
                f'    margin = {0.10 + 0.10 * (idx % 4):.2f}\n'
                f'    diff = log_probs_forget - (log_probs_retain - margin)\n'
                f'    return torch.nn.functional.softplus(lam * diff).mean()'
            )
        elif t == 2:
            body = (
                f'    """epochs: {ep}"""\n'
                f'    beta = {0.3 + 0.2 * (idx % 4):.2f}\n'
                f'    return (beta * (log_probs_forget ** 2) - log_probs_retain).mean()'
            )
        elif t == 3:
            body = (
                f'    """epochs: {ep}"""\n'
                f'    w = {0.7 + 0.1 * (idx % 4):.2f}\n'
                f'    margin = {0.20 + 0.10 * (idx % 3):.2f}\n'
                f'    # Hinge on (forget - retain + margin)\n'
                f'    return torch.clamp(w * (log_probs_forget - log_probs_retain + margin), min=0.0).mean()'
            )
        else:
            body = (
                f'    """epochs: {ep}"""\n'
                f'    gamma = {0.8 + 0.2 * (idx % 4):.2f}\n'
                f'    x = torch.clamp(log_probs_forget - gamma * log_probs_retain, max=10.0)\n'
                f'    return torch.exp(x).mean()'
            )

        name = f"loss_fn_{idx}"
        blocks.append((name, body, 0))
    return blocks


# -----------------------
# CLI
# -----------------------
def main() -> None:
    import gc
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["initial", "refine"], required=True)
    parser.add_argument("--N", type=int, default=30, help="number of losses for initial mode")
    parser.add_argument("--seeds_file", type=str, default="", help="JSON with seeds for refine mode")
    parser.add_argument("--children_per_parent", type=int, default=5)
    parser.add_argument("--out_file", type=str, default="")
    parser.add_argument("--snapshot_file", type=str, default="")

    args = parser.parse_args()

    RESULTS_DIR = get_results_dir()

    # Same model as your generate.py (keep consistent unless you decide otherwise)
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
    )

    if args.mode == "initial":
        n = int(args.N)
        system_prompt, user_prompt = build_prompts(n)
        messages = build_messages(system_prompt, user_prompt)

        max_attempts = 6
        collected_blocks: List[Tuple[str, str, int]] = []
        seen_bodies = set()
        attempt = 0

        thinking_chunks: List[str] = []

        while len(collected_blocks) < n and attempt < max_attempts:
            attempt += 1
            think_block, answer_code = generate_two_phase(llm=llm, tokenizer=tokenizer, messages=messages)

            think_tokens = len(tokenizer(think_block, add_special_tokens=False).input_ids)
            print(f"[generate_books] Attempt {attempt}: think tokens = {think_tokens}")
            thinking_chunks.append(f"### Attempt {attempt} | think_tokens={think_tokens}\n{think_block}\n")

            code1 = _sanitize_lambdas_and_empty_params(answer_code)
            attempt_blocks = _extract_loss_fn_blocks_robust(code1)

            added_now = 0
            for name, body, indent in attempt_blocks:
                canon = _canon_body(body)
                if canon in seen_bodies:
                    continue
                seen_bodies.add(canon)
                collected_blocks.append((name, body, indent))
                added_now += 1

            print(f"[generate_books] Attempt {attempt}: extracted {len(attempt_blocks)} defs, "
                  f"added {added_now} new (unique). total={len(collected_blocks)}/{n}")

        if len(collected_blocks) < n:
            remaining = n - len(collected_blocks)
            print(f"[generate_books] WARNING: collected only {len(collected_blocks)}/{n}. "
                  f"Synthesizing {remaining} fallback functions.")
            synth = _synthesize_fallback_blocks(remaining, start_index=len(collected_blocks) + 1)
            collected_blocks.extend(synth)

        collected_blocks = collected_blocks[:n]
        combined = _rebuild_with_standard_signature(collected_blocks)
        combined = _normalize_numeric_names_in_order(combined)
        combined = _truncate_to_n_functions(combined, n)
        combined = _rewrite_scalar_max_min_to_torch(combined)
        combined = _ensure_import_torch(combined)

        out_path = RESULTS_DIR / ("generated_unlearning_losses.txt" if not args.out_file else args.out_file)
        snapshot_path = RESULTS_DIR / ("initial.txt" if not args.snapshot_file else args.snapshot_file)
        save_results(combined, out_path, snapshot_path)

        (RESULTS_DIR / "thinking_initial.txt").write_text("\n\n".join(thinking_chunks), encoding="utf-8")
        print(f"Saved initial thinking to {(RESULTS_DIR / 'thinking_initial.txt').resolve()}")

    else:  # refine
        seeds_path = Path(args.seeds_file)
        if not seeds_path.exists():
            raise FileNotFoundError(f"Seeds file not found: {seeds_path}")
        seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
        k = int(args.children_per_parent)

        all_blocks_with_parent: List[Tuple[int, int, str, int]] = []  # (parent_idx, child_idx, body, indent)

        for parent_idx, parent in enumerate(seeds):
            metrics = parent.get("open_unlearning_summary", {}) or {}
            if _looks_like_muse_metrics(metrics):
                sys_prompt, usr_prompt = build_refine_prompts_muse(parent, k)
            else:
                sys_prompt, usr_prompt = build_refine_prompts(parent, k)

            messages = build_messages(sys_prompt, usr_prompt)

            think_block, answer_code = generate_two_phase(llm=llm, tokenizer=tokenizer, messages=messages)

            think_tokens = len(tokenizer(think_block, add_special_tokens=False).input_ids)
            print(f"[refine_books] Parent {parent_idx}: think tokens = {think_tokens}")
            (RESULTS_DIR / f"thinking1_{parent_idx}.txt").write_text(
                f"[tokens={think_tokens}]\n{think_block}", encoding="utf-8"
            )

            block_clean = _finalize_code_block(answer_code, n_expected=k)

            (RESULTS_DIR / f"initial1_{parent_idx}.txt").write_text(block_clean, encoding="utf-8")

            parent_blocks = _extract_loss_fn_blocks_robust(block_clean)

            for j, (_name, body, indent) in enumerate(parent_blocks, start=1):
                all_blocks_with_parent.append((parent_idx, j, body, indent))

        rebuilt_blocks: List[str] = []
        mapping: Dict[int, Dict[str, int]] = {}

        global_idx = 0
        for (pidx, cidx, body, indent) in all_blocks_with_parent:
            global_idx += 1
            indent_spaces = " " * indent
            header = (
                f"{indent_spaces}def loss_fn_{global_idx}(log_probs_forget, log_probs_retain, "
                f"ref_log_probs_forget=None, ref_log_probs_retain=None):"
            )
            if body.strip():
                rebuilt_blocks.append(header + "\n" + body)
            else:
                rebuilt_blocks.append(header + "\n" + "    return (log_probs_forget.mean() - log_probs_retain.mean())")
            mapping[global_idx] = {"parent_index": pidx, "child_index": cidx}

        combined = "\n".join(rebuilt_blocks).strip()
        combined = _ensure_import_torch(combined)

        out_path = RESULTS_DIR / ("generated_unlearning_losses_iter1.txt" if not args.out_file else args.out_file)
        snapshot_path = RESULTS_DIR / ("initial_iter1.txt" if not args.snapshot_file else args.snapshot_file)
        save_results(combined, out_path, snapshot_path)

        (RESULTS_DIR / "refine_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        print(f"Saved refine mapping to {(RESULTS_DIR / 'refine_mapping.json').resolve()}")

    # Cleanup
    del llm, tokenizer
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
