# generate_muse.py
import os
import json
import re
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# -----------------------
# Results directory handling
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
    # fallback: create a new timestamped folder
    base = repo_root / "results"
    base.mkdir(parents=True, exist_ok=True)
    return _make_timestamp_dir(base)


# -----------------------
# Prompt builders (MUSE / SimNPO-style, raw logits)
# -----------------------
def build_prompts(n_losses: int) -> Tuple[str, str]:
    """
    Returns (SYSTEM_PROMPT, USER_PROMPT) for initial generation.

    IMPORTANT FOR MUSE / SimNPO:
    - The tensors named `log_probs_forget` / `log_probs_retain` are NOT true log-probabilities.
      In the MUSE pipeline, they are *sequence-level scores computed directly from raw logits*:
        * we gather the raw logits of the ground-truth tokens
        * average them over the (non-padded) positions
      So they are arbitrary real-valued scores in "logit units", not normalized log p(y|x).
    - Likewise, ref_log_probs_* are the same kind of scores from a frozen reference model.
    - You SHOULD design SimNPO-style / NPO-like objectives that operate on these scores
      (and their differences vs the reference), instead of relying on probability log-ratios or CE losses.
    """
    system_prompt = (
        "You are an expert ML researcher specializing in machine unlearning for large language models.\n"
        "We are working in a MUSE-News-style setup, using SimNPO-like objectives built on raw logits.\n"
        "Your job is to DESIGN NOVEL LOSS FUNCTIONS for fine-tuning a model to forget specific data\n"
        "without catastrophically hurting performance on the retain set.\n\n"
        "DATA / INPUTS (CRITICAL):\n"
        "- You will receive four PyTorch tensors:\n"
        "    log_probs_forget, log_probs_retain,\n"
        "    ref_log_probs_forget, ref_log_probs_retain.\n"
        "- Despite their names, these are actually *sequence-level scores derived directly from raw logits*,\n"
        "  not true log-probabilities or CE losses.\n"
        "  Concretely, for each sequence, we:\n"
        "    * take the raw logits of the correct next tokens from the model,\n"
        "    * average them over all non-padded positions.\n"
        "  The references (ref_*) come from a frozen reference MUSE model.\n"
        "- This matches the spirit of SimNPO for MUSE, where they used raw logits instead of\n"
        "  probability / loss-based log-ratios.\n\n"
        "CRITICAL OUTPUT FORMAT RULES:\n"
        "1) First output a <think> block with your private reasoning.\n"
        "2) Then output an <answer> block containing ONLY Python code: EXACTLY "
        f"{n_losses} loss function definitions. No prose, no Markdown fences.\n\n"
        "Each function MUST:\n"
        f"- Be named EXACTLY: loss_fn_1, loss_fn_2, ..., loss_fn_{n_losses}.\n"
        "- Use THIS EXACT SIGNATURE (do NOT change names or order; no *args/**kwargs):\n"
        "    def loss_fn_K(log_probs_forget, log_probs_retain,\n"
        "                   ref_log_probs_forget=None, ref_log_probs_retain=None):\n"
        "  (The names are historical; treat them as sequence-level logit scores.)\n"
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
        "- Higher FORGET scores (log_probs_forget) should INCREASE the loss (push the model to lower them).\n"
        "- Higher RETAIN scores (log_probs_retain) should DECREASE the loss (push the model to raise them).\n"
        "- In other words, holding other terms fixed: the loss should be monotonically INCREASING in log_probs_forget\n"
        "  and monotonically DECREASING in log_probs_retain.\n\n"
        "REFERENCE USAGE (SimNPO-INSPIRATION, STRONGLY ENCOURAGED):\n"
        "- SimNPO-style objectives typically compare *scores* from the current model vs a reference model.\n"
        "- Good building blocks include:\n"
        "    * (log_probs_forget - ref_log_probs_forget)      # positive means current model is too confident\n"
        "    * (log_probs_retain - ref_log_probs_retain)      # positive means current model is better than ref\n"
        "- Penalize positive (log_probs_forget - ref_log_probs_forget) to push forget scores *below* reference.\n"
        "- Reward positive (log_probs_retain - ref_log_probs_retain) to keep retain scores *above* reference.\n"
        "- Use margins, hinges (via torch.clamp / torch.relu), softplus, squared terms, etc., on these differences.\n\n"
        "DIVERSITY GUIDANCE (INITIAL GENERATION ONLY):\n"
        "- Vary mechanisms where possible (e.g., margin/difference; hinge/clamp/softplus; squared/absolute;\n"
        "  exponential/logistic; ratio/normalization; reference-based deltas; multiplicative/geometric forms;\n"
        "  entropy/variance-style; focal/curriculum).\n\n"
        "NUMERICAL STABILITY:\n"
        "- These scores are in logit units and can be large in magnitude.\n"
        "- Use stable transforms where needed (e.g., softplus instead of log(1+exp(x)), small eps for divisions).\n"
        "- Avoid operations that can easily produce NaNs/Inf.\n"
        "- NEVER use Python max/min or torch.max/min with scalar constants (e.g., 0 or 0.0) for hinge-like behavior;\n"
        "  use torch.clamp(x, min=0.0) or torch.relu(x) instead.\n"
    )
    user_prompt = (
        "We are doing machine unlearning on a MUSE-News-style dataset.\n\n"
        "We have forget sets (must be forgotten) and retain sets (must be preserved).\n"
        "The inputs log_probs_forget/log_probs_retain and ref_* are sequence-level scores\n"
        "computed directly from raw logits (SimNPO-style), NOT true log-probabilities.\n\n"
        f"Propose {n_losses} candidate unlearning losses meeting the rules.\n"
        "VERY IMPORTANT:\n"
        f"- In <answer>, output ONLY the function definitions named loss_fn_1..loss_fn_{n_losses},\n"
        "  back-to-back. No Markdown fences. No prose. No prints.\n"
        "- After </answer>, output NOTHING else.\n"
    )
    return system_prompt, user_prompt


def build_refine_prompts(parent: Dict[str, Any], n_children: int) -> Tuple[str, str]:
    """
    Returns (SYSTEM_PROMPT, USER_PROMPT) for refinement, in the same MUSE / SimNPO logit-based setting.
    """
    parent_def = parent.get("loss_fn_definition", "")
    loss_history = parent.get("loss_history", [])
    metrics = parent.get("open_unlearning_summary", {})

    metrics_json = json.dumps(metrics, ensure_ascii=False)

    system_prompt = (
        "You are an expert ML researcher specializing in machine unlearning for large language models.\n"
        "You will IMPROVE a given parent loss by proposing refined variants.\n"
        "We are in the MUSE News setting with SimNPO-style, raw-logit-based sequence scores.\n\n"
        "INPUT SEMANTICS (CRITICAL):\n"
        "- log_probs_forget / log_probs_retain are sequence-level scores computed from raw logits\n"
        "  (average logits of the correct tokens). They are NOT normalized probabilities.\n"
        "- ref_log_probs_* are the same scores from a frozen reference model.\n"
        "- You should design objectives in the spirit of SimNPO: operate directly on these scores\n"
        "  and their differences vs the reference.\n\n"
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
        "  \"\"\"epochs: K\"\"\"  where K is a positive integer in [1, 10].\n"
        "- Prefer putting the entire function header on ONE line (helps tooling), but multiline is allowed.\n\n"
        "OBJECTIVE DIRECTION (CRITICAL):\n"
        "- You are minimizing the loss during training.\n"
        "- Higher FORGET scores should INCREASE the loss (push the model to lower them).\n"
        "- Higher RETAIN scores should DECREASE the loss (push the model to raise them).\n\n"
        "REFERENCE USAGE (SimNPO-INSPIRATION, STRONGLY ENCOURAGED):\n"
        "- Penalize positive (log_probs_forget - ref_log_probs_forget) to pull forget scores below reference.\n"
        "- Reward positive (log_probs_retain - ref_log_probs_retain) to keep retain scores above reference.\n"
        "- Use margins / hinges (torch.clamp or torch.relu), softplus, squared terms, etc. on these differences.\n\n"
        "MUSE METRIC-GUIDED REFINEMENT (IMPORTANT):\n"
        "- The metrics JSON includes keys like:\n"
        "    * muse_verbmem_forget, muse_knowmem_forget, muse_privleak_forget, muse_knowmem_retain\n"
        "    * and their *_good variants (e.g., muse_verbmem_forget_good) where HIGHER = BETTER.\n"
        "    * muse_assessment_score, an overall scalar.\n"
        "- If forget-related raw metrics are HIGH (bad), or *_good metrics are LOW, you should\n"
        "  strengthen forgetting: increase penalties on forget scores, especially when they exceed reference.\n"
        "- If muse_knowmem_retain or muse_knowmem_retain_good is LOW, you should reduce forgetting strength\n"
        "  and/or boost rewards on retain scores to preserve utility.\n"
        "- Always keep the monotonicity constraints: loss increasing in log_probs_forget, decreasing in log_probs_retain.\n\n"
        "NUMERICAL STABILITY:\n"
        "- Treat the inputs as real-valued logit scores. Avoid operations that blow up on large magnitudes.\n"
        "- Prefer softplus / sigmoid / tanh over raw exp when exponentiating.\n"
        "- NEVER use Python max/min or torch.max/min with scalar constants; use torch.clamp / torch.relu instead.\n"
    )

    user_prompt = (
        "Here is the PARENT loss (Python):\n\n"
        f"{parent_def}\n\n"
        "Parent training loss history (final loss per epoch):\n"
        f"{loss_history}\n\n"
        "Parent final evaluation metrics (MUSE / SimNPO-style scores):\n"
        f"{metrics_json}\n\n"
        f"Now produce {n_children} refined candidate loss functions that address weaknesses implied by the history/metrics.\n"
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


def extract_answer_code(full_model_text: str) -> str:
    start_tag = "<answer>"
    end_tag = "</answer>"
    start_idx = full_model_text.find(start_tag)
    end_idx = full_model_text.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        code_block = full_model_text[start_idx + len(start_tag):end_idx].strip()
        return code_block
    return full_model_text.strip()


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
        code,
    )
    return code


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
            # Ensure syntactically valid function; keep runtime minimal.
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

    If the model fails to produce any valid `loss_fn_*` blocks, we synthesize
    a set of safe fallback loss functions instead of raising an error. This
    avoids crashing multi-iteration training when generation misbehaves.
    """
    code = _sanitize_lambdas_and_empty_params(raw_code)
    code = _keep_only_loss_def_blocks(code)

    if not code.strip():
        # Fallback path: no valid functions detected
        print("[generate_muse] WARNING: refine mode: model produced no valid loss_fn_* "
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
    code = _rewrite_scalar_max_min_to_torch(code)   # ensure hinge-like ops are tensor-safe
    code = _ensure_import_torch(code)
    return code


def save_results(code_block: str,
                 out_path: Path,
                 snapshot_path: Path) -> None:
    out_path.write_text(code_block)
    snapshot_path.write_text(code_block)
    print(f"Saved generated loss functions to {out_path.resolve()}")
    print(f"Saved snapshot to {snapshot_path.resolve()}")


# -----------------------
# Helpers: dedup + fallback
# -----------------------
def _canon_body(body: str) -> str:
    # Normalize whitespace & numbers formatting to catch trivial duplicates
    s = re.sub(r'\s+', ' ', body).strip()
    return s


def _synthesize_fallback_blocks(count: int, start_index: int) -> List[Tuple[str, str, int]]:
    """
    Create 'count' valid loss_fn_* blocks (name is temporary; we renumber later).
    All satisfy: increasing in log_probs_forget, decreasing in log_probs_retain.
    Designed to be SimNPO-ish for MUSE (score differences vs reference).
    """
    blocks: List[Tuple[str, str, int]] = []
    # Cycle a few safe templates
    for i in range(count):
        idx = start_index + i
        ep = 1 + (idx % 10)
        t = idx % 4
        if t == 0:
            # Simple weighted gap between forget and retain scores
            body = (
                f'    """epochs: {ep}"""\n'
                f'    alpha = {0.55 + 0.05 * (idx % 5):.2f}\n'
                f'    return (alpha * log_probs_forget - log_probs_retain).mean()'
            )
        elif t == 1:
            # SimNPO-style: penalize forget score above reference, reward retain above reference
            body = (
                f'    """epochs: {ep}"""\n'
                f'    lam_f = {0.7 + 0.1 * (idx % 3):.2f}\n'
                f'    lam_r = {0.9 + 0.1 * (idx % 3):.2f}\n'
                f'    if ref_log_probs_forget is None or ref_log_probs_retain is None:\n'
                f'        base = lam_f * log_probs_forget - lam_r * log_probs_retain\n'
                f'        return base.mean()\n'
                f'    diff_f = log_probs_forget - ref_log_probs_forget\n'
                f'    diff_r = log_probs_retain - ref_log_probs_retain\n'
                f'    forget_term = torch.nn.functional.softplus(lam_f * diff_f)\n'
                f'    retain_term = torch.nn.functional.softplus(-lam_r * diff_r)\n'
                f'    return (forget_term + retain_term).mean()'
            )
        elif t == 2:
            # Quadratic penalty on forget scores, linear reward on retain
            body = (
                f'    """epochs: {ep}"""\n'
                f'    beta = {0.45 + 0.05 * (idx % 5):.2f}\n'
                f'    return (beta * (log_probs_forget ** 2) - log_probs_retain).mean()'
            )
        else:
            # Exponential gap in score space (clamped for stability)
            body = (
                f'    """epochs: {ep}"""\n'
                f'    gamma = {0.6 + 0.1 * (idx % 4):.1f}\n'
                f'    gap = log_probs_forget - gamma * log_probs_retain\n'
                f'    return torch.exp(torch.clamp(gap, max=10.0)).mean()'
            )
        # Temporary name; final names will be normalized later
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
    parser.add_argument("--N", type=int, default=20, help="number of losses for initial mode")
    parser.add_argument("--seeds_file", type=str, default="", help="JSON with seeds for refine mode")
    parser.add_argument("--children_per_parent", type=int, default=5)
    parser.add_argument("--out_file", type=str, default="")
    parser.add_argument("--snapshot_file", type=str, default="")

    args = parser.parse_args()

    # Resolve run dir for artifacts
    RESULTS_DIR = get_results_dir()

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

        # Try multiple attempts; deduplicate by body; then synthesize any shortfall
        max_attempts = 6
        collected_blocks: List[Tuple[str, str, int]] = []
        seen_bodies = set()
        attempt = 0

        thinking_chunks: List[str] = []

        while len(collected_blocks) < n and attempt < max_attempts:
            attempt += 1
            think_block, answer_code = generate_two_phase(llm=llm, tokenizer=tokenizer, messages=messages)

            # Log thinking + token count for visibility
            think_tokens = len(tokenizer(think_block, add_special_tokens=False).input_ids)
            print(f"[generate_muse] Attempt {attempt}: think tokens = {think_tokens}")
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

            print(
                f"[generate_muse] Attempt {attempt}: extracted {len(attempt_blocks)} defs, "
                f"added {added_now} new (unique). total={len(collected_blocks)}/{n}"
            )

        # If still short, synthesize the remainder deterministically
        if len(collected_blocks) < n:
            remaining = n - len(collected_blocks)
            print(
                f"[generate_muse] WARNING: collected only {len(collected_blocks)}/{n}. "
                f"Synthesizing {remaining} fallback functions."
            )
            synth = _synthesize_fallback_blocks(remaining, start_index=len(collected_blocks) + 1)
            collected_blocks.extend(synth)

        # Rebuild, normalize names, truncate to n, rewrite unsafe max/min, ensure import
        collected_blocks = collected_blocks[:n]
        combined = _rebuild_with_standard_signature(collected_blocks)
        combined = _normalize_numeric_names_in_order(combined)
        combined = _truncate_to_n_functions(combined, n)
        combined = _rewrite_scalar_max_min_to_torch(combined)
        combined = _ensure_import_torch(combined)

        out_path = RESULTS_DIR / ("generated_unlearning_losses.txt" if not args.out_file else args.out_file)
        snapshot_path = RESULTS_DIR / ("initial.txt" if not args.snapshot_file else args.snapshot_file)
        save_results(combined, out_path, snapshot_path)

        # Write initial thinking (with token counts)
        (RESULTS_DIR / "thinking_initial.txt").write_text("\n\n".join(thinking_chunks), encoding="utf-8")
        print(f"Saved initial thinking to {(RESULTS_DIR / 'thinking_initial.txt').resolve()}")

    else:  # refine
        seeds_path = Path(args.seeds_file)
        if not seeds_path.exists():
            raise FileNotFoundError(f"Seeds file not found: {seeds_path}")
        seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
        k = int(args.children_per_parent)

        # We will preserve per-parent thinking and raw initial children,
        # and also build a combined set (with global renumbering) + mapping.
        all_blocks_with_parent: List[Tuple[int, int, str, int]] = []  # (parent_idx, child_idx, body, indent)
        total = 0

        for parent_idx, parent in enumerate(seeds):
            sys_prompt, usr_prompt = build_refine_prompts(parent, k)
            messages = build_messages(sys_prompt, usr_prompt)

            think_block, answer_code = generate_two_phase(llm=llm, tokenizer=tokenizer, messages=messages)

            # Log tokens for refinement thinking as well
            think_tokens = len(tokenizer(think_block, add_special_tokens=False).input_ids)
            print(f"[generate_muse refine] Parent {parent_idx}: think tokens = {think_tokens}")
            (RESULTS_DIR / f"thinking1_{parent_idx}.txt").write_text(
                f"[tokens={think_tokens}]\n{think_block}", encoding="utf-8"
            )
            print(
                f"Saved refinement thinking for parent {parent_idx} -> "
                f"{(RESULTS_DIR / f'thinking1_{parent_idx}.txt').resolve()}"
            )

            # Keep whatever loss_fn_* blocks appear; finalize to exactly k
            block_clean = _finalize_code_block(answer_code, n_expected=k)

            # Save per-parent initial refined functions
            (RESULTS_DIR / f"initial1_{parent_idx}.txt").write_text(block_clean, encoding="utf-8")
            print(
                f"Saved refined functions for parent {parent_idx} -> "
                f"{(RESULTS_DIR / f'initial1_{parent_idx}.txt').resolve()}"
            )

            parent_blocks = _extract_loss_fn_blocks_robust(block_clean)

            # Record with parent index and local child index
            for j, (_name, body, indent) in enumerate(parent_blocks, start=1):
                all_blocks_with_parent.append((parent_idx, j, body, indent))
                total += 1

        # Renumber globally to loss_fn_1..total and ensure import, while building a mapping.
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
                rebuilt_blocks.append(
                    header + "\n" + "    return (log_probs_forget.mean() - log_probs_retain.mean())"
                )
            mapping[global_idx] = {"parent_index": pidx, "child_index": cidx}

        combined = "\n".join(rebuilt_blocks).strip()
        combined = _ensure_import_torch(combined)

        out_path = RESULTS_DIR / ("generated_unlearning_losses_iter1.txt" if not args.out_file else args.out_file)
        snapshot_path = RESULTS_DIR / ("initial_iter1.txt" if not args.snapshot_file else args.snapshot_file)
        save_results(combined, out_path, snapshot_path)

        # write mapping so train_muse_news.py/utils.py can write per-parent summaries
        (RESULTS_DIR / "refine_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        print(f"Saved refine mapping to {(RESULTS_DIR / 'refine_mapping.json').resolve()}")

    # Cleanup
    del llm, tokenizer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
