"""Loading the local System 2 model on Apple silicon.

The pinned 8-bit conversion ships a tokenizer without a chat template and with the known Mistral regex issue, so both
are fixed at load time: the template below is the model family's instruction format (system prompt block, then
instruction blocks; the tokenizer adds the BOS token itself), and `fix_mistral_regex` restores correct tokenization.
Message content may arrive as a string or as a list of text parts (agent frameworks send the latter).
"""
from __future__ import annotations

CHAT_TEMPLATE = (
    "{%- for m in messages %}"
    "{%- set c = m['content'] if m['content'] is string else (m['content'] | selectattr('type', 'equalto', 'text') | map(attribute='text') | join('\\n')) %}"
    "{%- if m['role'] == 'system' %}[SYSTEM_PROMPT]{{ c }}[/SYSTEM_PROMPT]"
    "{%- elif m['role'] == 'user' %}[INST]{{ c }}[/INST]"
    "{%- elif m['role'] == 'assistant' %}{{ c }}</s>"
    "{%- elif m['role'] == 'tool-response' or m['role'] == 'tool' %}[INST]{{ c }}[/INST]"
    "{%- endif %}"
    "{%- endfor %}"
)

TOKENIZER_CONFIG = {"fix_mistral_regex": True, "chat_template": CHAT_TEMPLATE}


def load_kwargs(revision: str | None = None) -> dict:
    kw = {"tokenizer_config": dict(TOKENIZER_CONFIG)}
    if revision:
        kw["revision"] = revision
    return kw


def load(repo: str, revision: str | None = None):
    from mlx_lm import load as _load

    model, tok = _load(repo, **load_kwargs(revision))
    if not getattr(tok, "chat_template", None):
        tok.chat_template = CHAT_TEMPLATE
    return model, tok
