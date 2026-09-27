from jinja2 import Environment

from onepass.traces.mlx_chat import CHAT_TEMPLATE


def test_chat_template_renders_string_and_parts():
    t = Environment().from_string(CHAT_TEMPLATE)
    out = t.render(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": [{"type": "text", "text": "U"}]},
                             {"role": "assistant", "content": "A"}, {"role": "user", "content": "U2"}])
    assert out == "[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][INST]U[/INST]A</s>[INST]U2[/INST]"
