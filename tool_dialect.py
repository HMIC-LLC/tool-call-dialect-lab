#!/usr/bin/env python3
"""Pure tool-call dialect parser and renderer for offline experiments.

Stdlib only. It makes no model or network calls and does no file I/O.

    from tool_dialect import parse, parse_detail, render
    calls = parse(text, tools=offered, native=message.get("tool_calls"))      # -> list[ToolCall]
    text  = render(calls, "hermes")                                           # -> str
    tcs   = render(calls, "openai")                                           # -> list[dict] (message.tool_calls)

Dialects that parse AND render (round-trip):
  hermes      <tool_call>{"name": N, "arguments": {...}}</tool_call>
  tool_calls  <tool_calls>[{"name": N, "arguments": {...}}, ...]</tool_calls>
  bare_json   {"name": N, "arguments": {...}}  one object per line, or a JSON list
  harmony     <|start|>assistant<|channel|>commentary to=functions.N <|constrain|>json<|message|>{...}<|call|>
  qwen_xml    <tool_call><function=N><parameter=K>V</parameter></function></tool_call>  (Strata, Qwen3-Coder, Nemotron 3)
  openai      native message.tool_calls (render returns the list, not text)
Parse-only recognisers (render is not offered because the upstream grammar is not pinned here):
  function_calls  <function_calls>[...]</function_calls> and <call>{...}</call>
  gemma           <|tool_call>call:N{k: v}<tool_call|>

Levels (how much text is trusted to be a call):
  "tagged"   only complete tagged forms + native. If a reply has
             <tool_call>{json}</tool_call> calls, other syntaxes in it are ignored
             (flag other_dialects_ignored).
  "standard" (default) tagged + replies whose WHOLE text (after closed <think> blocks, one enclosing ``` fence,
             and stray boundary <tool_call>/</tool_call> tags) is nothing but call objects; harmony without its
             stop token; whole-text <tool_calls> without the closing tag.
  "lenient"  standard + call objects embedded in prose or fences. Objects after the first invented
             <tool_result>/<function_result> in the same reply are dropped (flag fabricated_result).
Untagged objects must have keys exactly {name, arguments} (bare_json also accepts "parameters"), arguments an
object, and -- when `tools` is given -- a name that is offered (exact, case-sensitive).
Tagged calls keep any name so a downstream evaluator can report an unknown tool.
Malformed JSON is never repaired: inside tags it yields a call with
arguments=None; untagged it is refused.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

__all__ = ["ToolCall", "ParseResult", "parse", "parse_detail", "render", "detect", "DIALECTS", "RENDERABLE", "LEVELS"]
VERSION = "tool_dialect v1 (2026-11-05)"

RENDERABLE = ("hermes", "tool_calls", "bare_json", "harmony", "qwen_xml", "openai")
DIALECTS = RENDERABLE + ("function_calls", "gemma")
LEVELS = ("tagged", "standard", "lenient")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict | None          # None = the body was not a JSON object (runner: BAD_ARGS)
    dialect: str
    id: str | None = None
    raw_arguments: str | None = None
    flags: tuple = ()

    def key(self):
        return (self.name, json.dumps(self.arguments, sort_keys=True) if self.arguments is not None else None)


@dataclass
class ParseResult:
    calls: list
    flags: list = field(default_factory=list)     # notes: native_used, fabricated_result, half_tag, fenced, ...
    refused: list = field(default_factory=list)   # (reason, snippet) for call-looking text that was not accepted

    @property
    def dialects(self):
        return sorted({c.dialect for c in self.calls})


# ---------------------------------------------------------------- helpers
THINK_CLOSED = re.compile(r"<think>.*?</think>\s*", re.S | re.I)           # the frozen runner's THINK
HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)        # the frozen runner's TOOL_CALL
QWEN_XML = re.compile(r"<tool_call>\s*(<function=[^>\"]+>.*?</function>)\s*</tool_call>", re.S)
WRAPPED = re.compile(r"<(tool_calls|function_calls|call)>\s*(.*?)\s*</\1>", re.S)
GEMMA = re.compile(r"<\|tool_call>\s*call:([A-Za-z_][\w.\-]*)\s*(\{.*?\})\s*<tool_call\|>", re.S)
HARMONY_HEAD = re.compile(r"to=functions\.([A-Za-z_][\w\-]*)")
HARMONY_MSG = "<|message|>"
HARMONY_END = re.compile(r"<\|(?:call|end|return)\|>")
FENCE_WHOLE = re.compile(r"^```[a-zA-Z_]*[ \t]*\n(.*?)\n?```$", re.S)
FAKE_RESULT = re.compile(r"<(?:tool_result|function_result|tool_response)\b", re.I)
STRAY_TAG = re.compile(r"</?tool_call>", re.I)
_DEC = json.JSONDecoder()


def tool_index(tools) -> dict | None:
    """OpenAI [{type, function:{name, parameters}}], plain [{name, parameters}] or {name: schema} -> {name: schema}."""
    if tools is None:
        return None
    if isinstance(tools, dict):
        return {str(k): (v or {}) for k, v in tools.items()}
    idx = {}
    for t in tools:
        if isinstance(t, str):
            idx[t] = {}
            continue
        if not isinstance(t, dict):
            continue
        f = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(f.get("name"), str) and f["name"]:
            idx[f["name"]] = f.get("parameters") or {}
    return idx


def _args_obj(a):
    """dict -> dict; JSON string holding an object -> dict; anything else -> None."""
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            v = json.loads(a or "{}")
        except (json.JSONDecodeError, ValueError):
            return None
        return v if isinstance(v, dict) else None
    return None


def _call_from_obj(obj, dialect, *, alias_ok=False):
    """{"name", "arguments"} -> ToolCall, or (None, reason). Keys must be exactly name + arguments (or parameters)."""
    if not isinstance(obj, dict):
        return None, "not_an_object"
    keys = set(obj)
    akey = "arguments" if "arguments" in keys else ("parameters" if alias_ok and "parameters" in keys else None)
    if akey is None or keys != {"name", akey}:
        return None, "keys_not_name_arguments"
    if not isinstance(obj["name"], str) or not obj["name"]:
        return None, "name_not_string"
    args = _args_obj(obj[akey])
    if args is None:
        return None, "arguments_not_object"
    return ToolCall(obj["name"], args, dialect, flags=(("alias_parameters",) if akey == "parameters" else ())), None


def _strip_think(text):
    t = THINK_CLOSED.sub("", text or "")
    # an unclosed <think> means everything after it is still reasoning
    m = re.search(r"<think>", t, re.I)
    return t[:m.start()] if m else t


def _relaxed_obj(s):
    """Gemma-style {key: "v", n: 3} -> dict. Quotes bare keys only; never touches string contents."""
    s = s.replace('<|"|>', '"')
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    out, i, n, in_str, expect_key = [], 0, len(s), False, False
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(s[i + 1]); i += 2; continue
            if ch == '"':
                in_str = False
            i += 1; continue
        if ch == '"':
            in_str = True; out.append(ch); i += 1; expect_key = False; continue
        if ch in "{,":
            expect_key = True; out.append(ch); i += 1; continue
        if expect_key and (ch.isalpha() or ch == "_"):
            j = i
            while j < n and (s[j].isalnum() or s[j] in "_-"):
                j += 1
            k = j
            while k < n and s[k] in " \t":
                k += 1
            if k < n and s[k] == ":":
                out.append('"' + s[i:j] + '"'); i = j; expect_key = False; continue
        if not ch.isspace():
            expect_key = False
        out.append(ch); i += 1
    try:
        v = json.loads("".join(out))
    except (json.JSONDecodeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def _xml_value(raw, declared):
    v = raw[1:] if raw.startswith("\n") else raw
    v = v[:-1] if v.endswith("\n") else v
    if declared == "string":
        return v
    if declared is None and not re.match(r"^\s*[\[{\d\-tfn]", v):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def _parse_qwen_xml(body, idx):
    """<function=N><parameter=K>V</parameter>...</function> -> ToolCall or (None, reason). Mirrors Strata's
    parse_tool_call: values are JSON-decoded unless the schema declares them strings."""
    m = re.match(r"\s*<function=([^>\"\s]+)>(.*)</function>\s*$", body, re.S)
    if not m:
        return None, "malformed_function_tag"
    name, rest = m.group(1), m.group(2)
    props = (((idx or {}).get(name) or {}).get("properties") or {})
    args = {}
    for pm in re.finditer(r"<parameter=([^>\s]+)>(.*?)</parameter>(?=\s*(?:<parameter=|$))", rest, re.S):
        declared = (props.get(pm.group(1)) or {}).get("type") if props else None
        args[pm.group(1)] = _xml_value(pm.group(2), declared)
    leftover = re.sub(r"<parameter=([^>\s]+)>(.*?)</parameter>(?=\s*(?:<parameter=|$))", "", rest, flags=re.S).strip()
    if leftover:
        return None, "unparsed_text_inside_function"
    return ToolCall(name, args, "qwen_xml"), None


def _iter_json_objects(text):
    """Yield (start, end, obj) for every top-level JSON value that starts with '{' or '['."""
    i = 0
    while i < len(text):
        j = min([p for p in (text.find("{", i), text.find("[", i)) if p >= 0], default=-1)
        if j < 0:
            return
        try:
            obj, end = _DEC.raw_decode(text, j)
        except (json.JSONDecodeError, ValueError):
            i = j + 1
            continue
        yield j, end, obj
        i = end


def _objs_to_calls(obj, dialect, alias_ok):
    """An object or a list of objects -> list of (call|None, reason)."""
    items = obj if isinstance(obj, list) else [obj]
    if isinstance(obj, list) and not obj:
        return [(None, "empty_list")]
    return [_call_from_obj(o, dialect, alias_ok=alias_ok) for o in items]


# ---------------------------------------------------------------- tagged pass
def _tagged(text, idx, res):
    """Complete tagged forms. Returns list of (offset, ToolCall)."""
    found = []
    for m in HERMES.finditer(text):          # exactly the frozen runner's matches
        try:
            obj = json.loads(m.group(1))
            name, a = obj.get("name", ""), obj.get("arguments", {})
            args = _args_obj(a)
            raw = a if isinstance(a, str) else None
            found.append((m.start(), ToolCall(name if isinstance(name, str) else "", args, "hermes",
                                              raw_arguments=raw if args is None else None,
                                              flags=() if args is not None else ("bad_args",))))
        except (json.JSONDecodeError, ValueError, AttributeError):
            found.append((m.start(), ToolCall("", None, "hermes", raw_arguments=m.group(1), flags=("bad_json",))))
    taken = [(m.start(), m.end()) for m in HERMES.finditer(text)]
    if found:
        # The frozen runner's own protocol is present: return exactly what it parses (superset property),
        # and only note that other call syntax was also in the reply.
        if (QWEN_XML.search(text) or WRAPPED.search(text) or GEMMA.search(text) or HARMONY_HEAD.search(text)):
            res.flags.append("other_dialects_ignored")
        return found

    def free(a, b):
        return all(b <= s or a >= e for s, e in taken)

    for m in QWEN_XML.finditer(text):
        if not free(m.start(), m.end()):
            continue
        c, why = _parse_qwen_xml(m.group(1), idx)
        if c:
            found.append((m.start(), c)); taken.append((m.start(), m.end()))
        else:
            res.refused.append((why, m.group(0)[:120]))
    for m in WRAPPED.finditer(text):
        if not free(m.start(), m.end()):
            continue
        dialect = "tool_calls" if m.group(1) == "tool_calls" else "function_calls"
        body = m.group(2).strip()
        objs = [o for s, e, o in _iter_json_objects(body)]
        if not objs:
            res.refused.append(("wrapper_body_not_json", m.group(0)[:120])); continue
        calls = []
        for o in objs:
            calls += _objs_to_calls(o, dialect, alias_ok=False)
        if all(c for c, _ in calls):
            for k, (c, _) in enumerate(calls):
                found.append((m.start() + k * 1e-3, c))
            taken.append((m.start(), m.end()))
        else:
            res.refused.append(("wrapper_item_not_a_call", m.group(0)[:120]))
    for m in GEMMA.finditer(text):
        if not free(m.start(), m.end()):
            continue
        args = _relaxed_obj(m.group(2))
        if args is None:
            res.refused.append(("gemma_args_unreadable", m.group(0)[:120])); continue
        found.append((m.start(), ToolCall(m.group(1), args, "gemma"))); taken.append((m.start(), m.end()))
    for c_off, c in _harmony(text, res, require_end=True):
        found.append((c_off, c))
    return found


def _harmony(text, res, require_end):
    out = []
    for m in HARMONY_HEAD.finditer(text):
        k = text.find(HARMONY_MSG, m.end())
        nxt = HARMONY_HEAD.search(text, m.end())
        if k < 0 or (nxt and nxt.start() < k):
            res.refused.append(("harmony_no_message", text[m.start():m.start() + 120])); continue
        body_start = k + len(HARMONY_MSG)
        e = HARMONY_END.search(text, body_start)
        if e is None and require_end:
            continue                                   # standard pass accepts a missing stop token
        body = text[body_start:e.start() if e else len(text)].strip()
        args = _args_obj(body)
        if args is None:
            res.refused.append(("harmony_args_not_object", body[:120])); continue
        out.append((m.start(), ToolCall(m.group(1), args, "harmony", flags=(() if e else ("no_stop_token",)))))
    return out


# ---------------------------------------------------------------- public
def parse_detail(text, *, tools=None, native=None, level="standard") -> ParseResult:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    idx = tool_index(tools)
    res = ParseResult(calls=[])
    if native:
        for i, tc in enumerate(native):
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            args = _args_obj(a) if a not in (None, "") else {}
            res.calls.append(ToolCall(fn.get("name", ""), args, "openai", id=tc.get("id"),
                                      raw_arguments=a if args is None else None,
                                      flags=() if args is not None else ("bad_args",)))
        res.flags.append("native_used")
        return res
    body = _strip_think(text or "")
    tagged = _tagged(body, idx, res)
    if tagged:
        res.calls = [c for _, c in sorted(tagged, key=lambda p: p[0])]
        if FAKE_RESULT.search(body):
            res.flags.append("fabricated_result_present")
        return res
    if level == "tagged":
        return res

    # ---- standard: the whole reply is calls
    whole = body.strip()
    fm = FENCE_WHOLE.match(whole)
    if fm:
        whole = fm.group(1).strip(); res.flags.append("fenced")
    h = _harmony(whole, res, require_end=False)
    if h:
        res.calls = [c for _, c in h]
        return res
    wo = re.match(r"^<tool_calls>\s*(.*?)\s*(?:</tool_calls>)?$", whole, re.S)
    if wo:
        whole = wo.group(1); res.flags.append("tool_calls_unclosed")
    dialect = "tool_calls" if wo else "bare_json"
    stray = STRAY_TAG.findall(whole)
    cleaned = STRAY_TAG.sub("\n", whole).strip()
    calls, ok = [], bool(cleaned)
    pos = 0
    for s, e, obj in _iter_json_objects(cleaned):
        if cleaned[pos:s].strip():
            ok = False; break
        for c, why in _objs_to_calls(obj, dialect, alias_ok=True):
            if c is None:
                ok = False; res.refused.append((why, cleaned[s:s + 120])); break
            calls.append(c)
        if not ok:
            break
        pos = e
    if ok and calls and not cleaned[pos:].strip():
        bad = [c.name for c in calls if idx is not None and c.name not in idx]
        if bad:
            res.refused.append(("unknown_tool:" + ",".join(bad), cleaned[:120]))
        else:
            if stray:
                res.flags.append("half_tag")
            if idx is None:
                res.flags.append("names_unchecked")
            res.calls = calls
            return res
    if level == "standard":
        return res

    # ---- lenient: call objects anywhere before the first invented tool result
    cut = FAKE_RESULT.search(body)
    scan = body[:cut.start()] if cut else body
    calls, last_end = [], 0
    for s, e, obj in _iter_json_objects(scan):
        last_end = max(last_end, e)
        for c, why in _objs_to_calls(obj, "bare_json", alias_ok=True):
            if c is None:
                continue
            if idx is not None and c.name not in idx:
                res.refused.append(("unknown_tool:" + c.name, scan[s:s + 80])); continue
            calls.append(c)
    if calls and re.search(r"\{\s*\"name\"\s*:", scan[last_end:]):
        # a call-shaped object after the last complete one never closed: the reply was cut mid-plan
        res.refused.append(("truncated_tail", scan[last_end:last_end + 120]))
        return res
    if calls:
        res.calls = calls
        res.flags.append("embedded")
        if cut:
            res.flags.append("fabricated_result")
        if idx is None:
            res.flags.append("names_unchecked")
    return res


def parse(text, *, tools=None, native=None, level="standard") -> list:
    """Return the tool calls in a model reply (see module doc). `native` = message.tool_calls, used when non-empty."""
    return parse_detail(text, tools=tools, native=native, level=level).calls


def detect(text, *, tools=None, native=None) -> list:
    """Which dialects a reply uses (empty list = no call found at the lenient level)."""
    return parse_detail(text, tools=tools, native=native, level="lenient").dialects


def _xml_render_value(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def render(calls, dialect):
    """Render ToolCalls (or {"name","arguments"} dicts) in one dialect. 'openai' returns message.tool_calls."""
    norm = []
    for c in calls:
        if isinstance(c, ToolCall):
            norm.append((c.name, c.arguments if c.arguments is not None else {}, c.id))
        else:
            norm.append((c["name"], _args_obj(c.get("arguments", {})) or {}, c.get("id")))
    if dialect == "hermes":
        return "\n".join("<tool_call>" + json.dumps({"name": n, "arguments": a}, ensure_ascii=False) + "</tool_call>"
                         for n, a, _ in norm)
    if dialect == "tool_calls":
        return "<tool_calls>" + json.dumps([{"name": n, "arguments": a} for n, a, _ in norm], ensure_ascii=False) + "</tool_calls>"
    if dialect == "bare_json":
        return "\n".join(json.dumps({"name": n, "arguments": a}, ensure_ascii=False) for n, a, _ in norm)
    if dialect == "harmony":
        return "".join(f"<|start|>assistant<|channel|>commentary to=functions.{n} <|constrain|>json<|message|>"
                       + json.dumps(a, ensure_ascii=False) + "<|call|>" for n, a, _ in norm)
    if dialect == "qwen_xml":
        parts = []
        for n, a, _ in norm:
            ps = "".join(f"<parameter={k}>\n{_xml_render_value(v)}\n</parameter>\n" for k, v in a.items())
            parts.append(f"<tool_call>\n<function={n}>\n{ps}</function>\n</tool_call>")
        return "\n".join(parts)
    if dialect == "openai":
        return [{"id": i or f"call_{k}", "type": "function",
                 "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
                for k, (n, a, i) in enumerate(norm)]
    raise ValueError(f"render: dialect must be one of {RENDERABLE}")
