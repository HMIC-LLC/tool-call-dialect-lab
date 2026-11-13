# Tool Call Dialect Lab

An offline experiment harness for distinguishing a model's text from a harness's ability to parse that text. It accepts saved, human-labeled replies and reports exact parser agreement at three levels: `tagged`, `standard`, and `lenient`. The parser also recognizes native tool calls. This is a standalone public extraction of an HMIC parser. The bundled cases are **synthetic**, not benchmark evidence.

## Try it

Python 3.10+; standard library only. In PowerShell, from this folder:

```powershell
& python .\dialect_lab.py .\examples\synthetic-cases.jsonl
& python -m unittest discover -s .\tests -v
```

Each JSONL case requires a unique `id`, `response_text`, and manually labeled `expected_calls` array. `offered_tools` is optional; include it to test whether the parser accepts unoffered tool names. `native_tool_calls` is optional. The report includes the SHA-256 of the input file, exact matches by parser level, parsed calls, refusal reasons and flags.

The harness does not run models, execute tools, score whether a planned action was safe, or establish a quantization effect. Labels are supplied by the user and should be reviewed independently. The fixture includes a quoted example that is deliberately **not** a tool request, showing how a lenient parser can invent a call. A larger experiment should keep raw responses, model revisions, quantization details, runtime settings, seeds and execution receipts separately. Compare runs only after controlling those variables, and keep parser misses separate from model decisions.

`tool_dialect.py` parses Hermes, wrapped `tool_calls`, bare JSON, Harmony, Qwen XML, native OpenAI tool calls, and two parse-only dialects. `standard` accepts only whole-message call objects beyond tagged/native forms; `lenient` also detects embedded objects and is intentionally more exposed to false positives. The bundled tests verify a strictness difference and refusal of an unoffered untagged tool.

MIT license. Copyright (c) 2026 HMIC LLC.
