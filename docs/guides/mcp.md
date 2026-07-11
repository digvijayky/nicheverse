# MCP server (Claude Code / Codex tools)

`nicheverse` ships an [MCP](https://modelcontextprotocol.io/docs/getting-started/intro) server so agents
(Claude Code, Codex) can call the annotation and inference API as tools. The
server runs over stdio and lives in `nicheverse.mcp_server`.

## Install

```bash
pip install "nicheverse[mcp]"        # adds the `mcp` runtime
# for the LLM annotation tool also install the LLM extra + set an API key
pip install "nicheverse[llm]"
export ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY
```

## Tools

| Tool | What it does | Returns |
|---|---|---|
| `nicheverse_code_evidence` | Per-code markers / DEGs / distributions for a codebook column | JSON summary (top 15 codes) |
| `nicheverse_annotate` | LLM annotation of every code, optionally PubMed-grounded; writes a CSV | Markdown label table |
| `nicheverse_predict` | Assign cell + neighborhood codebook indices from a checkpoint | JSON code-usage summary |
| `nicheverse_pubmed` | PubMed literature search for a marker / cell-type call | JSON hits |

Every tool takes an `.h5ad` path, loads it lazily, and returns a compact
summary (paths, counts, top rows), never raw matrices.

## Register with Claude Code

```bash
claude mcp add nicheverse -- \
  /path/to/your/env/bin/python -m nicheverse.mcp_server
```

Use the interpreter that has `nicheverse` installed. After the `--`, everything
is the command Claude Code launches for the stdio server. Verify with
`claude mcp list`.

## Register with Codex

Add an entry under `[mcp_servers]` in `~/.codex/config.toml`:

```toml
[mcp_servers.nicheverse]
command = "/path/to/your/env/bin/python"
args = ["-m", "nicheverse.mcp_server"]
```

(An installed console script `nicheverse-mcp` is also available; you can use it
as `command = "nicheverse-mcp"` with no `args` once the package is on `PATH`.)

## Quick manual check

```bash
/path/to/your/env/bin/python -m nicheverse.mcp_server
```

The process waits on stdio for a client; Ctrl-C to exit. For an import check:

```bash
python -c "from nicheverse.mcp_server import main; print('ok')"
```
