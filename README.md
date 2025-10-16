# LikeC4 Mini – Lightweight C4-Style Code Visualization

LikeC4 Mini is a **stand-alone, zero-dependency Python tool** for generating **C4-inspired architecture and flow diagrams** directly from source code.  
Designed for restricted environments where installing large stacks isn’t possible.

---

## ✨ Features

- 🧠 Auto-discovery of **C1/C2/C3** from folders & files
- ⚙️ **Config-driven rules** (containers, externals, edges, components, dot-sourcing)
- 🧵 **Edge collapsing** (`×count`) so one clean line per relationship
- 🧩 **Fragments**: compose hand-authored JSONs into tailored views
- 📤 **Export to single HTML** (Mermaid in browser; no installs)
- 🧰 **VS Code tasks** (optional one-click runs)
- 🧼 Respects **exclude dirs** (`.git`, `.vscode`, etc.)
- 🧯 Split **C3**: one component diagram per container (with chunking)

---

## 🚀 Quick Start

```bash
# 1) Scan a folder (auto)
python likec4mini.py scan "C:\path\to\project" --include-components --out diagrams.md

# 2) With a rules config
python likec4mini.py scan "C:\path\to\project" --config project.config.json --include-components --out project.md

# 3) Export to shareable HTML
python likec4mini.py export-html project.md --out project.html
```

Open `project.md` in VS Code’s Markdown Preview, or open `project.html` in your browser.

---

## 🔧 Notable options

- `--direction TD|LR|BT|RL` – layout direction (default **TD**)
- `--edge-mode per_label|pair|all` – collapse edges (default **per_label**)
- `--edge-min N` – hide relationships with fewer than N occurrences
- `--split-components` – emit **one C3 diagram per container**
- `--component-chunk N` – chunk large component sets (default 40)
- `--exclude ".git,.vscode,__pycache__"` – extra directories to ignore

---

## 🧰 Minimal configuration (project.config.json)

```json
{
  "meta": { "title": "Example Project" },
  "exclude_dirs": [".git", ".vscode", "__pycache__"],
  "ignore_hint_matches": ["docs", "documentation"],

  "container_hints": [
    { "match": "scripts",  "label": "Scripts",  "tags": ["C2"] },
    { "match": "modules",  "label": "Modules",  "tags": ["C2"] },
    { "match": "sql",      "label": "Database", "tags": ["C2","db"] }
  ],

  "externals": [
    { "id": "sql", "label": "External: Database", "tags": ["C2","db"] }
  ],

  "edge_rules": [
    { "pattern": "\\bInvoke-Sqlcmd\\b", "label": "T-SQL", "target_id": "sql", "tags": ["db"], "file_exts": [".ps1", ".psm1"] }
  ],

  "component_rules": [
    { "parent_id": "sql", "pattern": "\\b(tbl[A-Za-z0-9_]+)\\b", "capture_group": 1, "max": 25, "tags": ["db"] }
  ],

  "link_rules": [
    { "type": "dot_source",    "label": "dot-source" },
    { "type": "import_module", "label": "Import-Module" }
  ],

  "max_components": 12
}
```

> For different stacks (PnP, SPMT, queues, HTTP, gRPC, etc.), add more `externals` and `edge_rules` with your regexes.

---

## 🧮 Other Commands

| Command | Description |
|--------|-------------|
| `scan` | Analyze code or use a config to generate diagrams |
| `flow` | Call graphs (Python or PowerShell) |
| `view` | Merge multiple JSON fragments and render Markdown |
| `merge` | Merge fragments → single model.json |
| `render` | Render model.json → Markdown |
| `export-html` | Convert Markdown (with Mermaid) → single HTML |

---

## 🧑‍💻 VS Code Integration (optional)

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "LikeC4 Mini: Scan (with config)",
      "type": "shell",
      "command": "python ${workspaceFolder}/likec4mini.py scan ${workspaceFolder} --config ${workspaceFolder}/project.config.json --include-components --direction TD --edge-mode per_label --out project.md"
    },
    {
      "label": "LikeC4 Mini: Export HTML",
      "type": "shell",
      "command": "python ${workspaceFolder}/likec4mini.py export-html ${workspaceFolder}/project.md --out ${workspaceFolder}/project.html"
    }
  ]
}
```

Run **Terminal → Run Task → LikeC4 Mini: Scan (with config)**.

---

## 🧠 Notes & Limitations

- Mermaid in VS Code requires a fairly recent build. If preview struggles, export to HTML.
- On restricted networks, download `mermaid.min.js` once and point export to it:
  `python likec4mini.py export-html project.md --out project.html --mermaid assets/mermaid.min.js`
- Diagrams convey structure; they aren’t full static analysis.

---

## 🧾 License

MIT License – free for internal or personal use.

---

## 💬 Credits

Inspired by [LikeC4](https://github.com/likec4/likec4) and the [C4 Model](https://c4model.com).
