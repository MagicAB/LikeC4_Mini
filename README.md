# LikeC4 Mini – Lightweight C4-Style Code Visualization

LikeC4 Mini is a **stand-alone, zero-dependency Python tool** for generating **C4-inspired architecture and flow diagrams** directly from source code.  
It’s designed for secure or restricted environments (like corporate or air-gapped systems) where tools like [LikeC4](https://github.com/likec4/likec4) or large visualization stacks can’t be approved or installed.

---

## ✨ Features

- 🧠 **Auto discovery** of systems, containers, and components from your folder structure  
- ⚙️ **Config-driven rules** for language or project-specific relationships (PowerShell, SQL, Python, etc.)  
- 📊 **Mermaid output** for easy visualization in Markdown or HTML  
- 🔁 **Fragment support** – combine small model JSONs into full C4 views  
- 📄 **Export to self-contained HTML** (renders anywhere without extra installs)  
- 💬 **No external dependencies** (Python standard library only)

---

## 🧩 Typical Use Cases

- Map out PowerShell automation frameworks  
- Visualize relationships between microservices or modules  
- Document large script repositories for compliance/governance  
- Generate architecture diagrams directly from code, for wikis or onboarding docs

---

## 🧱 Project Structure Example

```
LikeC4-Mini/
│
├── likec4mini.py          ← main script (drop anywhere)
├── project.config.json     ← project config defining containers, rules, externals
├── fragments/              ← optional small JSON diagram fragments
│   ├── featureA.json
│   ├── featureB.json
│   └── db.json
├── .vscode/
│   └── tasks.json          ← one-click VS Code automation
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Run a simple scan
```bash
python likec4mini.py scan "C:\path\to\project" --include-components --out diagrams.md
```

### 2️⃣ With a configuration file
```bash
python likec4mini.py scan "C:\path\to\project" --config project.config.json --include-components --out project.md
python likec4mini.py export-html project.md --out project.html
```
Then open `project.md` in VS Code’s Markdown Preview, or open `project.html` in a browser.

---

## 🧰 Configuration Example

```json
{
  "meta": { "title": "Example Project" },
  "container_hints": [
    { "match": "scripts",  "label": "Scripts", "tags": ["C2"] },
    { "match": "modules",  "label": "Modules", "tags": ["C2"] },
    { "match": "sql",      "label": "Database", "tags": ["C2","db"] }
  ],
  "externals": [
    { "id": "sql",  "label": "External: Database", "tags": ["C2","db"] }
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

---

## 🧮 Commands

| Command | Description |
|----------|--------------|
| `scan` | Auto-analyze code or use a config to generate diagrams |
| `flow` | Show call graphs (Python or PowerShell) |
| `view` | Merge multiple JSON fragments and render Markdown |
| `merge` | Merge fragments → single model.json |
| `render` | Render model.json → Markdown |
| `export-html` | Convert Markdown (with Mermaid) → standalone HTML |

---

## 🧑‍💻 VS Code Integration (optional)

Add a `.vscode/tasks.json` file to your project:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "LikeC4 Mini: Scan (with config)",
      "type": "shell",
      "command": "python ${workspaceFolder}/likec4mini.py scan ${workspaceFolder} --config ${workspaceFolder}/project.config.json --include-components --out project.md"
    },
    {
      "label": "LikeC4 Mini: Export HTML",
      "type": "shell",
      "command": "python ${workspaceFolder}/likec4mini.py export-html ${workspaceFolder}/project.md --out ${workspaceFolder}/project.html"
    }
  ]
}
```
Then run **Terminal → Run Task → LikeC4 Mini: Scan (with config)**.

---

## 🧠 Notes & Limitations

- Mermaid rendering in VS Code requires a recent build (1.90+).  
- On restricted networks, use a **local** `mermaid.min.js` and point to it with  
  `--mermaid assets/mermaid.min.js`.  
- Large repositories may produce very dense diagrams—use tags (`--tags C2`) to filter layers.  
- The goal is **structural insight**, not full static analysis.

---

## 🧾 License

MIT License – free for internal or personal use.

---

## 💬 Credits

Inspired by [LikeC4](https://github.com/likec4/likec4) and [C4 Model](https://c4model.com).  
Developed for environments where full stacks aren’t feasible — just **one Python file** and you’re diagramming.
