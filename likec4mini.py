#!/usr/bin/env python3
"""
LikeC4 Mini (Rules-based Scanner + Viewer + Exporter) — standard-library only.

- Scanner
  * Auto C4-ish (no config) OR Rule-driven (via --config <json>)
  * Python class/call graphs, PowerShell in-file call graphs
- Viewer/Composer
  * Merge many small JSON "fragments" into a model; tag filtering
- Exporter
  * Markdown (with ```mermaid blocks) → single-file HTML (shareable)

Examples:
  # Rule-driven scan (generic for any project)
  python likec4mini.py scan . --config aspmt.config.json --include-components --out aspmt.md
  # Auto scan (no config)
  python likec4mini.py scan . --include-components --out diagrams.md

  # Fragments → view
  python likec4mini.py view fragments/*.json --tags C2,db --out c2.md
  python likec4mini.py export-html c2.md --out c2.html --mermaid cdn

  # Flow
  python likec4mini.py flow Scripts/ProcessBlock.ps1 --out ps_flow.md
  python likec4mini.py flow src/module.py --func handle_request --out py_flow.md

How to Run:
  # Scan using the config rules
  python likec4mini.py scan . --config aspmt.config.json --include-components --out aspmt.md

  # Preview as markdown (VS Code), or export a shareable HTML:
  python likec4mini.py export-html aspmt.md --out aspmt.html
  # If you can't use a CDN, download mermaid.min.js once and do:
  python likec4mini.py export-html aspmt.md --out aspmt.html --mermaid assets/mermaid.min.js

  Fragments (hand-authored small pieces):
  python likec4mini.py view fragments/*.json --tags C2,db,aspmt --out platform.md
  python likec4mini.py export-html platform.md --out platform.html

Notes:
  * Keep IDs stable in fragments/config (sql, pnp, spmt, pre, mig, etc.) so views compose cleanly.
  * Use tags in nodes/edges and filter with --tags to create tailored views (e.g., only C2, or only db).
  * The rules engine is intentionally simple: regex → edge to target, regex capture → components under parent. No dependencies, easy to audit for approvals.
  * For PowerShell cross-file behavior, the rules already create edges for dot-source and Import-Module when enabled in link_rules.

"""
import argparse, ast, json, os, re, glob, html
from typing import Dict, List, Tuple, Set, Optional

# ------------------------------
# Defaults
# ------------------------------
EXCLUDE_DIRS_DEFAULT = {'.git','.hg','.svn','node_modules','.venv','venv','__pycache__','dist','build','out','bin','obj'}
CODE_EXTENSIONS = {'.py','.js','.ts','.tsx','.cs','.java','.ps1','.psm1','.psd1','.sql'}

# ------------------------------
# Utilities
# ------------------------------
def norm_id(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

def relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root).replace('\\', '/')
    except Exception:
        return path.replace('\\', '/')

def read_text(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""

def write_text(path: str, data: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)

def list_code_files(root: str, exclude_dirs: Set[str]) -> List[str]:
    out=[]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in CODE_EXTENSIONS:
                out.append(os.path.join(dirpath, fn))
    return out

def top_level_dirs(root: str, exclude: Set[str]) -> List[str]:
    out=[]
    for name in os.listdir(root):
        p=os.path.join(root,name)
        if os.path.isdir(p) and name not in exclude:
            out.append(p)
    return out

# ------------------------------
# Model
# ------------------------------
class Node:
    def __init__(self, id: str, label: str, type_: str, parent: Optional[str]=None, tags: Optional[Set[str]]=None):
        self.id=id; self.label=label; self.type=type_; self.parent=parent; self.tags=set(tags or [])

class Edge:
    def __init__(self, src: str, dst: str, label: str="", tags: Optional[Set[str]]=None):
        self.src=src; self.dst=dst; self.label=label; self.tags=set(tags or [])

class Model:
    def __init__(self, title: str):
        self.title=title
        self.nodes: Dict[str, Node]={}
        self.edges: List[Edge]=[]

    def add_node(self, id: str, label: str, type_: str, parent: Optional[str]=None, tags: Optional[Set[str]]=None):
        if id in self.nodes:
            n=self.nodes[id]
            n.label = label or n.label
            n.type  = type_ or n.type
            n.parent = parent if parent is not None else n.parent
            n.tags |= set(tags or [])
        else:
            self.nodes[id]=Node(id,label,type_,parent,tags)

    def add_edge(self, src: str, dst: str, label: str="", tags: Optional[Set[str]]=None):
        self.edges.append(Edge(src,dst,label,tags))

    def children_of(self, parent_id: Optional[str]) -> List[Node]:
        return [n for n in self.nodes.values() if n.parent==parent_id]

# ------------------------------
# Auto-scan (no config)
# ------------------------------
def build_auto_model(root: str, title: Optional[str], exclude_dirs: Set[str], max_components_per_container: int=10) -> Model:
    title = title or os.path.basename(os.path.abspath(root))
    model = Model(title)
    system_id = norm_id(f"system_{title}")
    model.add_node(system_id, title, 'system', parent=None)

    tlds = top_level_dirs(root, EXCLUDE_DIRS_DEFAULT.union(exclude_dirs))
    containers=[]
    for d in tlds:
        code_files = list_code_files(d, EXCLUDE_DIRS_DEFAULT.union(exclude_dirs))
        if not code_files: continue
        cid = norm_id(f"container_{os.path.basename(d)}")
        model.add_node(cid, os.path.basename(d), 'container', parent=system_id)
        containers.append((cid,d,code_files))

    for cid, dirpath, code_files in containers:
        sizes=[]
        for fp in code_files:
            try: loc=sum(1 for _ in open(fp,'r',encoding='utf-8',errors='ignore'))
            except Exception: loc=0
            sizes.append((loc,fp))
        sizes.sort(reverse=True)
        for _, fp in sizes[:max_components_per_container]:
            comp_id=norm_id(f"comp_{relpath(fp,root)}")
            model.add_node(comp_id, relpath(fp,root), 'component', parent=cid)

    # naive cross-container imports (py/js/ts)
    name_to_cid = {os.path.basename(path): cid for cid,path,_ in containers}
    for cid, _, code_files in containers:
        for fp in code_files:
            text=read_text(fp)
            # python
            for m in re.finditer(r'^(?:from|import)\s+([a-zA-Z0-9_\.]+)', text, flags=re.MULTILINE):
                target=m.group(1).split('.')[0]
                if target in name_to_cid and name_to_cid[target]!=cid:
                    model.add_edge(cid, name_to_cid[target], "import")
            # js/ts
            for m in re.finditer(r'import\s+.*?from\s+[\'"](.+?)[\'"]', text):
                target=m.group(1).split('/')[0]
                if target in name_to_cid and name_to_cid[target]!=cid:
                    model.add_edge(cid, name_to_cid[target], "import")
    return model

# ------------------------------
# Rules config loader (generic)
# ------------------------------
def load_rules_config(path: str) -> dict:
    raw = json.loads(read_text(path) or "{}")
    # normalize expected keys
    cfg = {
        "meta": raw.get("meta", {}),
        "exclude_dirs": set(raw.get("exclude_dirs", [])),
        "ignore_hint_matches": set(raw.get("ignore_hint_matches", ["docs","documentation"])),
        "container_hints": raw.get("container_hints", []),  # list of {match,label,tags?}
        "externals": raw.get("externals", []),              # list of {id,label,tags?}
        "edge_rules": raw.get("edge_rules", []),            # list of {pattern,label,target_id,tags?,file_exts?}
        "component_rules": raw.get("component_rules", []),  # list of {parent_id,pattern,capture_group,max?,type?,tags?}
        "link_rules": raw.get("link_rules", []),            # [{type:"dot_source"/"import_module", label:"..."}]
        "max_components": int(raw.get("max_components", 10))
    }
    # pre-compile regex for speed
    for r in cfg["edge_rules"]:
        r["_re"] = re.compile(r["pattern"], re.IGNORECASE|re.MULTILINE)
        r["_exts"] = set(x.lower() for x in r.get("file_exts", []))
    for r in cfg["component_rules"]:
        r["_re"] = re.compile(r["pattern"])
        r["_max"] = int(r.get("max", 25))
        r["_cg"]  = int(r.get("capture_group", 1))
    return cfg

def rules_infer_containers(root: str, system_id: str, model: Model, cfg: dict) -> Dict[str,str]:
    """Map top-level dir -> container_id based on container_hints; also add externals."""
    result={}
    for d in top_level_dirs(root, set()):
        base = os.path.basename(d)
        low  = base.lower()
        if any(x in low for x in cfg["ignore_hint_matches"]):  # skip docs, etc.
            continue
        label = None
        tags  = set()
        for hint in cfg["container_hints"]:
            if hint.get("match","").lower() in low:
                label = hint.get("label", base)
                tags  = set(hint.get("tags", []))
                break
        if label is None:
            label = base
        cid = norm_id(f"container_{label}")
        model.add_node(cid, label, 'container', parent=system_id, tags=tags)
        result[d]=cid
    # externals as containers
    for ext in cfg["externals"]:
        cid = norm_id(f"container_{ext['id']}")
        model.add_node(cid, ext["label"], 'container', parent=system_id, tags=set(ext.get("tags",[])))
    return result

def rules_apply(root: str, model: Model, dir_to_cid: Dict[str,str], cfg: dict):
    """Apply edge_rules, component_rules, and link_rules (dot-source/import-module)."""
    # Helper: map logical target_id -> container node id
    def target_cid(target_id: str) -> str:
        return norm_id(f"container_{target_id}")
    # Per-container: add components (top-N by size)
    for dirpath, cid in dir_to_cid.items():
        files = list_code_files(dirpath, EXCLUDE_DIRS_DEFAULT.union(cfg["exclude_dirs"]))
        sizes=[]
        for fp in files:
            try: loc=sum(1 for _ in open(fp,'r',encoding='utf-8',errors='ignore'))
            except Exception: loc=0
            sizes.append((loc,fp))
        sizes.sort(reverse=True)
        for _, fp in sizes[:cfg["max_components"]]:
            comp_id=norm_id(f"comp_{relpath(fp,root)}")
            model.add_node(comp_id, relpath(fp,root), 'component', parent=cid)

        # Scan files with rules
        for fp in files:
            ext = os.path.splitext(fp)[1].lower()
            text=read_text(fp)
            if not text: continue

            # edge_rules: regex presence => edge to target_id
            for rule in cfg["edge_rules"]:
                if rule["_exts"] and ext not in rule["_exts"]:
                    continue
                if rule["_re"].search(text):
                    model.add_edge(cid, target_cid(rule["target_id"]), rule.get("label",""), tags=set(rule.get("tags",[])))

            # component_rules: surface named things (e.g., DB tables)
            for cr in cfg["component_rules"]:
                matches = set(m.group(cr["_cg"]) for m in cr["_re"].finditer(text) if m.group(cr["_cg"]))
                for i, name in enumerate(sorted(matches)):
                    if i>=cr["_max"]: break
                    nid = norm_id(f"comp_{cr['parent_id']}_{name}")
                    model.add_node(nid, name, cr.get("type","component"),
                                   parent=target_cid(cr["parent_id"]),
                                   tags=set(cr.get("tags",[])))

            # link_rules
            for lr in cfg["link_rules"]:
                if lr.get("type")=="dot_source":
                    for m in re.finditer(r'^\s*\.\s+["\']?(\.\\[^"\r\n]+)["\']?', text, flags=re.MULTILINE):
                        seg = m.group(1).lstrip('.\\/').split('\\')[0].split('/')[0]
                        tgt=None
                        for d2,c2 in dir_to_cid.items():
                            if os.path.basename(d2).lower()==seg.lower():
                                tgt=c2; break
                        if tgt and tgt!=cid:
                            model.add_edge(cid, tgt, lr.get("label","dot-source"))
                elif lr.get("type")=="import_module":
                    for m in re.finditer(r'^\s*Import-Module\s+([^\s\r\n]+)', text, flags=re.MULTILINE|re.IGNORECASE):
                        mod=m.group(1).strip("'\"")
                        if mod.startswith('.'):
                            seg=mod.strip('./\\').split('\\')[0].split('/')[0]
                            tgt=None
                            for d2,c2 in dir_to_cid.items():
                                if os.path.basename(d2).lower()==seg.lower():
                                    tgt=c2; break
                            if tgt and tgt!=cid:
                                model.add_edge(cid, tgt, lr.get("label","Import-Module"))

# ------------------------------
# Mermaid renderers (with tag filtering)
# ------------------------------
def _visible(model: Model, level: str, include_tags: Optional[Set[str]]) -> Tuple[Set[str], List[Edge]]:
    def ok_node(n: Node) -> bool:
        if level=='context' and n.type!='system': return False
        if level=='container' and n.type=='component': return False
        if include_tags:
            return (not n.tags) or bool(n.tags & include_tags)
        return True
    keep_nodes={nid for nid,n in model.nodes.items() if ok_node(n)}
    keep_edges=[]
    for e in model.edges:
        if e.src in keep_nodes and e.dst in keep_nodes:
            if include_tags:
                if (not e.tags) or bool(e.tags & include_tags):
                    keep_edges.append(e)
            else:
                keep_edges.append(e)
    return keep_nodes, keep_edges

def mermaid_graph_from_model(model: Model, level: str='container', include_tags: Optional[Set[str]]=None) -> str:
    keep_nodes, keep_edges = _visible(model, level, include_tags)

    lines = ["```mermaid", "graph LR"]
    emitted: Set[str] = set()

    def emit_node(n: Node):
        if n.id in emitted:
            return
        lines.append(f'  {n.id}["{n.label}"]')
        emitted.add(n.id)

    def nest(parent: Optional[str], indent: int = 1):
        # render containers/systems as subgraphs; plain components as nodes
        for ch in [x for x in model.children_of(parent) if x.id in keep_nodes]:
            if ch.type in ('system', 'container'):
                lines.append('  ' * indent + f'subgraph {ch.id}_sg["{ch.label}"]')
                emit_node(ch)
                nest(ch.id, indent + 1)
                lines.append('  ' * indent + 'end')
            else:
                lines.append('  ' * indent + '%% component')
                emit_node(ch)

    # start from top-level roots (parent=None)
    for root in [x for x in model.children_of(None) if x.id in keep_nodes]:
        lines.append(f'  subgraph {root.id}_sg["{root.label}"]')
        emit_node(root)
        nest(root.id, 2)
        lines.append('  end')

    # edges last (between the single set of emitted nodes)
    for e in keep_edges:
        if e.label:
            lines.append(f'  {e.src} -->|{e.label}| {e.dst}')
        else:
            lines.append(f'  {e.src} --> {e.dst}')

    lines.append("```")
    return "\n".join(lines)


def markdown_for_model(model: Model, include_components: bool=False, include_tags: Optional[Set[str]]=None) -> str:
    parts=[f"# {model.title} — Diagrams","",
           "## System Context (C1)",
           mermaid_graph_from_model(model,'context', include_tags),
           "",
           "## Containers (C2)",
           mermaid_graph_from_model(model,'container', include_tags)]
    if include_components:
        parts+=["","## Components (C3)", mermaid_graph_from_model(model,'component', include_tags)]
    return "\n".join(parts)

# ------------------------------
# Python diagrams
# ------------------------------
def mermaid_class_diagram_from_python(file_path: str) -> str:
    src=read_text(file_path)
    if not src: return ""
    try: tree=ast.parse(src)
    except Exception: return ""
    classes=[]; relationships=[]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods=[]; bases=[]
            for b in node.bases:
                if isinstance(b,ast.Name): bases.append(b.id)
                elif isinstance(b,ast.Attribute): bases.append(b.attr)
                else:
                    try: bases.append(ast.unparse(b))
                    except Exception: bases.append("Base")
            for item in node.body:
                if isinstance(item, ast.FunctionDef): methods.append(item.name+"()")
            classes.append((node.name,methods,bases))
            for base in bases: relationships.append((node.name,base))
    if not classes: return ""
    lines=["```mermaid","classDiagram"]
    for cls,methods,_ in classes:
        lines.append(f"  class {cls} {{")
        for m in methods: lines.append(f"    {m}")
        lines.append("  }")
    for child,base in relationships:
        lines.append(f"  {base} <|-- {child}")
    lines.append("```")
    return "\n".join(lines)

def mermaid_call_graph_from_python(file_path: str) -> str:
    src=read_text(file_path)
    if not src: return ""
    try: tree=ast.parse(src)
    except Exception: return ""
    defs:set=set(); calls:List[Tuple[str,str]]=[]; stack=[]
    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            defs.add(n.name); stack.append(n.name); self.generic_visit(n); stack.pop()
        def visit_Call(self, n):
            name=None
            if isinstance(n.func, ast.Name): name=n.func.id
            elif isinstance(n.func, ast.Attribute): name=n.func.attr
            if name and stack: calls.append((stack[-1], name))
            self.generic_visit(n)
    V().visit(tree)
    if not defs: return ""
    lines=["```mermaid","graph TD"]
    for fn in sorted(defs): lines.append(f'  {norm_id(fn)}["{fn}()"]')
    seen=set()
    for a,b in calls:
        if b in defs:
            lines.append(f"  {norm_id(a)} --> {norm_id(b)}")
        else:
            ext=norm_id("ext_"+b)
            if ext not in seen:
                lines.append(f'  {ext}(("{b}"))'); seen.add(ext)
            lines.append(f"  {norm_id(a)} --> {ext}")
    lines.append("```")
    return "\n".join(lines)

# ------------------------------
# PowerShell in-file call graph
# ------------------------------
FUNC_DEF = re.compile(r'^\s*function\s+([A-Za-z_][\w\-]*)\s*\{', re.IGNORECASE|re.MULTILINE)

def _brace_match(text: str, start_idx: int) -> int:
    depth=0
    for i,c in enumerate(text[start_idx:], start=start_idx):
        if c=='{': depth+=1
        elif c=='}':
            depth-=1
            if depth==0: return i
    return -1

def parse_ps_functions_and_calls(text: str) -> Tuple[Set[str], List[Tuple[str,str]], Set[str]]:
    defs=set(m.group(1) for m in FUNC_DEF.finditer(text))
    bodies={}
    for m in FUNC_DEF.finditer(text):
        name=m.group(1); open_idx=text.find('{', m.end()-1)
        if open_idx==-1: continue
        close_idx=_brace_match(text, open_idx)
        if close_idx!=-1: bodies[name]=text[open_idx+1:close_idx]
    calls=[]; externals=set()
    for caller, body in bodies.items():
        for callee in sorted(defs, key=len, reverse=True):
            pat=re.compile(r'(?<![\w-])'+re.escape(callee)+r'(?:\s|\(|$|-)', re.IGNORECASE)
            for _ in pat.finditer(body):
                if callee.lower()!=caller.lower(): calls.append((caller,callee))
        for ext in re.findall(r'(?m)^\s*([A-Za-z]+-[A-Za-z0-9]+)', body):
            if ext not in defs: externals.add(ext)
    return defs, calls, externals

def mermaid_call_graph_from_powershell(file_path: str) -> str:
    text=read_text(file_path)
    if not text: return ""
    defs,calls,externals = parse_ps_functions_and_calls(text)
    if not defs and not externals: return ""
    lines=["```mermaid","graph TD"]
    for fn in sorted(defs): lines.append(f'  {norm_id(fn)}["{fn}()"]')
    for ext in sorted(externals): lines.append(f'  {norm_id("ext_"+ext)}(("{ext}"))')
    for a,b in calls:
        tgt = norm_id(b) if b in defs else norm_id("ext_"+b)
        lines.append(f"  {norm_id(a)} --> {tgt}")
    lines.append("```")
    return "\n".join(lines)

# ------------------------------
# JSON IO + fragments + merge
# ------------------------------
def read_json_model(path: str) -> Model:
    data=json.loads(read_text(path) or "{}")
    title=data.get("meta",{}).get("title") or os.path.splitext(os.path.basename(path))[0]
    model=Model(title)
    for n in data.get("nodes",[]):
        model.add_node(n["id"], n.get("label",n["id"]), n.get("type","component"),
                       n.get("parent"), set(n.get("tags",[])))
    for e in data.get("edges",[]):
        model.add_edge(e["from"], e["to"], e.get("label",""), set(e.get("tags",[])))
    return model

def write_json_model(model: Model, path: str) -> None:
    data={
        "meta":{"title": model.title},
        "nodes":[{"id":n.id,"label":n.label,"type":n.type,"parent":n.parent,"tags":sorted(n.tags)} for n in model.nodes.values()],
        "edges":[{"from":e.src,"to":e.dst,"label":e.label,"tags":sorted(e.tags)} for e in model.edges]
    }
    write_text(path, json.dumps(data, indent=2))

def merge_models(models: List[Model], title: str="Composed View") -> Model:
    out=Model(title)
    for m in models:
        for n in m.nodes.values():
            out.add_node(n.id, n.label, n.type, n.parent, n.tags)
        for e in m.edges:
            out.add_edge(e.src, e.dst, e.label, e.tags)
    return out

# ------------------------------
# Markdown → single-file HTML exporter
# ------------------------------
HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;}}
h1,h2,h3{{margin:12px 0}} .mblock{{margin:16px 0;}}
pre.code{{background:#f6f8fa;padding:12px;border-radius:6px;overflow:auto}}
</style>
{mermaid_loader}
<script>mermaid.initialize({{startOnLoad:true, securityLevel:'loose'}});</script>
</head><body>
<h1>{title}</h1>
{body}
<hr/><p style="font-size:12px;color:#888">Generated by LikeC4 Mini</p>
</body></html>"""

def build_mermaid_loader(mermaid_source: str) -> str:
    if mermaid_source.lower()=="cdn":
        src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"
        return f'<script src="{html.escape(src)}"></script>'
    else:
        return f'<script src="{html.escape(mermaid_source)}"></script>'

def export_html_from_markdown(md_path: str, out_path: str, mermaid_source: str="cdn"):
    md = read_text(md_path)

    # Tokenize: keep order of headings and mermaid blocks
    token_re = re.compile(r"(^#{1,6}\s+.*?$)|(```mermaid\s+[\s\S]*?```)", re.MULTILINE)
    pos = 0
    html_chunks = []
    title = os.path.splitext(os.path.basename(md_path))[0]

    def h_to_html(line: str) -> str:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m: return ""
        level = len(m.group(1)); text = m.group(2).strip()
        return f"<h{level}>{html.escape(text)}</h{level}>"

    for m in token_re.finditer(md):
        # any plain text before this token
        if m.start() > pos:
            chunk = md[pos:m.start()].strip()
            if chunk:
                html_chunks.append(f'<pre class="code">{html.escape(chunk)}</pre>')
        token = m.group(0)
        if token.startswith("#"):
            html_chunks.append(h_to_html(token.strip()))
        else:
            # mermaid block
            inner = re.sub(r"^```mermaid\s+|\s*```$", "", token, flags=re.MULTILINE).strip()
            html_chunks.append(f'<div class="mblock"><div class="mermaid">{html.escape(inner)}</div></div>')
        pos = m.end()

    # tail text
    if pos < len(md):
        tail = md[pos:].strip()
        if tail:
            html_chunks.append(f'<pre class="code">{html.escape(tail)}</pre>')

    html_out = HTML_TEMPLATE.format(
        title = html.escape(title),
        mermaid_loader = build_mermaid_loader(mermaid_source),
        body = "\n".join(html_chunks)
    )
    write_text(out_path, html_out)

# ------------------------------
# CLI
# ------------------------------
def main(argv=None):
    p=argparse.ArgumentParser(description="LikeC4 Mini: Rules-based Scanner + Viewer + Exporter")
    sub=p.add_subparsers(dest="cmd", required=True)

    sp_scan=sub.add_parser("scan", help="Scan & render diagrams")
    sp_scan.add_argument("root"); sp_scan.add_argument("--title")
    sp_scan.add_argument("--exclude", default="")
    sp_scan.add_argument("--max-components", type=int, default=10)
    sp_scan.add_argument("--include-components", action="store_true")
    sp_scan.add_argument("--config", help="Rules JSON for containers/edges/components")
    sp_scan.add_argument("--out", default="diagrams.md")

    sp_render=sub.add_parser("render", help="Render a single JSON model to Markdown")
    sp_render.add_argument("model"); sp_render.add_argument("--out", default="model.md")
    sp_render.add_argument("--include-components", action="store_true")
    sp_render.add_argument("--tags", default="")

    sp_flow=sub.add_parser("flow", help="Call/class graphs; optional Python function flowchart")
    sp_flow.add_argument("file"); sp_flow.add_argument("--func")
    sp_flow.add_argument("--out", default="flow.md")
    sp_flow.add_argument("--skip-calls", action="store_true")
    sp_flow.add_argument("--skip-classes", action="store_true")

    sp_merge=sub.add_parser("merge", help="Merge fragment JSONs → combined model.json")
    sp_merge.add_argument("inputs", nargs="+")
    sp_merge.add_argument("--title", default="Composed View")
    sp_merge.add_argument("--out", default="model.json")

    sp_view=sub.add_parser("view", help="Merge fragments then render to Markdown")
    sp_view.add_argument("inputs", nargs="+")
    sp_view.add_argument("--title", default="Composed View")
    sp_view.add_argument("--include-components", action="store_true")
    sp_view.add_argument("--tags", default="")
    sp_view.add_argument("--out", default="view.md")

    sp_html=sub.add_parser("export-html", help="Markdown (Mermaid) → single HTML")
    sp_html.add_argument("markdown"); sp_html.add_argument("--out", default="diagram.html")
    sp_html.add_argument("--mermaid", default="cdn")

    args=p.parse_args(argv)

    if args.cmd=="scan":
        exclude=set([x.strip() for x in args.exclude.split(',') if x.strip()])
        if args.config:
            cfg=load_rules_config(args.config)
            title = args.title or cfg["meta"].get("title") or os.path.basename(os.path.abspath(args.root))
            model=Model(title); system_id=norm_id(f"system_{title}")
            model.add_node(system_id, title, 'system', parent=None)
            # containers via hints + externals
            dmap = rules_infer_containers(args.root, system_id, model, cfg)
            # apply rules
            rules_apply(args.root, model, dmap, cfg)
            md=markdown_for_model(model, include_components=args.include_components)
            write_text(args.out, md); print(f"Wrote {args.out}")
        else:
            model = build_auto_model(args.root, args.title, exclude, args.max_components)
            md=markdown_for_model(model, include_components=args.include_components)
            write_text(args.out, md); print(f"Wrote {args.out}")

    elif args.cmd=="render":
        model=read_json_model(args.model)
        tags=set([t.strip() for t in args.tags.split(',') if t.strip()]) or None
        md=markdown_for_model(model, include_components=args.include_components, include_tags=tags)
        write_text(args.out, md); print(f"Wrote {args.out}")

    elif args.cmd=="flow":
        parts=[f"# Flow for {args.file}"]
        ext=os.path.splitext(args.file)[1].lower()
        if ext in {'.ps1','.psm1'}:
            if not args.skip_calls:
                cg=mermaid_call_graph_from_powershell(args.file)
                if cg: parts+=["","## PowerShell Call Graph", cg]
        else:
            if args.func:
                ff=mermaid_simple_flow_for_function(args.file, args.func)
                if ff: parts+=["",f"## Function Flow — {args.func}()", ff]
            if not args.skip_classes:
                cd=mermaid_class_diagram_from_python(args.file)
                if cd: parts+=["","## Classes", cd]
            if not args.skip_calls:
                cg=mermaid_call_graph_from_python(args.file)
                if cg: parts+=["","## Call Graph", cg]
        write_text(args.out, "\n".join(parts)); print(f"Wrote {args.out}")

    elif args.cmd=="merge":
        files=[]
        for pattern in args.inputs:
            files.extend(glob.glob(pattern))
        models=[read_json_model(p) for p in files]
        combined=merge_models(models, title=args.title)
        write_json_model(combined, args.out); print(f"Merged {len(files)} files → {args.out}")

    elif args.cmd=="view":
        files=[]
        for pattern in args.inputs:
            files.extend(glob.glob(pattern))
        models=[read_json_model(p) for p in files]
        combined=merge_models(models, title=args.title)
        tags=set([t.strip() for t in args.tags.split(',') if t.strip()]) or None
        md=markdown_for_model(combined, include_components=args.include_components, include_tags=tags)
        write_text(args.out, md); print(f"Composed {len(files)} files → {args.out}")

    elif args.cmd=="export-html":
        export_html_from_markdown(args.markdown, args.out, args.mermaid)
        print(f"Wrote {args.out}")

    else:
        p.print_help()

if __name__=="__main__":
    main()
