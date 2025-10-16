#!/usr/bin/env python3
"""
LikeC4 Mini — Rules-based Scanner + Viewer + Exporter (stdlib only)
- Scan codebases and produce C4-ish diagrams (C1/C2/C3) as Mermaid in Markdown
- Config-driven rules (containers, externals, edges, components, links)
- PowerShell + Python call graphs
- Export Markdown → single-file HTML (with Mermaid)
- Edge collapsing (combine parallel edges with counts)
- Respect exclude dirs (config + --exclude)
- Split component (C3) view per container, with optional chunking
"""
import argparse, ast, json, os, re, glob, html
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

# ---------------- Defaults ----------------
EXCLUDE_DIRS_DEFAULT = {'.git','.hg','.svn','node_modules','.venv','venv','__pycache__','dist','build','out','bin','obj','.vscode'}
CODE_EXTENSIONS = {'.py','.js','.ts','.tsx','.cs','.java','.ps1','.psm1','.psd1','.sql'}

# ---------------- Utils ----------------
def norm_id(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

def relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root).replace('\\','/')
    except Exception:
        return path.replace('\\','/')

def read_text(path: str) -> str:
    try:
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            return f.read()
    except Exception:
        return ""

def write_text(path: str, data: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
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

# ---------------- Model ----------------
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

# ---------------- Auto-scan (no config) ----------------
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

    # naive import edges (py/js/ts)
    name_to_cid = {os.path.basename(path): cid for cid,path,_ in containers}
    for cid, _, code_files in containers:
        for fp in code_files:
            text=read_text(fp)
            for m in re.finditer(r'^(?:from|import)\s+([a-zA-Z0-9_\.]+)', text, flags=re.MULTILINE):
                target=m.group(1).split('.')[0]
                if target in name_to_cid and name_to_cid[target]!=cid:
                    model.add_edge(cid, name_to_cid[target], "import")
            for m in re.finditer(r'import\s+.*?from\s+[\'"](.+?)[\'"]', text):
                target=m.group(1).split('/')[0]
                if target in name_to_cid and name_to_cid[target]!=cid:
                    model.add_edge(cid, name_to_cid[target], "import")
    return model

# ---------------- Rules config ----------------
def load_rules_config(path: str) -> dict:
    raw=json.loads(read_text(path) or "{}")
    cfg={
        "meta": raw.get("meta", {}),
        "exclude_dirs": set(raw.get("exclude_dirs", [])),
        "ignore_hint_matches": set(raw.get("ignore_hint_matches", ["docs","documentation"])),
        "container_hints": raw.get("container_hints", []),
        "externals": raw.get("externals", []),
        "edge_rules": raw.get("edge_rules", []),
        "component_rules": raw.get("component_rules", []),
        "link_rules": raw.get("link_rules", []),
        "max_components": int(raw.get("max_components", 10))
    }
    for r in cfg["edge_rules"]:
        r["_re"]=re.compile(r["pattern"], re.IGNORECASE|re.MULTILINE)
        r["_exts"]=set(x.lower() for x in r.get("file_exts", []))
    for r in cfg["component_rules"]:
        r["_re"]=re.compile(r["pattern"])
        r["_max"]=int(r.get("max", 25))
        r["_cg"]=int(r.get("capture_group", 1))
    return cfg

def rules_infer_containers(root: str, system_id: str, model: Model, cfg: dict, extra_exclude: Set[str]) -> Dict[str,str]:
    result={}
    combined_exclude = EXCLUDE_DIRS_DEFAULT.union(set(cfg.get("exclude_dirs", []))).union(extra_exclude)
    for d in top_level_dirs(root, combined_exclude):
        base=os.path.basename(d); low=base.lower()
        if any(x in low for x in cfg.get("ignore_hint_matches", ["docs","documentation"])):
            continue
        label=None; tags=set()
        for hint in cfg.get("container_hints", []):
            if hint.get("match","").lower() in low:
                label=hint.get("label", base); tags=set(hint.get("tags", [])); break
        if label is None: label=base
        cid=norm_id(f"container_{label}")
        model.add_node(cid, label, 'container', parent=system_id, tags=tags)
        result[d]=cid
    # externals as containers
    for ext in cfg.get("externals", []):
        cid=norm_id(f"container_{ext['id']}")
        model.add_node(cid, ext["label"], 'container', parent=system_id, tags=set(ext.get("tags",[])))
    return result

def rules_apply(root: str, model: Model, dir_to_cid: Dict[str,str], cfg: dict):
    def target_cid(target_id: str) -> str:
        return norm_id(f"container_{target_id}")
    combined_exclude = EXCLUDE_DIRS_DEFAULT.union(set(cfg.get("exclude_dirs", [])))
    for dirpath, cid in dir_to_cid.items():
        files = list_code_files(dirpath, combined_exclude)
        sizes=[]
        for fp in files:
            try: loc=sum(1 for _ in open(fp,'r',encoding='utf-8',errors='ignore'))
            except Exception: loc=0
            sizes.append((loc,fp))
        sizes.sort(reverse=True)
        for _, fp in sizes[:cfg["max_components"]]:
            rel = relpath(fp, root)
            # show last folder + file for readability
            parts = rel.split("/")
            label = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            comp_id=norm_id(f"comp_{rel}")
            model.add_node(comp_id, label, 'component', parent=cid)

        for fp in files:
            ext=os.path.splitext(fp)[1].lower()
            text=read_text(fp)
            if not text: continue

            # prevent repeated edges per file
            seen_targets=set()

            for rule in cfg["edge_rules"]:
                if rule["_exts"] and ext not in rule["_exts"]:
                    continue
                if rule["_re"].search(text):
                    key=(cid, rule["target_id"], rule.get("label",""))
                    if key in seen_targets: 
                        continue
                    seen_targets.add(key)
                    model.add_edge(cid, target_cid(rule["target_id"]), rule.get("label",""), tags=set(rule.get("tags",[])))

            for cr in cfg["component_rules"]:
                matches=set(m.group(cr["_cg"]) for m in cr["_re"].finditer(text) if m.group(cr["_cg"]))
                for i,name in enumerate(sorted(matches)):
                    if i>=cr["_max"]: break
                    nid=norm_id(f"comp_{cr['parent_id']}_{name}")
                    model.add_node(nid, name, cr.get("type","component"), parent=target_cid(cr["parent_id"]), tags=set(cr.get("tags",[])))

            for lr in cfg.get("link_rules", []):
                if lr.get("type")=="dot_source":
                    for m in re.finditer(r'^\s*\.\s+["\']?(\.\\[^"\r\n]+)["\']?', text, flags=re.MULTILINE):
                        seg=m.group(1).lstrip('.\\/').split('\\')[0].split('/')[0]
                        tgt=None
                        for d2,c2 in dir_to_cid.items():
                            if os.path.basename(d2).lower()==seg.lower(): tgt=c2; break
                        if tgt and tgt!=cid:
                            model.add_edge(cid, tgt, lr.get("label","dot-source"))
                elif lr.get("type")=="import_module":
                    for m in re.finditer(r'^\s*Import-Module\s+([^\s\r\n]+)', text, flags=re.MULTILINE|re.IGNORECASE):
                        mod=m.group(1).strip("'\"")
                        if mod.startswith('.'):
                            seg=mod.strip('./\\').split('\\')[0].split('/')[0]
                            tgt=None
                            for d2,c2 in dir_to_cid.items():
                                if os.path.basename(d2).lower()==seg.lower(): tgt=c2; break
                            if tgt and tgt!=cid:
                                model.add_edge(cid, tgt, lr.get("label","Import-Module"))

# ---------------- Visibility & collapse ----------------
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

def _collapse_edges(edges, mode: str = "per_label", min_count: int = 1):
    if mode == "all":
        return edges
    if mode == "pair":
        counts = defaultdict(int)
        for e in edges:
            counts[(e.src, e.dst)] += 1
        out = []
        for (src, dst), c in counts.items():
            if c >= min_count:
                lbl = f"×{c}" if c > 1 else ""
                out.append(Edge(src, dst, lbl))
        return out
    # per_label
    by_pair_label = defaultdict(int)
    for e in edges:
        key = (e.src, e.dst, e.label or "")
        by_pair_label[key] += 1
    out = []
    for (src, dst, lab), c in by_pair_label.items():
        if c >= min_count:
            label = f"{lab} ×{c}" if lab and c > 1 else (f"×{c}" if c > 1 and not lab else lab)
            out.append(Edge(src, dst, label))
    return out

# subtree helper
def subtree_ids(model: Model, parent_id: str) -> Set[str]:
    keep=set()
    def walk(pid):
        for n in model.children_of(pid):
            keep.add(n.id)
            walk(n.id)
    walk(parent_id)
    return keep

# ---------------- Mermaid renderers ----------------
def mermaid_graph_from_model(model: Model,
                             level: str = 'container',
                             include_tags: Optional[Set[str]] = None,
                             direction: str = 'TD',
                             edge_mode: str = 'per_label',
                             edge_min: int = 1,
                             parent_filter: Optional[str] = None) -> str:
    keep_nodes, keep_edges = _visible(model, level, include_tags)
    if parent_filter:
        allowed = subtree_ids(model, parent_filter)
        allowed.add(parent_filter)
        keep_nodes = keep_nodes & allowed
        keep_edges = [e for e in keep_edges if e.src in keep_nodes and e.dst in keep_nodes]
    # Direction: components read better LR
    use_dir = (direction or 'TD').upper()
    if level == 'component':
        use_dir = 'LR'
    if use_dir not in ('TD','LR','BT','RL'):
        use_dir = 'TD'
    collapsed = _collapse_edges(keep_edges, mode=edge_mode, min_count=edge_min)

    init_line = "%%{init: {'flowchart': {'useMaxWidth': true, 'htmlLabels': true, 'diagramPadding': 8, 'nodeSpacing': 28, 'rankSpacing': 40}}}%%"

    lines = ["```mermaid",
             init_line,
             f"flowchart {use_dir}",]
    emitted: Set[str] = set()

    def emit_node(n: Node):
        if n.id in emitted:
            return
        lines.append(f'  {n.id}["{n.label}"]')
        emitted.add(n.id)

    def nest(parent: Optional[str], indent: int = 1):
        for ch in [x for x in model.children_of(parent) if x.id in keep_nodes]:
            if ch.type in ('system', 'container'):
                lines.append('  ' * indent + f'subgraph {ch.id}_sg["{ch.label}"]')
                emit_node(ch)
                nest(ch.id, indent + 1)
                lines.append('  ' * indent + 'end')
            else:
                emit_node(ch)

    # top-level roots
    if parent_filter and parent_filter in model.nodes:
        # When filtering to a container, treat that container as the root
        roots = [model.nodes[parent_filter]]
    else:
        roots = [x for x in model.children_of(None) if x.id in keep_nodes]
        if not roots:
            # try to find any system node
            roots = [n for n in model.nodes.values() if n.type=='system' and n.id in keep_nodes]
    for root in roots:
        lines.append(f'  subgraph {root.id}_sg["{root.label}"]')
        emit_node(root)
        nest(root.id, 2)
        lines.append('  end')

    for e in collapsed:
        if e.label:
            lines.append(f'  {e.src} -->|{e.label}| {e.dst}')
        else:
            lines.append(f'  {e.src} --> {e.dst}')
    lines.append("```")
    return "\n".join(lines)

def markdown_for_model(model: Model,
                       include_components: bool = False,
                       include_tags: Optional[Set[str]] = None,
                       direction: str = 'TD',
                       edge_mode: str = 'per_label',
                       edge_min: int = 1,
                       split_components: bool = True,
                       component_chunk: int = 12) -> str:
    parts = [f"# {model.title} — Diagrams",
             "",
             "## System Context (C1)",
             mermaid_graph_from_model(model, 'context', include_tags, direction, edge_mode, edge_min),
             "",
             "## Containers (C2)",
             mermaid_graph_from_model(model, 'container', include_tags, direction, edge_mode, edge_min)]
    if include_components:
        parts += ["", "## Components (C3)"]
        if not split_components:
            parts.append(mermaid_graph_from_model(model, 'component', include_tags, 'LR', edge_mode, edge_min))
        else:
            for node in model.nodes.values():
                if node.type != 'container':
                    continue
                comps = [n for n in model.nodes.values() if n.type=='component' and n.id in subtree_ids(model, node.id)]
                if not comps:
                    continue
                parts.append(f"### {node.label}")
                # simple per-container filtered graph (Mermaid handles size better this way)
                parts.append(mermaid_graph_from_model(model, 'component', include_tags, 'LR', edge_mode, edge_min, parent_filter=node.id))
    return "\n".join(parts)

# ---------------- Python diagrams ----------------
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

# ---------------- PowerShell call graph ----------------
FUNC_DEF = re.compile(r'^\s*function\s+([A-Za-z_][\w\-]*)\s*\{', re.IGNORECASE|re.MULTILINE)
def _brace_match(text: str, start_idx: int) -> int:
    depth=0
    for i,c in enumerate(text[start_idx:], start=start_idx):
        if c=='{': depth+=1
        elif c=='}':
            depth-=1
            if depth==0: return i
    return -1

def parse_ps_functions_and_calls(text: str):
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

# ---------------- JSON IO & merge ----------------
def read_json_model(path: str) -> Model:
    data=json.loads(read_text(path) or "{}")
    title=data.get("meta",{}).get("title") or os.path.splitext(os.path.basename(path))[0]
    model=Model(title)
    for n in data.get("nodes",[]):
        model.add_node(n["id"], n.get("label",n["id"]), n.get("type","component"), n.get("parent"), set(n.get("tags",[])))
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

# ---------------- HTML export ----------------
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
    if (mermaid_source or "cdn").lower()=="cdn":
        src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"
        return f'<script src="{html.escape(src)}"></script>'
    else:
        return f'<script src="{html.escape(mermaid_source)}"></script>'

def export_html_from_markdown(md_path: str, out_path: str, mermaid_source: str="cdn"):
    md=read_text(md_path)
    title=os.path.splitext(os.path.basename(md_path))[0]
    # Preserve order of headings and diagrams
    token_re = re.compile(r"(^#{1,6}\s+.*?$)|(```mermaid\s+[\s\S]*?```)", re.MULTILINE)
    pos=0; html_chunks=[]
    def h_to_html(line: str) -> str:
        m=re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m: return ""
        lvl=len(m.group(1)); text=m.group(2).strip()
        return f"<h{lvl}>{html.escape(text)}</h{lvl}>"
    for m in token_re.finditer(md):
        if m.start()>pos:
            chunk=md[pos:m.start()].strip()
            if chunk:
                html_chunks.append(f'<pre class="code">{html.escape(chunk)}</pre>')
        token=m.group(0)
        if token.startswith("#"):
            html_chunks.append(h_to_html(token.strip()))
        else:
            inner=re.sub(r"^```mermaid\s+|\s*```$","",token,flags=re.MULTILINE).strip()
            html_chunks.append(f'<div class="mblock"><div class="mermaid">{html.escape(inner)}</div></div>')
        pos=m.end()
    if pos<len(md):
        tail=md[pos:].strip()
        if tail:
            html_chunks.append(f'<pre class="code">{html.escape(tail)}</pre>')
    html_out=HTML_TEMPLATE.format(title=html.escape(title),
                                  mermaid_loader=build_mermaid_loader(mermaid_source),
                                  body="\n".join(html_chunks))
    write_text(out_path, html_out)

# ---------------- CLI ----------------
def main(argv=None):
    p=argparse.ArgumentParser(description="LikeC4 Mini — Scanner + Viewer + Exporter")
    sub=p.add_subparsers(dest="cmd", required=True)

    sp_scan=sub.add_parser("scan", help="Scan & render diagrams")
    sp_scan.add_argument("root"); sp_scan.add_argument("--title")
    sp_scan.add_argument("--exclude", default="")
    sp_scan.add_argument("--max-components", type=int, default=10)
    sp_scan.add_argument("--include-components", action="store_true")
    sp_scan.add_argument("--config", help="Rules JSON for containers/edges/components")
    sp_scan.add_argument("--out", default="diagrams.md")
    sp_scan.add_argument("--direction", choices=["TD","LR","BT","RL"], default="TD")
    sp_scan.add_argument("--edge-mode", choices=["all","pair","per_label"], default="per_label")
    sp_scan.add_argument("--edge-min", type=int, default=1)
    sp_scan.add_argument("--split-components", action="store_true", default=True)
    sp_scan.add_argument("--component-chunk", type=int, default=40)

    sp_render=sub.add_parser("render", help="Render a single JSON model to Markdown")
    sp_render.add_argument("model"); sp_render.add_argument("--out", default="model.md")
    sp_render.add_argument("--include-components", action="store_true")
    sp_render.add_argument("--tags", default="")
    sp_render.add_argument("--direction", choices=["TD","LR","BT","RL"], default="TD")
    sp_render.add_argument("--edge-mode", choices=["all","pair","per_label"], default="per_label")
    sp_render.add_argument("--edge-min", type=int, default=1)
    sp_render.add_argument("--split-components", action="store_true", default=True)
    sp_render.add_argument("--component-chunk", type=int, default=40)

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
    sp_view.add_argument("--direction", choices=["TD","LR","BT","RL"], default="TD")
    sp_view.add_argument("--edge-mode", choices=["all","pair","per_label"], default="per_label")
    sp_view.add_argument("--edge-min", type=int, default=1)
    sp_view.add_argument("--split-components", action="store_true", default=True)
    sp_view.add_argument("--component-chunk", type=int, default=40)
    sp_view.add_argument("--out", default="view.md")

    sp_html=sub.add_parser("export-html", help="Markdown (Mermaid) → single HTML")
    sp_html.add_argument("markdown"); sp_html.add_argument("--out", default="diagram.html")
    sp_html.add_argument("--mermaid", default="cdn")

    args=p.parse_args(argv)

    if args.cmd=="scan":
        exclude=set([x.strip() for x in args.exclude.split(',') if x.strip()])
        if args.config:
            cfg=load_rules_config(args.config)
            title=args.title or cfg["meta"].get("title") or os.path.basename(os.path.abspath(args.root))
            model=Model(title); system_id=norm_id(f"system_{title}")
            model.add_node(system_id, title, 'system', parent=None)
            dmap=rules_infer_containers(args.root, system_id, model, cfg, exclude)
            rules_apply(args.root, model, dmap, cfg)
            md=markdown_for_model(model, include_components=args.include_components,
                                  include_tags=None, direction=args.direction,
                                  edge_mode=args.edge_mode, edge_min=args.edge_min,
                                  split_components=args.split_components, component_chunk=args.component_chunk)
            write_text(args.out, md); print(f"Wrote {args.out}")
        else:
            model = build_auto_model(args.root, args.title, exclude, args.max_components)
            md=markdown_for_model(model, include_components=args.include_components,
                                  include_tags=None, direction=args.direction,
                                  edge_mode=args.edge_mode, edge_min=args.edge_min,
                                  split_components=args.split_components, component_chunk=args.component_chunk)
            write_text(args.out, md); print(f"Wrote {args.out}")

    elif args.cmd=="render":
        model=read_json_model(args.model)
        tags=set([t.strip() for t in args.tags.split(',') if t.strip()]) or None
        md=markdown_for_model(model, include_components=args.include_components,
                              include_tags=tags, direction=args.direction,
                              edge_mode=args.edge_mode, edge_min=args.edge_min,
                              split_components=args.split_components, component_chunk=args.component_chunk)
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
                # optional simple flow could be added here; keeping call/class for stdlib
                pass
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
        md=markdown_for_model(combined, include_components=args.include_components,
                              include_tags=tags, direction=args.direction,
                              edge_mode=args.edge_mode, edge_min=args.edge_min,
                              split_components=args.split_components, component_chunk=args.component_chunk)
        write_text(args.out, md); print(f"Composed {len(files)} files → {args.out}")

    elif args.cmd=="export-html":
        export_html_from_markdown(args.markdown, args.out, args.mermaid)
        print(f"Wrote {args.out}")

    else:
        p.print_help()

if __name__=="__main__":
    main()
