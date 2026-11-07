"""Rendered Sphinx documentation configuration.

Pipeline table:

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| EXTENSIONS | Load Sphinx-Needs and Sphinx Graphviz directives. | [mandatory] | R1 |
| NEED TYPES | Define component needs and directed flow links. | [mandatory] | R2 |
| RENDER | Render needflow and class-style Graphviz diagrams. | [mandatory] | R3 |

R1 The docs environment SHALL install the pinned Sphinx and Sphinx-Needs versions in requirements.txt.
R2 Component IDs and flows SHALL match the source-backed architecture page.
R3 WHEN Graphviz is installed, THEN a Sphinx HTML build SHALL render both diagrams; WHERE `dot` is absent from PATH, the local Graphviz install path is used when present.

SUPERSEDED 2026-11-07: PlantUML is used. Sphinx-Needs `needuml` renders UML through sphinxcontrib-plantuml, which needs Java and the PlantUML jar (1.2026.9, at PLANTUML_JAR or the default path in conf.py). The earlier CLOSED decision (Graphviz only) is withdrawn because needuml cannot run without PlantUML.

Preconditions: Sphinx, Sphinx-Needs, and Graphviz are installed. Failure modes: missing Sphinx extensions, Graphviz executable, or invalid need links stop the documentation build.
"""
from pathlib import Path
import os
import shutil

project = "ChunkGraph AGE Architecture"
root_doc = "index"
extensions = ["sphinx_needs", "sphinx.ext.graphviz", "sphinxcontrib.plantuml"]
PLANTUML_JAR = os.environ.get("PLANTUML_JAR", r"C:\Users\user\bin\plantuml.jar")
plantuml = f'java -jar "{PLANTUML_JAR}"'
plantuml_output_format = "svg"
exclude_patterns = ["_build"]
html_theme = "alabaster"

needs_types = [
    {
        "directive": "req",
        "title": "Requirement",
        "prefix": "REQ_",
        "color": "#fde68a",
        "style": "node",
    },
    {
        "directive": "comp",
        "title": "Component",
        "prefix": "C",
        "color": "#dbeafe",
        "style": "node",
    }
]
needs_links = {
    "feeds": {"incoming": "receives from", "outgoing": "feeds"},
    "produces": {"incoming": "receives from", "outgoing": "produces"},
    "persists": {"incoming": "is persisted by", "outgoing": "persists"},
    "writes_to": {"incoming": "is written by", "outgoing": "writes snapshot to"},
    "requests": {"incoming": "receives request from", "outgoing": "requests"},
    "reads": {"incoming": "is read by", "outgoing": "reads"},
    "queries": {"incoming": "is queried by", "outgoing": "queries"},
    "returns_neighbors": {"incoming": "returns neighbors to", "outgoing": "returns neighbors to"},
    "calls_model": {"incoming": "is called by", "outgoing": "calls for synthesis"},
    "returns_response": {"incoming": "returns response to", "outgoing": "returns response to"},
}
needs_flow_engine = "graphviz"
needs_flow_show_links = True

graphviz_dot = shutil.which("dot") or str(
    Path(r"C:\Program Files\Graphviz\bin\dot.exe")
    if Path(r"C:\Program Files\Graphviz\bin\dot.exe").exists()
    else "dot"
)
