"""Scratch: generate needuml class diagrams from docs/entity_model/*.md.

Writes docs/sphinx/uml-class-view.rst with one needuml block per module.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "entity_model"
OUT = ROOT / "docs" / "sphinx" / "uml-class-view.rst"
MODULE_FILES = [
    ("chunk-graph", "Chunk graph"),
    ("lexical-index", "Lexical index"),
    ("section-map", "Section map"),
    ("global-aggregate", "Global aggregate"),
]
REL = re.compile(r'^\s+([A-Z_]+) \|\|--(o\{|o\|) ([A-Z_]+) : "(.*)"\s*$')
CARD = {"o{": '"0..*"', "o|": '"0..1"'}


def parse(md: str):
    classes, rels = {}, []
    entity = None
    in_table = False
    for line in md.splitlines():
        if line.startswith("### "):
            entity = line[4:].strip()
            classes[entity] = []
            in_table = False
            continue
        if entity and line.startswith("| Attribute"):
            in_table = True
            continue
        if in_table and re.match(r"^\|[-| ]+\|$", line):
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 5:
                continue
            name, _desc, dtype, _length, validation = cells
            stereo = ""
            if "Primary Key" in validation and "Foreign Key" in validation:
                stereo = " <<PK,FK>>"
            elif "Primary Key" in validation:
                stereo = " <<PK>>"
            elif "Foreign Key" in validation:
                stereo = " <<FK>>"
            classes[entity].append(f"  {name} : {dtype}{stereo}")
            continue
        if line.strip() == "" or line.startswith("#"):
            in_table = False
        m = REL.match(line)
        if m:
            left, card, right, _label = m.groups()
            rels.append((left, card, right, _label))
    return classes, rels


def block(classes, rels) -> list[str]:
    out = ["@startuml"]
    for name, attrs in classes.items():
        out.append(f"class {name} {{")
        out += attrs
        out.append("}")
    for left, card, right, label in rels:
        out.append(f'{left} "1" -- {CARD[card]} {right} : {label}')
    out.append("@enduml")
    return out


def main() -> None:
    lines = [
        "UML class view of the entity model",
        "==================================",
        "",
        "Class diagrams generated from ``docs/entity_model/*.md`` by",
        "``tools/entity_to_uml.py``. Primary and foreign keys are stereotyped. Each",
        "association is a foreign key from the entity model's relationship diagram.",
        "",
        "This view documents the entity model. It does not yet inform it: the",
        "conceptual class model that the sw-design flow puts before the entity model",
        "has not been authored from the requirements.",
        "",
    ]
    for slug, heading in MODULE_FILES:
        md = (SRC / f"{slug}.md").read_text(encoding="utf-8")
        classes, rels = parse(md)
        n_attr = sum(len(v) for v in classes.values())
        print(f"{slug:18s} classes={len(classes)} attributes={n_attr} associations={len(rels)}")
        lines += [heading, "-" * len(heading), "", f".. needuml:: {heading} classes", ""]
        lines += ["   " + ln for ln in block(classes, rels)]
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
