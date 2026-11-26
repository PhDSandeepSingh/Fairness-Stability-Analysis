# Runs the data/definition/run cells of the notebook (skips the smoke test) until MAX_SECONDS is reached; resumable.
import nbformat, sys, re
nb = nbformat.read("Paper1_Fairness_Audit.ipynb", as_version=4)
ns = {"__name__": "__main__"}
for c in nb.cells:
    if c.cell_type != "code" or "smoke test" in c.source: continue
    src = "\n".join((l[:len(l)-len(l.lstrip())] + "pass") if l.strip().startswith("%pip") else l for l in c.source.splitlines())
    if src.startswith("# Headline") or src.startswith("import matplotlib") or "def holm" in src: break
    exec(compile(src, "<cell>", "exec"), ns)
print("rows so far:", len(ns["R"]), "| splits per dataset:", ns["R"].groupby("dataset").rep.nunique().to_dict())
