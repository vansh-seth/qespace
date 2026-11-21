"""Build the Overleaf project and the Word manuscript from main_src.tex.

  python3 build.py            (run inside paper/)
Outputs: overleaf/main.tex, overleaf/references.bib, overleaf/figures/*, QESPACE_manuscript.docx
"""
import re, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import refs

HERE = Path(__file__).parent
FIG = HERE.parent / "qespace_nc" / "figures"
src = "\n".join(l for l in (HERE / "main_src.tex").read_text().split("\n") if not l.startswith("%%"))
display = (HERE / "display_items.tex").read_text()
src = src.replace("@DISPLAY@", display)

# ---------------------------------------------------------------- citations
order = []
for m in re.finditer(r"\\cite\{([^}]*)\}", src):
    for k in m.group(1).split(","):
        k = k.strip()
        if k not in refs.R:
            raise SystemExit(f"unknown citation key {k}")
        if k not in order:
            order.append(k)
num = {k: i + 1 for i, k in enumerate(order)}
unused = [k for k in refs.R if k not in num]
print(f"{len(order)} references cited; unused in database: {unused}")


def compress(ns):
    ns = sorted(set(ns)); out = []; i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if j == i else (f"{ns[i]},{ns[j]}" if j == i + 1 else f"{ns[i]}--{ns[j]}"))
        i = j + 1
    return ",".join(out)


# ---------------------------------------------------------------- Overleaf version
PRE = r"""\documentclass[12pt,a4paper]{article}
%% Q-ESPACE manuscript draft -- Nature Communications article structure.
%% Compiles on Overleaf with pdfLaTeX, no extra files needed.
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\IfFileExists{newtxtext.sty}{\usepackage{newtxtext,newtxmath}}{\usepackage{mathptmx}}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{booktabs,array}
\usepackage{setspace}
\usepackage{lineno}
\usepackage[dvipsnames]{xcolor}
\usepackage[super,sort&compress,numbers]{natbib}
\usepackage[colorlinks=true,allcolors=MidnightBlue]{hyperref}
\usepackage{caption}
\captionsetup{labelfont=bf,labelsep=space,font=small,justification=raggedright,singlelinecheck=false}
\renewcommand{\figurename}{Fig.}
\graphicspath{{figures/}}
\newcommand{\todo}[1]{\textcolor{red}{\textbf{[TO COMPLETE: #1]}}}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.6em}
\makeatletter\renewcommand{\@biblabel}[1]{#1.}\makeatother
\begin{document}
\doublespacing
\linenumbers
"""
body = src
title = re.search(r"\\title\{(.*?)\}\n", body, re.S).group(1)
author = re.search(r"\\author\{(.*?)\}\n", body, re.S).group(1)
affil = re.search(r"\\affil\{(.*?)\}\n", body, re.S).group(1)
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1).strip()
main = body[body.index(r"\section*{Introduction}"):]
head = (f"\\begin{{flushleft}}\n{{\\LARGE\\bfseries {title}\\par}}\n\\vspace{{1em}}\n{{\\large {author}\\par}}\n"
        f"\\vspace{{0.5em}}\n{{\\small {affil}\\par}}\n\\end{{flushleft}}\n\n\\section*{{Abstract}}\n{abstract}\n\n")
bib = "\\begin{thebibliography}{99}\n\\setlength{\\itemsep}{0pt}\n" + \
      "".join(f"\\bibitem{{{k}}} {refs.nature(k)}\n" for k in order) + "\\end{thebibliography}\n"
tex = PRE + head + main.replace("@REFERENCES@", bib) + "\n\\end{document}\n"
ov = HERE / "overleaf"; (ov / "figures").mkdir(parents=True, exist_ok=True)
(ov / "main.tex").write_text(tex)
(ov / "references.bib").write_text("% BibTeX export of the reference list (main.tex uses an embedded bibliography).\n"
                                   + "\n".join(refs.bibtex(k) for k in order))
for f in sorted(FIG.glob("fig*")):
    shutil.copy(f, ov / "figures" / f.name)
if (HERE / "supplementary.tex").exists():
    shutil.copy(HERE / "supplementary.tex", ov / "supplementary.tex")

# ---------------------------------------------------------------- Word version
d = main
eqn = {}
def eq_sub(m):
    inner = m.group(1)
    lab = re.search(r"\\label\{([^}]*)\}", inner)
    n = len(eqn) + 1
    if lab:
        eqn[lab.group(1)] = n
    inner = re.sub(r"\\label\{[^}]*\}", "", inner).strip()
    return f"\\[ {inner} \\qquad\\qquad (\\mathrm{{{n}}}) \\]"
d = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eq_sub, d, flags=re.S)
d = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({eqn[m.group(1)]})", d)
d = re.sub(r"\\cite\{([^}]*)\}", lambda m: "\\textsuperscript{" + compress([num[k.strip()] for k in m.group(1).split(",")]).replace("--", "–") + "}", d)
d = re.sub(r"\\todo\{", r"\\textbf{[TO COMPLETE: ", d)          # closing brace gets a bracket below
# close the bracket of every todo: find matching brace
def close_todos(t):
    out = []; i = 0; key = "\\textbf{[TO COMPLETE: "
    while True:
        j = t.find(key, i)
        if j < 0:
            out.append(t[i:]); break
        out.append(t[i:j + len(key)]); k = j + len(key); depth = 1
        while depth:
            depth += {"{": 1, "}": -1}.get(t[k], 0); k += 1
        out.append(t[j + len(key):k - 1] + "]}"); i = k
    return "".join(out)
d = close_todos(d)
reflist = "\\section*{References}\n\\begin{enumerate}\n" + "".join(
    "\\item " + refs.nature(k).replace("--", "–") + "\n" for k in order) + "\\end{enumerate}\n"
d = d.replace("@REFERENCES@", reflist)
d = d.replace("\\includegraphics[width=\\textwidth]{", "\\includegraphics[width=\\textwidth]{" + str(ov / "figures") + "/")
d = re.sub(r"(figures/fig\w+)\}", r"\1.png}", d)
d = d.replace("\\clearpage", "")
head_d = (f"\\section*{{{title}}}\n\n{author}\n\n{affil}\n\n\\section*{{Abstract}}\n{abstract}\n\n")
doc = ("\\documentclass{article}\\usepackage{amsmath,amssymb,graphicx,booktabs}\\begin{document}\n" + head_d + d + "\n\\end{document}\n")
tmp = HERE / "_word.tex"; tmp.write_text(doc)
out = HERE / "QESPACE_manuscript.docx"
subprocess.run(["pandoc", str(tmp), "-f", "latex", "-t", "docx", "-o", str(out), "--reference-doc", str(HERE / "reference.docx")], check=True)
print("equations:", len(eqn), "->", out)
