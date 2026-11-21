"""Post-process the pandoc Word file: manuscript typography, line numbers, highlighted to-dos."""
import sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

path = sys.argv[1]
doc = Document(path)
BYNAME = {s.name.lower(): s for s in doc.styles}
for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Image Caption", "Table Caption", "Captioned Figure"):
    st = BYNAME.get(name.lower())
    if st is None:
        continue
    st.font.name = "Times New Roman"; st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    if name in ("Body Text", "First Paragraph"):
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        st.paragraph_format.space_after = Pt(6)
    if name in ("Image Caption", "Table Caption"):
        st.font.size = Pt(10); st.font.italic = False
for lvl, size in (("Heading 1", 14), ("Heading 2", 12)):
    st = BYNAME[lvl.lower()]
    st.font.name = "Arial"; st.font.size = Pt(size); st.font.bold = True; st.font.color.rgb = RGBColor(0, 0, 0)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    st.paragraph_format.space_before = Pt(18 if lvl == "Heading 1" else 12); st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True

for sec in doc.sections:
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.5)
    ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous")
    sec._sectPr.append(ln)

first = True
for p in doc.paragraphs:
    if first and p.style.name.startswith("Heading"):          # manuscript title
        for r in p.runs:
            r.font.size = Pt(17)
        first = False
    if p.style.name == "Compact":                             # table cells and reference list
        p.paragraph_format.space_after = Pt(3)
    for r in p.runs:
        if "TO COMPLETE" in r.text or (r.bold and r.text.strip().startswith("[")):
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
# highlight continuation runs of a to-do (pandoc splits runs at inline maths)
for p in doc.paragraphs:
    on = False
    for r in p.runs:
        if "[TO COMPLETE" in r.text:
            on = True
        if on:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if on and "]" in r.text and "[TO COMPLETE" not in r.text.split("]")[-1]:
            on = False
for t in doc.tables:
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.5)
doc.core_properties.title = "Q-ESPACE manuscript draft"
doc.core_properties.author = ""
doc.save(path)
print("polished", path)
