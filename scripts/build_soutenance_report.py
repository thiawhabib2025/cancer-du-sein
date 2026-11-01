"""Build a formatted Word copy of the fact-checked Markdown report."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deliverables" / "MEMOIRE_SOUTENANCE_VERSION_2026-11-01.md"
DESTINATION = ROOT / "deliverables" / "MEMOIRE_SOUTENANCE_VERSION_2026-11-01.docx"


def plain_text(markdown: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", markdown)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    return re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def set_cell_shading(cell, fill: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run._r.addnext(field)


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(38, 50, 66)
    normal.paragraph_format.space_after = Pt(6)

    for style_name, size, color in (
        ("Title", 24, "17365D"),
        ("Heading 1", 17, "17365D"),
        ("Heading 2", 13, "24527A"),
        ("Heading 3", 11, "3E6A8E"),
    ):
        style = document.styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)

    header = section.header.paragraphs[0]
    header.text = "EPT · M1 ILIA  |  Fusion multimodale pour la détection du cancer du sein"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor(94, 109, 128)
    add_page_number(section.footer.paragraphs[0])


def add_markdown_table(document: Document, lines: list[str]) -> None:
    rows = [table_cells(line) for line in lines]
    if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
        rows.pop(1)
    columns = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=columns)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row_index, values in enumerate(rows):
        for column_index in range(columns):
            cell = table.cell(row_index, column_index)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = plain_text(values[column_index]) if column_index < len(values) else ""
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                for run in paragraph.runs:
                    run.font.size = Pt(8.5)
            if row_index == 0:
                set_cell_shading(cell, "DCEAF5")
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True


def build_report() -> Path:
    source_lines = SOURCE.read_text(encoding="utf-8").splitlines()
    document = Document()
    configure_document(document)

    index = 0
    while index < len(source_lines):
        stripped = source_lines[index].strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while index < len(source_lines) and source_lines[index].strip().startswith("|"):
                table_lines.append(source_lines[index].strip())
                index += 1
            add_markdown_table(document, table_lines)
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            paragraph = document.add_paragraph(style="Title" if level == 1 else f"Heading {level}")
            paragraph.add_run(plain_text(heading.group(2)))
            index += 1
            continue

        if stripped.startswith("```"):
            index += 1
            while index < len(source_lines) and not source_lines[index].strip().startswith("```"):
                paragraph = document.add_paragraph()
                run = paragraph.add_run(source_lines[index])
                run.font.name = "Consolas"
                run.font.size = Pt(8)
                index += 1
            index += 1
            continue

        if stripped.startswith(">"):
            paragraph = document.add_paragraph(style="Quote")
            paragraph.add_run(plain_text(stripped.lstrip("> ")))
            index += 1
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if bullet or numbered:
            paragraph = document.add_paragraph(style="List Bullet" if bullet else "List Number")
            paragraph.add_run(plain_text((bullet or numbered).group(1)))
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(source_lines):
            next_line = source_lines[index].strip()
            if (
                not next_line
                or next_line.startswith(("#", "|", ">", "```"))
                or re.match(r"^[-*]\s+|^\d+[.)]\s+", next_line)
            ):
                break
            paragraph_lines.append(next_line)
            index += 1
        document.add_paragraph(plain_text(" ".join(paragraph_lines)))

    document.save(DESTINATION)
    return DESTINATION


if __name__ == "__main__":
    print(build_report())
