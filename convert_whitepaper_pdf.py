#!/usr/bin/env python3
"""Convert whitepaper.md to a professionally formatted PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
import re
import os

BASE_DIR = "/Users/srimanarayana/Research Project I"
OUTPUT_PATH = os.path.join(BASE_DIR, "whitepaper.pdf")

def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'PaperTitle', parent=styles['Title'],
        fontSize=18, leading=22, alignment=TA_CENTER,
        spaceAfter=6, textColor=HexColor('#1a1a2e'),
        fontName='Times-Bold'
    ))
    styles.add(ParagraphStyle(
        'AbstractLabel', parent=styles['Normal'],
        fontSize=11, leading=14, fontName='Times-Bold',
        alignment=TA_CENTER, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        'AbstractBody', parent=styles['Normal'],
        fontSize=10, leading=14, fontName='Times-Italic',
        alignment=TA_JUSTIFY, leftIndent=36, rightIndent=36,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        'SectionHeading', parent=styles['Heading1'],
        fontSize=14, leading=18, fontName='Times-Bold',
        spaceBefore=18, spaceAfter=8,
        textColor=HexColor('#1a1a2e')
    ))
    styles.add(ParagraphStyle(
        'SubHeading', parent=styles['Heading2'],
        fontSize=12, leading=15, fontName='Times-Bold',
        spaceBefore=12, spaceAfter=6,
        textColor=HexColor('#2d2d44')
    ))
    styles.add(ParagraphStyle(
        'BodyText2', parent=styles['Normal'],
        fontSize=10.5, leading=14, fontName='Times-Roman',
        alignment=TA_JUSTIFY, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        'BoldBody', parent=styles['Normal'],
        fontSize=10.5, leading=14, fontName='Times-Bold',
        alignment=TA_LEFT, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        'TableCell', parent=styles['Normal'],
        fontSize=9, leading=11, fontName='Times-Roman',
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        'TableHeader', parent=styles['Normal'],
        fontSize=9, leading=11, fontName='Times-Bold',
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        'Reference', parent=styles['Normal'],
        fontSize=9.5, leading=12, fontName='Times-Roman',
        leftIndent=24, firstLineIndent=-24, spaceAfter=3
    ))
    return styles


def escape_xml(text):
    """Escape XML special characters but preserve our intentional tags."""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text


def format_inline(text):
    """Convert markdown inline formatting to reportlab XML."""
    # Escape XML first
    text = escape_xml(text)

    # Bold+italic
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="9">\1</font>', text)
    # Subscripts like W_QK -> W<sub>QK</sub>
    # Leave as-is for now (plain text is clearer than mangled subscripts)

    return text


def parse_table(lines):
    """Parse markdown table lines into list of lists."""
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith('|') and not re.match(r'^\|[-\s|]+\|$', line):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            rows.append(cells)
    return rows


def build_pdf():
    styles = build_styles()

    with open(os.path.join(BASE_DIR, "whitepaper.md"), 'r') as f:
        md = f.read()

    story = []
    lines = md.split('\n')
    i = 0
    in_abstract = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip horizontal rules
        if stripped == '---':
            if in_abstract:
                in_abstract = False
            i += 1
            continue

        # Title (# heading)
        if stripped.startswith('# ') and not stripped.startswith('## '):
            title_text = stripped[2:].strip()
            # Split long title
            parts = title_text.split(': ', 1)
            if len(parts) == 2:
                story.append(Spacer(1, 0.5*inch))
                story.append(Paragraph(format_inline(parts[0] + ':'), styles['PaperTitle']))
                story.append(Paragraph(format_inline(parts[1]), styles['PaperTitle']))
            else:
                story.append(Spacer(1, 0.5*inch))
                story.append(Paragraph(format_inline(title_text), styles['PaperTitle']))
            story.append(Spacer(1, 0.3*inch))
            i += 1
            continue

        # Abstract
        if stripped == '**Abstract**':
            story.append(Paragraph('Abstract', styles['AbstractLabel']))
            story.append(HRFlowable(width="60%", thickness=0.5, color=colors.grey))
            story.append(Spacer(1, 4))
            in_abstract = True
            i += 1
            # Collect abstract text
            abstract_lines = []
            while i < len(lines) and lines[i].strip() != '---':
                if lines[i].strip():
                    abstract_lines.append(lines[i].strip())
                i += 1
            if abstract_lines:
                abstract_text = ' '.join(abstract_lines)
                story.append(Paragraph(format_inline(abstract_text), styles['AbstractBody']))
            story.append(HRFlowable(width="60%", thickness=0.5, color=colors.grey))
            story.append(Spacer(1, 12))
            continue

        # Section heading (## N. Title)
        if stripped.startswith('## '):
            heading = stripped[3:].strip()
            story.append(Paragraph(format_inline(heading), styles['SectionHeading']))
            i += 1
            continue

        # Subsection heading (### N.N Title)
        if stripped.startswith('### '):
            heading = stripped[4:].strip()
            story.append(Paragraph(format_inline(heading), styles['SubHeading']))
            i += 1
            continue

        # Table detection
        if stripped.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[-\s|]+\|$', lines[i+1].strip()):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1

            rows = parse_table(table_lines)
            if rows:
                # Build reportlab table
                header = rows[0]
                data_rows = rows[1:]

                table_data = [[Paragraph(format_inline(c), styles['TableHeader']) for c in header]]
                for row in data_rows:
                    table_data.append([Paragraph(format_inline(c), styles['TableCell']) for c in row])

                n_cols = len(header)
                col_width = (6.5 * inch) / n_cols

                t = Table(table_data, colWidths=[col_width]*n_cols)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e8e8f0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1a1a2e')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f8f8fc')]),
                ]))
                story.append(Spacer(1, 6))
                story.append(t)
                story.append(Spacer(1, 6))
            continue

        # Bold paragraph labels like **S-Inhibition L9H6:**
        if stripped.startswith('**') and ':' in stripped:
            story.append(Paragraph(format_inline(stripped), styles['BodyText2']))
            i += 1
            continue

        # Regular paragraph - collect consecutive non-empty lines
        if stripped and not stripped.startswith('#'):
            para_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#') and not lines[i].strip().startswith('|') and not lines[i].strip() == '---':
                para_lines.append(lines[i].strip())
                i += 1

            para_text = ' '.join(para_lines)

            # Check if it's a numbered list item (reference)
            if re.match(r'^\d+\.', para_text) and 'References' in ''.join(l for l in lines[:i]):
                story.append(Paragraph(format_inline(para_text), styles['Reference']))
            else:
                story.append(Paragraph(format_inline(para_text), styles['BodyText2']))
            continue

        # Empty line
        i += 1

    # Build the PDF
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=1*inch,
        rightMargin=1*inch,
        title="Attention Heads Are Not Monolithic",
        author="Research Project"
    )

    doc.build(story)
    print(f"PDF written to: {OUTPUT_PATH}")
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024*1024)
    print(f"Size: {size_mb:.2f} MB")


if __name__ == '__main__':
    build_pdf()
