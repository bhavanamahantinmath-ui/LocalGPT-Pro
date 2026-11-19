"""
Chat export module for LocalGPT Pro.
Exports full conversation threads to formatted Markdown (.md) or publication-quality PDF (.pdf).
"""

import os
import io
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib import colors

def export_to_markdown(chat_title: str, messages: List[Dict[str, Any]]) -> str:
    """
    Converts a chat conversation into clean, structured GitHub Flavored Markdown.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {chat_title}",
        f"*Exported from LocalGPT Pro on {now_str}*",
        "",
        "---",
        ""
    ]

    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        sources = m.get("sources")
        ts = m.get("created_at", "")

        prefix = "👤 **User**" if role.lower() == "user" else "🤖 **LocalGPT Pro**"
        lines.append(f"### {prefix}")
        if ts:
            lines.append(f"*Sent: {ts}*")
        lines.append("")
        lines.append(content)
        lines.append("")

        if sources:
            lines.append("**📚 Grounded Sources:**")
            for s in sources:
                lines.append(f"- **{s.get('source', 'Document')}** (Page {s.get('page', 1)}) — *Relevance: {s.get('score', 'N/A')}*")
                lines.append(f"  > \"{s.get('snippet', '')}\"")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)

def export_to_pdf(
    chat_title: str,
    messages: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> bytes:
    """
    Compiles conversation history into a styled PDF document using ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer if not output_path else output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=14
    )

    user_header_style = ParagraphStyle(
        "UserHeader",
        parent=styles["Heading3"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#2563eb"),
        spaceBefore=10,
        spaceAfter=4
    )

    ai_header_style = ParagraphStyle(
        "AIHeader",
        parent=styles["Heading3"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#059669"),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "MessageBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=8
    )

    source_style = ParagraphStyle(
        "SourceCitation",
        parent=styles["Italic"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4b5563"),
        leftIndent=15,
        spaceAfter=4
    )

    story = []

    # Title & Metadata
    clean_title = re.sub(r"[^\w\s-]", "", chat_title)
    story.append(Paragraph(f"<b>{clean_title}</b>", title_style))
    now_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"LocalGPT Pro Conversation Archive &bull; Exported on {now_str}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=14))

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        sources = m.get("sources")

        # Clean content HTML entities for ReportLab XML parser
        safe_content = (
            content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

        if role == "user":
            story.append(Paragraph("<b>👤 User</b>", user_header_style))
        else:
            story.append(Paragraph("<b>🤖 LocalGPT Pro</b>", ai_header_style))

        story.append(Paragraph(safe_content, body_style))

        if sources:
            story.append(Spacer(1, 4))
            for s in sources:
                src_txt = f"&bull; <b>{s.get('source', 'Doc')}</b> (p. {s.get('page', 1)}) — {s.get('snippet', '')[:100]}..."
                story.append(Paragraph(src_txt, source_style))

        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#f1f5f9"), spaceAfter=10))

    doc.build(story)

    if output_path:
        with open(output_path, "rb") as f:
            pdf_bytes = f.read()
    else:
        pdf_bytes = buffer.getvalue()
        buffer.close()

    return pdf_bytes

