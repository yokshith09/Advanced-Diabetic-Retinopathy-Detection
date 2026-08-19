import sys
import os
import re
from urllib.parse import urlparse, unquote
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import mm


def md_to_flowables(md_text):
    styles = getSampleStyleSheet()
    normal = styles['BodyText']
    heading1 = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=18, leading=22)
    heading2 = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=14, leading=18)
    flowables = []
    lines = md_text.splitlines()
    buffer = []

    def flush_buffer():
        nonlocal buffer
        if not buffer:
            return
        paragraph_text = '<br/>'.join(buffer)
        flowables.append(Paragraph(paragraph_text, normal))
        flowables.append(Spacer(1, 4*mm))
        buffer = []

    img_pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')

    for line in lines:
        line = line.rstrip()
        if not line:
            flush_buffer()
            continue
        # image handling: flush current paragraph and insert image flowable
        m = img_pattern.match(line)
        if m:
            flush_buffer()
            img_url = m.group(1).strip()
            # handle file:// paths and url-decoding
            if img_url.startswith('file://'):
                parsed = urlparse(img_url)
                path = unquote(parsed.path)
                # On Windows paths may start with /e:/ or /C:/ - strip leading slash if present
                if re.match(r'^/[A-Za-z]:', path):
                    path = path[1:]
            else:
                path = unquote(img_url)
            # resolve relative to workspace if necessary
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            if os.path.exists(path):
                # dynamically compute image size preserving aspect ratio
                from PIL import Image as PILImage
                try:
                    with PILImage.open(path) as pil_img:
                        w_orig, h_orig = pil_img.size
                    aspect = h_orig / w_orig
                    # Max width in points (ReportLab units). 160mm is safe for A4
                    max_width = 160 * mm
                    max_height = 200 * mm
                    
                    width = max_width
                    height = width * aspect
                    if height > max_height:
                        height = max_height
                        width = height / aspect
                        
                    img = RLImage(path, width=width, height=height)
                    flowables.append(img)
                    flowables.append(Spacer(1, 4*mm))
                except Exception as e:
                    flowables.append(Paragraph(f'Error loading image {path}: {str(e)}', normal))
                    flowables.append(Spacer(1, 4*mm))
            else:
                flowables.append(Paragraph(f'Image not found: {path}', normal))
                flowables.append(Spacer(1, 4*mm))
            continue

        if line.startswith('# '):
            flush_buffer()
            flowables.append(Paragraph(line[2:].strip(), heading1))
            flowables.append(Spacer(1, 2*mm))
        elif line.startswith('## '):
            flush_buffer()
            flowables.append(Paragraph(line[3:].strip(), heading2))
            flowables.append(Spacer(1, 2*mm))
        else:
            # simple handling for bold/italic **text** and *text*
            # naive bold handling: convert **bold** to <b>bold</b>
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            # naive: we won't fully parse markdown; keep plain text
            buffer.append(line)
    flush_buffer()
    return flowables


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python md_to_pdf_simple.py input.md output.pdf')
        sys.exit(2)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    with open(input_path, 'r', encoding='utf-8') as f:
        md = f.read()
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    flowables = md_to_flowables(md)
    doc.build(flowables)
    print('Wrote', output_path)
