import os
import logfire
import docx as python_docx
from pptx import Presentation

def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) using native python-docx and python-pptx.
    Direct parsing avoids unstructured layout model hangs on Windows.
    """
    with logfire.span("📄 Office Document Parsing", filename=file_path):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".docx":
                doc = python_docx.Document(file_path)
                full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            elif ext == ".pptx":
                prs = Presentation(file_path)
                texts = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            texts.append(shape.text.strip())
                full_text = "\n".join(texts)
            else:
                full_text = ""

            if not full_text.strip():
                logfire.warning(f"⚠️ Empty text extracted for {file_path}")
            else:
                logfire.info(f"✅ Successfully parsed {len(full_text)} characters")

            return full_text
        except Exception as e:
            logfire.error(f"❌ Office Parse Failed: {e}")
            raise e
