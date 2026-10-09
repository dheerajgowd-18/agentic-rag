import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import uuid
import json
import logfire
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim, get_active_embedding_metadata
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client lazily
_qdrant_client = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return _qdrant_client


@dataclass
class IngestionReport:
    files_discovered: int = 0
    files_processed_successfully: int = 0
    empty_documents: int = 0
    unsupported_files: int = 0
    failed_documents: int = 0
    chunks_generated: int = 0
    indexed_vectors: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "success"  # "success", "partial_success", "failed"

    def finalize(self):
        if self.failed_documents > 0:
            if self.files_processed_successfully > 0:
                self.status = "partial_success"
            else:
                self.status = "failed"
        elif self.files_discovered == 0:
            self.status = "empty"
        else:
            self.status = "success"

    def summary(self) -> str:
        return (
            f"--- Ingestion Report [{self.status.upper()}] ---\n"
            f"  Files Discovered:   {self.files_discovered}\n"
            f"  Processed Success:  {self.files_processed_successfully}\n"
            f"  Empty Documents:    {self.empty_documents}\n"
            f"  Unsupported Files:  {self.unsupported_files}\n"
            f"  Failed Documents:   {self.failed_documents}\n"
            f"  Chunks Generated:   {self.chunks_generated}\n"
            f"  Vectors Indexed:    {self.indexed_vectors}\n"
            f"  Total Errors:       {len(self.errors)}"
        )


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(
    file_path: str,
    filename: str,
    source_type: str,
    report: IngestionReport
) -> bool:
    """Parse → chunk → save locally → embed → index in Qdrant with detailed tracking."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext in ("txt", "md"):
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                report.unsupported_files += 1
                return False

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} (document may be empty or image-only).")
                report.empty_documents += 1
                return False

            # Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                report.empty_documents += 1
                return False

            report.chunks_generated += len(chunks)

            # Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "char_count": len(full_text),
            }
            save_processed_locally(processed_data, source_type, filename)

            # Embed and index in Qdrant
            client = get_qdrant_client()
            embed_meta = get_active_embedding_metadata()

            with logfire.span("Vectorizing & Indexing", chunks=len(chunks)):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "embedding_provider": embed_meta.get("provider"),
                            "embedding_model": embed_meta.get("model_name"),
                        },
                    )
                    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings))
                ]

                client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )
                report.indexed_vectors += len(points)
                report.files_processed_successfully += 1
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")
                return True

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            report.failed_documents += 1
            report.errors.append({
                "file": filename,
                "error": str(e),
                "type": type(e).__name__
            })
            return False


def process_directory(dir_path: str, source_type: str, report: IngestionReport):
    """Process every file in a directory and populate report."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in sorted(os.listdir(dir_path)) if os.path.isfile(os.path.join(dir_path, f))]
        report.files_discovered += len(files)
        logfire.info(f"Found {len(files)} files in {dir_path}.")

        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type, report)


def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: str = None,
    wipe: bool = False
) -> IngestionReport:
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Preserves distinction between true_data and noisy_data.
    Returns an IngestionReport detailing counts and errors.
    """
    report = IngestionReport()

    with logfire.span("Universal Ingestion Started", base_directory=base_dir):
        client = get_qdrant_client()

        # Wipe collection if explicitly requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if client.collection_exists(settings.QDRANT_COLLECTION):
                    client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection if it does not exist
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(f"Created collection '{settings.QDRANT_COLLECTION}' ({dim}-dim, Cosine).")

        subdirs = [
            d for d in sorted(os.listdir(base_dir))
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type, report)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type, report)

        report.finalize()
        logfire.info(f"Universal ingestion finished. Status: {report.status}")
        return report


if __name__ == "__main__":
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    ingest_report = run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    print("\n" + ingest_report.summary())

    if ingest_report.status == "failed" or ingest_report.failed_documents > 0:
        sys.exit(1)
    sys.exit(0)
