"""Script per l'ingestion incrementale e sicura dei PDF nel database vettoriale Chroma.
Non sovrascrive né cancella i chunk già presenti nel database.
"""

import os
import re
import sys
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chroma_dao import ChromaDAO
from app.chunks import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_CHUNK_LENGTH,
    _read_pdf_local,
    _source_id,
)
from app.embeddings import create_embeddings
from app.pipeline import COLLECTION_NAME, load_env

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# NOTA CRUCIALE: Usiamo percorsi relativi per evitare problemi di encoding
# dei caratteri accentati (es. 'Università') nel driver Rust/HNSW di Chroma su Windows.
DB_PATH = Path("chroma_db_bilanci")
PDF_DIR = Path("app/data")
BATCH_SIZE = 100


def run_incremental_ingestion():
    load_env()
    print("=" * 60)
    print("AVVIO INGESTION INCREMENTALE IN CHROMA")
    print(f"Directory PDF: {PDF_DIR}")
    print(f"Database Chroma: {DB_PATH}")
    print("=" * 60)

    if not PDF_DIR.is_dir():
        print(f"Errore: la cartella {PDF_DIR} non esiste.")
        return

    embeddings = create_embeddings(local_files_only=True)
    vector_db = ChromaDAO(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # 1. Recupero metadati e ID esistenti
    print("\n[1/3] Verifica dei chunk e dei file già presenti nel database...")
    coll = vector_db.client._collection
    total_in_db = coll.count()
    print(f"Chunk attualmente memorizzati nel database: {total_in_db:,}")

    existing_sources = set()
    existing_ids = set()

    batch_get = 3000
    for offset in range(0, total_in_db, batch_get):
        res = coll.get(limit=batch_get, offset=offset, include=["metadatas"])
        ids = res.get("ids") or []
        existing_ids.update(ids)
        metas = res.get("metadatas") or []
        for m in metas:
            if m and "source" in m:
                existing_sources.add(m["source"])

    print(f"Fonti (PDF) già indicizzate nel database: {len(existing_sources)}")

    # 2. Rilevamento nuovi file
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    pending_pdfs = [p for p in all_pdfs if p.name not in existing_sources]

    print(f"\n[2/3] Totale PDF trovati in app/data: {len(all_pdfs)}")
    print(f"Nuovi PDF da elaborare: {len(pending_pdfs)}")

    if not pending_pdfs:
        print("\n✅ Tutti i PDF sono già presenti nell'indice Chroma! Nessuna operazione necessaria.")
        return

    # 3. Elaborazione e inserimento incrementale
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    print(f"\n[3/3] Inizio estrazione testo, chunking ed embedding per {len(pending_pdfs)} file...")
    start_total_time = time.time()
    total_new_chunks = 0

    for file_index, pdf_path in enumerate(pending_pdfs, start=1):
        file_start = time.time()
        try:
            record = _read_pdf_local(pdf_path)
        except Exception as exc:
            print(f"[{file_index}/{len(pending_pdfs)}] {pdf_path.name}: Errore lettura: {exc}")
            continue

        if not record or not record.get("text"):
            print(f"[{file_index}/{len(pending_pdfs)}] {pdf_path.name}: Nessun testo estratto, saltato.")
            continue

        clean_text = re.sub(r"\r", "", record["text"])
        clean_text = re.sub(r"[ \t]+", " ", clean_text)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
        texts = [
            text.strip()
            for text in splitter.split_text(clean_text)
            if len(text.strip()) >= MIN_CHUNK_LENGTH
        ]
        source_id = _source_id(record)

        file_chunks: list[Document] = []
        file_chunk_ids: list[str] = []

        for index, text in enumerate(texts):
            chunk_id = f"{source_id}-chunk-{index}"
            if chunk_id in existing_ids:
                continue

            metadata = {
                "company": record["company"],
                "fiscal_year": int(record["fiscal_year"]),
                "source": record["source"],
                "document_type": "10-K",
                "year_source": record["year_source"],
                "chunk_id": chunk_id,
                "chunk_index": index,
                "chunk_total": len(texts),
            }
            if index > 0:
                metadata["chunk_prev_id"] = f"{source_id}-chunk-{index - 1}"
            if index + 1 < len(texts):
                metadata["chunk_next_id"] = f"{source_id}-chunk-{index + 1}"

            file_chunks.append(Document(page_content=text, metadata=metadata))
            file_chunk_ids.append(chunk_id)

        if not file_chunks:
            print(f"[{file_index}/{len(pending_pdfs)}] {pdf_path.name}: 0 nuovi chunk da aggiungere.")
            continue

        # Inserimento a batch in Chroma
        for b_start in range(0, len(file_chunks), BATCH_SIZE):
            b_docs = file_chunks[b_start : b_start + BATCH_SIZE]
            b_ids = file_chunk_ids[b_start : b_start + BATCH_SIZE]
            vector_db.add_documents(documents=b_docs, ids=b_ids)
            existing_ids.update(b_ids)

        total_new_chunks += len(file_chunks)
        elapsed = time.time() - file_start
        print(
            f"[{file_index}/{len(pending_pdfs)}] {pdf_path.name} "
            f"({record['company']}, {record['fiscal_year']}): "
            f"+{len(file_chunks)} chunk in {elapsed:.1f}s"
        )

    total_time = time.time() - start_total_time
    final_count = vector_db.client._collection.count()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETATA CON SUCCESSO!")
    print(f"Nuovi chunk aggiunti: {total_new_chunks:,}")
    print(f"Totale complessivo chunk in Chroma: {final_count:,}")
    print(f"Tempo impiegato: {total_time:.1f} secondi ({total_time / 60:.1f} minuti)")
    print("=" * 60)


if __name__ == "__main__":
    run_incremental_ingestion()
