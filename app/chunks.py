"""Lettura dei PDF (da locale o da AWS S3) e creazione dei chunk con i relativi metadati."""

import io
import os
import re
import unicodedata
from pathlib import Path

import boto3
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 60

TICKER_TO_COMPANY = {
    "aapl": "apple",
    "amzn": "amazon",
    "meta": "meta",
    "fb": "meta",
    "msft": "microsoft",
    "goog": "alphabet",
    "googl": "alphabet",
    "tsla": "tesla",
    "intc": "intel",
    "ibm": "international business machines",
    "csco": "cisco systems",
    "pypl": "paypal",
    "nflx": "netflix",
    "nvda": "nvidia",
    "amd": "amd",
    "orcl": "oracle",
    "crm": "salesforce",
    "salesforce": "salesforce",
    "qcom": "qualcomm",
    "adbe": "adobe",
    "adobe": "adobe",
    "avgo": "broadcom",
    "uber": "uber",
    "dis": "disney",
    "wmt": "walmart",
    "walmart": "walmart",
    "wallmart": "walmart",
    "v": "visa",
    "abnb": "airbnb",
    "airbnb": "airbnb",
    "dell": "dell",
    "dell technologies": "dell",
    "ko": "coca cola",
    "coca-cola": "coca cola",
    "coca cola": "coca cola",
    "pep": "pepsico",
    "pepsi": "pepsico",
    "pepsico": "pepsico",
    "nke": "nike",
    "nike": "nike",
    "cost": "costco",
    "costco": "costco",
    "ma": "mastercard",
    "mastercard": "mastercard",
    "jpm": "jpmorgan chase",
    "jpmorgan": "jpmorgan chase",
    "mcd": "mcdonalds",
    "mcdonald's": "mcdonalds",
    "f": "ford",
    "ford": "ford",
    "pfe": "pfizer",
    "pfizer": "pfizer",
    "sbux": "starbucks",
    "starbucks": "starbucks",
    "now": "servicenow",
    "servicenow": "servicenow",
    "spot": "spotify",
    "spotify": "spotify",
    "tgt": "target",
    "target": "target",
    "bac": "bank of america",
    "bank of america": "bank of america",
    "bofa": "bank of america",
    "ccl": "coca cola",
}


def normalize_company(value: str) -> str:
    """Normalizza ticker e ragioni sociali in un nome confrontabile."""
    normalized = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(
        r"\b(incorporated|inc|corp|corporation|holdings|company|co|ltd)\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return TICKER_TO_COMPANY.get(normalized, normalized)


def company_and_year(file_name: str) -> tuple[str, int]:
    """Ricava azienda e anno fiscale dal nome del PDF."""
    stem = Path(file_name).stem
    year_pattern = r"(?<!\d)((?:19|20)\d{2})(?!\d)"
    years = re.findall(year_pattern, stem)
    fiscal_year = int(years[-1]) if years else 0
    company_name = re.sub(r"^(?:NASDAQ|NYSE|SEC|ASX)[ _-]+", "", stem, flags=re.IGNORECASE)
    company_name = re.sub(year_pattern, " ", company_name)
    company_name = re.sub(
        r"(?:annual[ _-]?report|10[ _-]?k|fiscal[ _-]?year|annual|report|fiscal|year)",
        " ",
        company_name,
        flags=re.IGNORECASE,
    )
    return normalize_company(company_name.strip(" _-") or stem), fiscal_year


def _year_from_text(text: str) -> int:
    match = re.search(
        r"(?:fiscal\s+year|year)\s+(?:ended|ending)[^0-9]{0,80}((?:19|20)\d{2})",
        text,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else 0


def _read_pdf_from_stream(file_stream, file_name: str) -> dict | None:
    """Legge un PDF da uno stream binario (es. S3)."""
    company, fiscal_year = company_and_year(file_name)
    try:
        reader = PdfReader(file_stream)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[PAGE {page_number}]\n{page_text}")
        text = "\n\n".join(pages)
        if not text.strip():
            print(f"PDF senza testo estraibile: {file_name}")
            return None

        year_source = "filename" if fiscal_year else "unknown"
        if not fiscal_year:
            fiscal_year = _year_from_text(text)
            year_source = "pdf_text" if fiscal_year else "unknown"

        print(
            f"Caricato da S3: {file_name} -> azienda={company}, "
            f"anno fiscale={fiscal_year}, pagine={len(reader.pages)}"
        )
        return {
            "source": Path(file_name).name,
            "company": company,
            "fiscal_year": fiscal_year,
            "year_source": year_source,
            "text": text,
        }
    except Exception as exc:
        print(f"Errore nella lettura da S3 di {file_name}: {exc}")
        return None


def _read_pdf_local(pdf_file: Path) -> dict | None:
    """Legge un PDF locale e restituisce testo e metadati essenziali."""
    company, fiscal_year = company_and_year(pdf_file.name)
    try:
        reader = PdfReader(str(pdf_file))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[PAGE {page_number}]\n{page_text}")
        text = "\n\n".join(pages)
        if not text.strip():
            print(f"PDF senza testo estraibile: {pdf_file.name}")
            return None

        year_source = "filename" if fiscal_year else "unknown"
        if not fiscal_year:
            fiscal_year = _year_from_text(text)
            year_source = "pdf_text" if fiscal_year else "unknown"

        print(
            f"Caricato locale: {pdf_file.name} -> azienda={company}, "
            f"anno fiscale={fiscal_year}, pagine={len(reader.pages)}"
        )
        return {
            "source": pdf_file.name,
            "company": company,
            "fiscal_year": fiscal_year,
            "year_source": year_source,
            "text": text,
        }
    except Exception as exc:
        print(f"Errore nella lettura di {pdf_file.name}: {exc}")
        return None


def load_pdfs(pdf_folder: str | Path) -> list[dict]:
    """Legge i PDF da un Bucket S3 (se inizia con s3://) o da cartella locale."""
    records = []
    target = str(pdf_folder)
    
    if target.startswith("s3://"):
        # Parsing del bucket S3 (es. s3://mio-bucket-10k)
        bucket_name = target.replace("s3://", "").strip("/")
        s3_client = boto3.client("s3")
        
        print(f"Connessione al Bucket S3: {bucket_name}")
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if "Contents" not in response:
            raise FileNotFoundError(f"Nessun file trovato nel bucket S3: {bucket_name}")
            
        for obj in response["Contents"]:
            file_key = obj["Key"]
            if file_key.lower().endswith(".pdf"):
                file_obj = io.BytesIO()
                s3_client.download_fileobj(bucket_name, file_key, file_obj)
                file_obj.seek(0)
                record = _read_pdf_from_stream(file_obj, file_key)
                if record:
                    records.append(record)
    else:
        folder = Path(pdf_folder)
        if not folder.exists():
            raise FileNotFoundError(f"Cartella PDF non trovata: {folder}")

        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"Nessun PDF trovato in: {folder}")

        records = [record for file in pdf_files if (record := _read_pdf_local(file))]

    if not records:
        raise ValueError("Nessun PDF leggibile è stato trovato.")
    print(f"\nPDF caricati: {len(records)}")
    return records


def _source_id(record: dict) -> str:
    value = f"{record['company']}-{record['fiscal_year']}-{record['source']}".lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def chunk_pdfs(pdf_folder: str | Path) -> list[Document]:
    """Legge i PDF (S3 o locali), divide il testo e collega ogni chunk ai suoi vicini."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks = []

    for record in load_pdfs(pdf_folder):
        clean_text = re.sub(r"\r", "", record["text"])
        clean_text = re.sub(r"[ \t]+", " ", clean_text)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
        texts = [
            text.strip()
            for text in splitter.split_text(clean_text)
            if len(text.strip()) >= MIN_CHUNK_LENGTH
        ]
        source_id = _source_id(record)

        for index, text in enumerate(texts):
            metadata = {
                "company": record["company"],
                "fiscal_year": int(record["fiscal_year"]),
                "source": record["source"],
                "document_type": "10-K",
                "year_source": record["year_source"],
                "chunk_id": f"{source_id}-chunk-{index}",
                "chunk_index": index,
                "chunk_total": len(texts),
            }
            if index > 0:
                metadata["chunk_prev_id"] = f"{source_id}-chunk-{index - 1}"
            if index + 1 < len(texts):
                metadata["chunk_next_id"] = f"{source_id}-chunk-{index + 1}"
            chunks.append(Document(page_content=text, metadata=metadata))

        print(
            f"{record['source']}: {len(texts)} chunk | "
            f"azienda={record['company']} | anno fiscale={record['fiscal_year']}"
        )

    if not chunks:
        raise ValueError("Il chunking non ha prodotto alcun chunk utilizzabile.")
    print(f"\nChunk totali: {len(chunks)}")
    return chunks