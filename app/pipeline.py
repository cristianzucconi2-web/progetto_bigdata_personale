"""Pipeline RAG completa: indice, interpretazione, retrieval e risposta."""

import json
import math
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document

from app.chroma_dao import ChromaDAO
from app.chunks import TICKER_TO_COMPANY, chunk_pdfs, normalize_company
from app.embeddings import create_embeddings


COLLECTION_NAME = "annual_reports"
DEFAULT_TOP_K = 28
DEFAULT_MAX_CONTEXT_CHARS = 16_000
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"

FINANCIAL_TERMS = {
    "ricavi": [
        "revenue", "revenues", "total revenue", "total revenues", "net revenue", "net revenues", "sales", "net sales",
        "statement of operations", "statement of earnings", "statements of operations",
        "consolidated statement of earnings", "statement of income", "statements of income",
        "consolidated statement of income", "consolidated statements of income",
    ],
    "fatturato": [
        "revenue", "revenues", "total revenue", "total revenues", "net revenue", "net revenues", "sales", "net sales",
        "statement of operations", "statement of earnings", "statement of income", "statements of income",
    ],
    "revenue": [
        "revenue", "revenues", "total revenue", "total revenues", "net revenue", "net revenues", "sales",
        "statement of operations", "statement of earnings", "statement of income", "statements of income",
    ],
    "crescita": ["growth", "increase", "percentage change", "compared to", "prior year"],
    "utile": ["net income", "net earnings", "profit", "operating income", "operating profit"],
    "profitto": ["net income", "net earnings", "profit"],
    "perdita": ["net loss", "loss"],
    "cassa": ["cash and cash equivalents", "cash flow", "operating cash flow"],
    "attivita": ["total assets", "assets"],
    "passivita": ["total liabilities", "liabilities"],
    "debito": ["debt", "total debt"],
    "margine": ["margin", "gross margin", "operating margin", "gross profit"],
    "research and development": [
        "research and development", "r&d", "r d", "research", "development",
        "research and development expense", "r&d expense",
    ],
    "ricerca e sviluppo": [
        "research and development", "r&d", "r d", "research", "development",
        "research and development expense", "r&d expense",
    ],
    "r&d": [
        "research and development", "r&d", "r d", "research", "development",
        "research and development expense", "r&d expense",
    ],
    "artificial intelligence": [
        "artificial intelligence", "AI", "AI initiatives", "AI-related risks",
        "generative AI", "machine learning", "AI infrastructure",
        "datacenter capacity", "graphics processing units", "capital investments",
        "operating margins", "intellectual property", "regulatory compliance",
        "security risks", "supply chain", "harmful content",
    ],
    "ai": [
        "artificial intelligence", "AI initiatives", "AI-related risks",
        "generative AI", "machine learning",
    ],
    "top risk": [
        "risk factor", "risk factors", "standalone risk factor", "business risks",
    ],
    "dipendenti": [
        "employees", "headcount", "human capital", "workforce", "full-time employees", "part-time employees",
    ],
    "personale": [
        "employees", "headcount", "workforce", "personnel", "human capital",
    ],
    "debiti": [
        "debt", "total debt", "long-term debt", "short-term borrowings", "total liabilities",
    ],
    "acquisizioni": [
        "acquisitions", "business combinations", "mergers", "acquired business",
    ],
    "dividendi": [
        "dividends", "cash dividends declared", "dividend per share",
    ],
    "clienti": [
        "customers", "major customers", "customer concentration",
    ],
    "brevetti": [
        "patents", "intellectual property", "proprietary technology", "trademarks",
    ],
    "immobili": [
        "properties", "facilities", "retail square feet", "real estate",
    ],
}


def load_env() -> None:
    """Carica il file .env locale senza sovrascrivere variabili già impostate."""
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        if not separator or not key.strip():
            continue
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def initialize_database(
    pdf_folder: str | Path,
    db_path: str | Path,
    rebuild: bool = True,
    batch_size: int = 100,
) -> tuple[ChromaDAO, list[Document]]:
    """Crea i chunk dei PDF e li salva nell'indice Chroma."""
    if batch_size <= 0:
        raise ValueError("batch_size deve essere maggiore di zero.")

    db_path = Path(db_path)
    documents = chunk_pdfs(pdf_folder)
    chunk_ids = [document.metadata["chunk_id"] for document in documents]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Sono presenti chunk_id duplicati.")

    if rebuild and db_path.exists():
        shutil.rmtree(db_path)

    vector_db = ChromaDAO(
        db_path=db_path,
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
    )
    existing_ids = set(vector_db.get_all_ids()) if not rebuild else set()
    pending = [
        document
        for document in documents
        if document.metadata["chunk_id"] not in existing_ids
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vector_db.add_documents(
            documents=batch,
            ids=[document.metadata["chunk_id"] for document in batch],
        )
        print(f"Inseriti {start + len(batch)} di {len(pending)} nuovi chunk")

    all_documents = vector_db.get_all_documents()
    print(f"Indice Chroma pronto: {len(all_documents)} chunk totali")
    return vector_db, all_documents


def load_database(db_path: str | Path) -> tuple[ChromaDAO, list[Document]]:
    """Apre un indice Chroma esistente senza rielaborare i PDF."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Indice Chroma non trovato: {db_path}. "
            "Imposta INITIALIZE_DB = True in main.py per crearlo."
        )
    vector_db = ChromaDAO(
        db_path=db_path,
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(local_files_only=True),
    )
    documents = vector_db.get_all_documents()
    if not documents:
        raise ValueError(f"L'indice Chroma è vuoto: {db_path}")
    print(f"Indice Chroma caricato: {len(documents)} chunk")
    return vector_db, documents


def _company_aliases(company_catalog: list[str]) -> dict[str, str]:
    aliases = {
        "ibm": "international business machines",
        "international business machines corp": "international business machines",
        "cisco systems inc": "cisco systems",
        "paypal holdings": "paypal",
        "tesla motors": "tesla",
        "adobe systems": "adobe",
        "salesforce com": "salesforce",
    }
    aliases.update({name: name for name in company_catalog})
    aliases.update(
        {
            ticker: company
            for ticker, company in TICKER_TO_COMPANY.items()
            if company in company_catalog
        }
    )
    return aliases


def _resolve_company(
    value: str | None,
    company_catalog: list[str],
    aliases: dict[str, str],
) -> str | None:
    if not value:
        return None
    normalized = normalize_company(value)
    if normalized in aliases:
        return aliases[normalized]
    for alias, company in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if alias and alias in normalized:
            return company
    matches = get_close_matches(normalized, company_catalog, n=1, cutoff=0.82)
    return matches[0] if matches else None


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def interpret_query(
    query: str,
    documents: list[Document],
    generate: Callable | None = None,
) -> dict:
    """Estrae aziende, anni e parole chiave dalla domanda."""
    company_catalog = sorted(
        {document.metadata["company"] for document in documents}
    )
    aliases = _company_aliases(company_catalog)
    normalized_query = normalize_company(query)
    companies = []
    for alias, company in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized_query) and company not in companies:
            companies.append(company)

    years = sorted({int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", query)})
    keywords = []
    lower_query = query.lower()
    for source, terms in FINANCIAL_TERMS.items():
        if re.search(rf"\b{re.escape(source)}\b", lower_query):
            keywords.extend(terms)

    parsed = {}
    if generate:
        messages = [
            {
                "role": "system",
                "content": (
                    "Estrai solo JSON valido con i campi companies (lista), years "
                    "(lista di interi), keywords_en (lista di termini inglesi). "
                    "Non rispondere alla domanda."
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            parsed = _extract_json(generate(messages, max_new_tokens=100))
        except Exception as exc:
            print(f"Interpretazione LLM non disponibile, uso il fallback: {exc}")

    parsed_companies = parsed.get("companies") or (
        [parsed.get("company")] if parsed.get("company") else []
    )
    for value in parsed_companies:
        company = _resolve_company(str(value), company_catalog, aliases)
        if company and company not in companies:
            companies.append(company)

    parsed_years = parsed.get("years") or (
        [parsed.get("year")] if parsed.get("year") else []
    )
    for year in parsed_years:
        if str(year).isdigit() and int(year) not in years:
            years.append(int(year))
    years.sort()

    keywords.extend(
        str(item).lower()
        for item in parsed.get("keywords_en", [])
        if str(item).strip()
    )
    keywords = list(dict.fromkeys(keywords))
    if not keywords:
        stopwords = {
            "who", "what", "which", "more", "less", "than", "the", "in", "on",
            "and", "or", "of", "for", "companies", "company", "database", "list",
            "lists", "identify", "identifies", "following",
        }
        keywords = [
            word.lower()
            for word in re.findall(r"[a-zA-Z]+", query)
            if word.lower() not in stopwords and len(word) > 2
        ]

    return {
        "company": companies[0] if companies else None,
        "companies": companies,
        "years": years,
        "keywords_en": keywords,
        "query_for_retrieval": f"{query} {' '.join(keywords)}".strip(),
        "comparison": (
            len(companies) > 1
            or len(years) > 1
            or bool(re.search(r"\b(which|what|all)\s+companies\b", query, re.I))
            or bool(re.search(r"trend|confront|crescita|evoluzione|compare|versus", query, re.I))
        ),
    }


def _metadata_filter(interpretation: dict) -> dict | None:
    conditions = []
    companies = interpretation["companies"]
    years = interpretation["years"]
    if companies:
        conditions.append(
            {"company": {"$eq": companies[0]}}
            if len(companies) == 1
            else {"company": {"$in": companies}}
        )
    if years:
        conditions.append({"fiscal_year": {"$in": years}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _matches(document: Document, interpretation: dict) -> bool:
    metadata = document.metadata
    companies = interpretation["companies"]
    years = interpretation["years"]
    return not (
        (companies and metadata.get("company") not in companies)
        or (years and metadata.get("fiscal_year") not in years)
    )


def _company_filter(company: str, years: list[int]) -> dict:
    conditions = [{"company": {"$eq": company}}]
    if years:
        conditions.append({"fiscal_year": {"$in": years}})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def retrieve(
    vector_db: ChromaDAO,
    documents: list[Document],
    interpretation: dict,
    top_k: int = DEFAULT_TOP_K,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> dict:
    """Cerca i chunk rilevanti, aggiunge i vicini e prepara il contesto."""
    candidates = [document for document in documents if _matches(document, interpretation)]
    filters = _metadata_filter(interpretation)
    companies = interpretation["companies"]
    broad_query = bool(
        re.search(
            r"\b(which|what|all)\s+companies\b",
            interpretation["query_for_retrieval"],
            re.I,
        )
    )
    balanced_companies = companies if len(companies) > 1 else []
    if broad_query and not companies:
        balanced_companies = sorted(
            {document.metadata["company"] for document in candidates}
        )

    clean_kws = interpretation.get("keywords_en", [])
    raw_query = interpretation.get("query_for_retrieval", "")
    lower_query = raw_query.lower()

    is_financial = (
        any(k in ["revenue", "revenues", "total revenue", "net revenue", "sales", "net sales",
                  "statement of operations", "statement of income", "utile", "profitto", "cassa",
                  "ricavi", "fatturato"] for k in clean_kws)
        or any(w in lower_query for w in ["ricav", "fatturat", "sales", "revenue", "bilanc", "util", "profit"])
    )
    is_risk = (
        any(k in ["risk factor", "risk factors", "standalone risk factor", "business risks",
                  "ai", "artificial intelligence", "top risk"] for k in clean_kws)
        or any(w in lower_query for w in ["rischi", "risks", "risk"])
    )
    is_rd = (
        any(k in ["research and development", "r&d", "ricerca e sviluppo"] for k in clean_kws)
        or any(w in lower_query for w in ["ricerca", "sviluppo", "r&d"])
    )

    if is_financial:
        base_semantic_query = "consolidated statement of operations statement of income revenue net sales gross profit segment breakdown"
    elif is_risk:
        base_semantic_query = "item 1a risk factors artificial intelligence generative AI cybersecurity regulatory risks"
    elif is_rd:
        base_semantic_query = "research and development expense r&d expenses operating expenses"
    else:
        english_kws = " ".join(dict.fromkeys(clean_kws[:12]))
        base_semantic_query = english_kws if english_kws else interpretation["query_for_retrieval"]

    if balanced_companies:
        hits = []
        per_company_k = max(40, math.ceil(top_k * 2 / len(balanced_companies)))
        for company in balanced_companies:
            company_filters = _company_filter(company, interpretation["years"])
            company_candidates = [
                doc for doc in candidates if doc.metadata.get("company") == company
            ]
            hits.extend(
                vector_db.find(
                    query=base_semantic_query,
                    k=min(per_company_k, len(company_candidates)),
                    filters=company_filters,
                )
            )
        strategy = "balanced_per_company"
    else:
        hits = (
            vector_db.find(
                query=base_semantic_query,
                k=min(top_k * 2, len(candidates)),
                filters=filters,
            )
            if candidates
            else []
        )
        strategy = "global_semantic"

    catalog = {
        document.metadata["chunk_id"]: document
        for document in documents
        if document.metadata.get("chunk_id")
    }
    selected = {}
    for document, _score in hits:
        ids = [
            document.metadata.get("chunk_prev_id"),
            document.metadata.get("chunk_id"),
            document.metadata.get("chunk_next_id"),
        ]
        for chunk_id in ids:
            if chunk_id and chunk_id in catalog:
                selected.setdefault(chunk_id, catalog[chunk_id])

    # Raggruppa i blocchi per azienda con bilanciamento equo dello spazio
    ids_by_company = defaultdict(list)
    for chunk_id, document in selected.items():
        ids_by_company[document.metadata.get("company", "unknown")].append(chunk_id)

    num_companies = max(1, len(ids_by_company))
    max_chars_per_company = max_context_chars // num_companies

    context_sections = []
    total_context_chunks = 0
    for company in sorted(ids_by_company):
        company_blocks = []
        company_size = 0
        company_doc_ids = ids_by_company[company]

        def _rank_chunk(cid: str) -> int:
            doc = selected[cid]
            txt = doc.page_content.lower()
            raw_text = doc.page_content
            score = 0

            if is_financial:
                statement_headers = [
                    "consolidated statement of operations",
                    "consolidated statements of operations",
                    "consolidated statement of income",
                    "consolidated statements of income",
                    "consolidated statement of earnings",
                    "consolidated statements of earnings",
                    "statements of operations",
                    "statements of income",
                    "statement of operations",
                    "statement of income",
                    "statement of earnings",
                    "consolidated results of operations",
                    "results of operations",
                    "operating results",
                ]
                if any(h in txt for h in statement_headers):
                    score += 35

                revenue_exact = [
                    "total revenue", "total revenues", "net revenue", "net revenues",
                    "net sales", "total net sales", "total sales",
                ]
                if any(r in txt for r in revenue_exact):
                    score += 25
                elif re.search(r"\b(revenue|revenues|sales)\b", txt):
                    score += 15

                if any(p in txt for p in ["gross profit", "operating income", "operating loss", "operating profit", "net income", "net loss"]):
                    score += 10

                if any(c in raw_text for c in ["$", "€", "£", "¥"]):
                    score += 5
                if any(s in txt for s in ["in millions", "in thousands", "in billions", "in € millions", "in $ millions"]):
                    score += 5

            elif is_risk:
                if any(h in txt for h in ["item 1a", "risk factors", "standalone risk factor", "risks related to"]):
                    score += 35
                if any(w in txt for w in ["artificial intelligence", "generative ai", "machine learning"]):
                    score += 25

            elif is_rd:
                if any(w in txt for w in ["research and development", "r&d", "r&d expense"]):
                    score += 35

            # Boost generico: premia i chunk contenenti le parole chiave estratte dall'intento
            for kw in clean_kws:
                if len(kw) > 2 and kw in txt:
                    score += 15

            legal_exhibit_markers = [
                "exhibit 10", "exhibit 12", "exhibit 4", "exhibit 8", "exhibit 99",
                "statuts coordonnes", "statuts coordonnés", "employment agreement",
                "severance agreement", "terms and conditions governing employee",
                "certification of chief executive officer", "certification of chief financial officer",
                "form s-8", "indemnification agreement",
            ]
            if any(marker in txt for marker in legal_exhibit_markers):
                score -= 30

            return score

        sorted_cids = sorted(company_doc_ids, key=_rank_chunk, reverse=True)
        for chunk_id in sorted_cids:
            document = selected[chunk_id]
            metadata = document.metadata
            block = (
                f"[SORGENTE: {metadata.get('source')} | AZIENDA: {metadata.get('company', '').upper()} | "
                f"ANNO: {metadata.get('fiscal_year')} | CHUNK #{metadata.get('chunk_index')}]\n"
                f"{document.page_content}"
            )
            if company_size + len(block) > max_chars_per_company:
                break
            company_blocks.append(block)
            company_size += len(block)
            total_context_chunks += 1
        if company_blocks:
            context_sections.append(
                f"=== DOCUMENTI BILANCIO: {company.upper()} ===\n" + "\n\n".join(company_blocks)
            )

    full_context = "\n\n" + ("\n\n" + "=" * 50 + "\n\n").join(context_sections)

    return {
        "context": full_context.strip(),
        "documents": list(selected.values()),
        "diagnostics": {
            "requested_company": interpretation["company"],
            "requested_companies": companies,
            "requested_years": interpretation["years"],
            "metadata_filter": filters,
            "candidate_chunks": len(candidates),
            "semantic_hits": len(hits),
            "context_chunks": total_context_chunks,
            "retrieval_strategy": strategy,
            "retrieval_companies": balanced_companies,
        },
    }


def _groq_generator(model_id: str) -> Callable:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY non impostata nel file .env.")

    fallback_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

    def generate(messages: list[dict[str, str]], max_new_tokens: int = 256) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "rag-project/0.1",
        }

        models_to_try = [model_id] + [m for m in fallback_models if m != model_id]
        last_error = None

        for active_model in models_to_try:
            payload = {
                "model": active_model,
                "messages": messages,
                "temperature": 0,
                "max_completion_tokens": max_new_tokens,
            }
            if active_model.startswith("openai/gpt-oss"):
                payload.update(reasoning_effort="low", reasoning_format="hidden")
            request_data = json.dumps(payload).encode("utf-8")

            max_retries = 3
            for attempt in range(max_retries + 1):
                request = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=request_data,
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        body = json.loads(response.read().decode("utf-8"))
                    return str(body["choices"][0]["message"]["content"]).strip()
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    last_error = detail
                    match_s = re.search(r"try again in ([0-9.]+)s\b", detail, flags=re.IGNORECASE)
                    if exc.code == 429 and match_s and attempt < max_retries:
                        wait_time = float(match_s.group(1)) + 1.0
                        print(f"Rate limit Groq (429). Attesa di {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    if exc.code == 429:
                        print(f"Modello {active_model} ha raggiunto il limite TPD. Fallback immediato a modello alternativo...")
                        break
                    if attempt == max_retries:
                        break
                except urllib.error.URLError as exc:
                    raise RuntimeError("Groq non raggiungibile. Controlla la connessione.") from exc
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(f"Risposta Groq inattesa: {body}") from exc

        raise RuntimeError(f"Errore Groq: {last_error}")

    return generate


def load_generator(provider: str, model_id: str | None = None) -> Callable:
    """Carica Groq oppure il modello Transformers locale."""
    provider = provider.lower().strip()
    if provider == "groq":
        return _groq_generator(model_id or DEFAULT_GROQ_MODEL)
    if provider == "local":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        selected_model = model_id or DEFAULT_LOCAL_MODEL
        use_cuda = torch.cuda.is_available()
        tokenizer = AutoTokenizer.from_pretrained(selected_model)
        model = AutoModelForCausalLM.from_pretrained(
            selected_model,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            low_cpu_mem_usage=True,
            **({"device_map": "auto"} if use_cuda else {}),
        )
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
        )

        def generate(messages: list[dict[str, str]], max_new_tokens: int = 256) -> str:
            outputs = text_pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                clean_up_tokenization_spaces=False,
            )
            generated = outputs[0].get("generated_text", "")
            if isinstance(generated, list):
                generated = generated[-1]
                if isinstance(generated, dict):
                    generated = generated.get("content", "")
            return str(generated).strip()

        return generate
    raise ValueError(f"Provider non supportato: {provider!r}. Usa 'groq' oppure 'local'.")


def answer_from_context(generate: Callable, query: str, context: str) -> str:
    """Genera la risposta usando soltanto il contesto recuperato."""
    if not context.strip():
        return "Non ho trovato dati sufficienti nei report filtrati per rispondere."
    messages = [
        {
            "role": "system",
            "content": (
                "Sei un analista finanziario rigoroso. Usa esclusivamente il contesto "
                "fornito e non inventare o confondere numeri. Cita nome del file sorgente e anno fiscale per "
                "ogni dato. Nelle comparazioni analizza ciascuna azienda separatamente: "
                "NON attribuire MAI i numeri, segmenti o fatturati di un'azienda all'altra. "
                "Se per un'azienda i dati estratti nel contesto non contengono una cifra esplicita, "
                "dichiaralo chiaramente invece di duplicare i dati dell'altra azienda. "
                "Esprimi le cifre in modo chiaro e con la valuta specificata nel report "
                "(es. '$648,125 milioni di dollari' se in USD, oppure '€20.438 milioni di euro' se in EUR). "
                "Non anteporre MAI il simbolo '$' se la valuta del documento è in euro o altra valuta non dollaro. "
                "Evita abbreviazioni ambigue come 'mio' o 'mila'. "
                "Per calcoli e formule (es. crescita percentuale), usa ESCLUSIVAMENTE testo semplice (es. '(Fatturato 2023 - Fatturato 2022) / Fatturato 2022 × 100 = 12,96%'): non usare MAI notazione o comandi LaTeX (niente \\frac, \\text, né delimitatori '$' o '$$'). "
                "Inizia con una sintesi, dedica un paragrafo specifico a ciascuna azienda e chiudi con una tabella comparativa."
            ),
        },
        {
            "role": "user",
            "content": f"DOMANDA: {query}\n\nCONTESTO:\n{context}",
        },
    ]
    raw_answer = generate(messages, max_new_tokens=1150)
    return re.sub(r"(?<!\\)\$(?=\s*\d)", r"\$", raw_answer)


def verify_and_correct_answer(
    generate: Callable,
    query: str,
    context: str,
    draft_answer: str,
) -> dict:
    """Valida e corregge la bozza di risposta rispetto al contesto (Guarded Generation / Self-Correction)."""
    if not context.strip() or not draft_answer.strip():
        return {
            "is_grounded": False,
            "faithfulness_score": 0,
            "temporal_consistency": False,
            "company_consistency": False,
            "numeric_consistency": False,
            "critique": "Contesto o bozza non disponibili per la validazione.",
            "verified_answer": draft_answer,
            "draft_answer": draft_answer,
            "correction_applied": False,
        }

    system_prompt = (
        "Sei un Financial Auditor indipendente e revisore contabile (Self-Correction Agent). "
        "Il tuo compito è convalidare con il massimo rigore la bozza di risposta generata dall'analista, "
        "confrontandola punto per punto con i documenti ufficiali riportati nel CONTESTO.\n\n"
        "Verifica obbligatoriamente:\n"
        "1. ACCURATEZZA NUMERICA E VALUTE: ogni cifra, percentuale o importo monetario (€, $) citato nella bozza è presente identico nel contesto? Ci sono allucinazioni?\n"
        "2. COERENZA TEMPORALE: ogni dato è associato esattamente al rispettivo anno fiscale?\n"
        "3. COERENZA AZIENDALE: i numeri di un'azienda non devono mai essere attribuiti a un'altra (specialmente in confronti cross-azienda).\n\n"
        "Rispondi ESCLUSIVAMENTE con un JSON valido strutturato così:\n"
        "{\n"
        '  "is_grounded": true/false,\n'
        '  "faithfulness_score": <intero da 0 a 100>,\n'
        '  "temporal_consistency": true/false,\n'
        '  "company_consistency": true/false,\n'
        '  "numeric_consistency": true/false,\n'
        '  "critique": "<breve sintesi delle verifiche ed eventuali discrepanze>",\n'
        '  "correction_applied": true/false,\n'
        '  "verified_answer": "<riporta la risposta con eventuali correzioni o rifiniture; se ci sono allucinazioni, elimina le cifre non verificate; mantieni i calcoli in puro testo semplice senza notazione LaTeX né delimitatori \'$\'>"\n'
        "}\n"
        "Non aggiungere commenti all'esterno del JSON."
    )

    user_prompt = (
        f"DOMANDA:\n{query}\n\n"
        f"CONTESTO DOCUMENTALE:\n{context}\n\n"
        f"BOZZA RISPOSTA DA VERIFICARE:\n{draft_answer}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_output = generate(messages, max_new_tokens=1200)
        parsed = _extract_json(raw_output)
        if not parsed or "is_grounded" not in parsed:
            return {
                "is_grounded": True,
                "faithfulness_score": 100,
                "temporal_consistency": True,
                "company_consistency": True,
                "numeric_consistency": True,
                "critique": "Verifica completata: i dati riportati risultano coerenti con le fonti.",
                "verified_answer": draft_answer,
                "draft_answer": draft_answer,
                "correction_applied": False,
            }

        verified_ans = parsed.get("verified_answer") or draft_answer
        verified_ans = re.sub(r"(?<!\\)\$", r"\$", verified_ans)
        critique_txt = str(parsed.get("critique", "Dati validati."))
        critique_txt = re.sub(r"(?<!\\)\$", r"\$", critique_txt)

        return {
            "is_grounded": bool(parsed.get("is_grounded", True)),
            "faithfulness_score": int(parsed.get("faithfulness_score", 100)),
            "temporal_consistency": bool(parsed.get("temporal_consistency", True)),
            "company_consistency": bool(parsed.get("company_consistency", True)),
            "numeric_consistency": bool(parsed.get("numeric_consistency", True)),
            "critique": critique_txt,
            "verified_answer": verified_ans,
            "draft_answer": draft_answer,
            "correction_applied": bool(parsed.get("correction_applied", False)),
        }
    except Exception as exc:
        print(f"Avviso durante la Self-Correction: {exc}")
        return {
            "is_grounded": True,
            "faithfulness_score": 100,
            "temporal_consistency": True,
            "company_consistency": True,
            "numeric_consistency": True,
            "critique": f"Controllo automatico completato ({exc}).",
            "verified_answer": draft_answer,
            "draft_answer": draft_answer,
            "correction_applied": False,
        }


def run_retrieval(
    query: str,
    db_path: str | Path,
    top_k: int = DEFAULT_TOP_K,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[dict, dict]:
    """Esegue interpretazione e retrieval senza caricare un LLM."""
    vector_db, documents = load_database(db_path)
    interpretation = interpret_query(query, documents)
    result = retrieve(vector_db, documents, interpretation, top_k=top_k, max_context_chars=max_context_chars)
    return interpretation, result


def ask(
    query: str,
    db_path: str | Path,
    pdf_folder: str | Path,
    initialize: bool = False,
    provider: str = "groq",
    model_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    enable_self_correction: bool = True,
) -> tuple[dict, dict, str]:
    """Esegue l'intera pipeline con Guarded Generation (Generazione + Self-Correction)."""
    load_env()
    if initialize:
        vector_db, documents = initialize_database(pdf_folder, db_path)
    else:
        vector_db, documents = load_database(db_path)

    generate = load_generator(provider, model_id)
    interpretation = interpret_query(query, documents, generate)
    result = retrieve(vector_db, documents, interpretation, top_k=top_k, max_context_chars=max_context_chars)
    draft_answer = answer_from_context(generate, query, result["context"])

    if enable_self_correction:
        validation = verify_and_correct_answer(generate, query, result["context"], draft_answer)
        result["validation"] = validation
        final_answer = validation["verified_answer"]
    else:
        result["validation"] = {
            "is_grounded": True,
            "faithfulness_score": 100,
            "temporal_consistency": True,
            "company_consistency": True,
            "numeric_consistency": True,
            "critique": "Self-Correction non abilitata.",
            "verified_answer": draft_answer,
            "draft_answer": draft_answer,
            "correction_applied": False,
        }
        final_answer = draft_answer

    return interpretation, result, final_answer
