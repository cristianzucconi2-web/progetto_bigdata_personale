"""Punto di accesso semplice alla pipeline RAG."""

import json
import os
import sys
from pathlib import Path

from app.pipeline import ask, load_env, run_retrieval


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parent

# Modifica qui la domanda da eseguire.
QUERY = "Confronta i ricavi di Coca-Cola e Pepsi nel 2022"
# Impostazioni principali.
# Per usare AWS S3 (quando riattivato):
# PDF_FOLDER = "s3://miei-bilanci-rag-10k"
PDF_FOLDER = PROJECT_ROOT / "app" / "data"
DB_PATH = Path("chroma_db_bilanci")
INITIALIZE_DB = False
RETRIEVAL_ONLY = False
TOP_K = 28

load_env()
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-20b")


def main() -> None:
    if RETRIEVAL_ONLY:
        interpretation, result = run_retrieval(QUERY, DB_PATH, top_k=TOP_K)
        answer = None
    else:
        interpretation, result, answer = ask(
            query=QUERY,
            db_path=DB_PATH,
            pdf_folder=PDF_FOLDER,
            initialize=INITIALIZE_DB,
            provider=LLM_PROVIDER,
            model_id=LLM_MODEL,
            top_k=TOP_K,
        )

    print("INTERPRETAZIONE")
    print(json.dumps(interpretation, indent=2, ensure_ascii=False))
    print("\nDIAGNOSTICA RETRIEVAL")
    print(json.dumps(result["diagnostics"], indent=2, ensure_ascii=False))

    if answer is not None:
        validation = result.get("validation")
        if validation:
            print("\n" + "=" * 55)
            print("🛡️ REPORT GUARDED GENERATION (SELF-CORRECTION)")
            print(f"- Grounded sulle fonti: {'SÌ' if validation.get('is_grounded') else 'NO'}")
            print(f"- Faithfulness Score: {validation.get('faithfulness_score', 100)}%")
            print(f"- Accuratezza Numerica: {'CORRETTA' if validation.get('numeric_consistency') else 'ATTENZIONE'}")
            print(f"- Coerenza Temporale: {'ALLINEATA' if validation.get('temporal_consistency') else 'ATTENZIONE'}")
            print(f"- Attribuzione Aziendale: {'ESATTA' if validation.get('company_consistency') else 'ATTENZIONE'}")
            print(f"- Note Auditor: {validation.get('critique')}")
            print("=" * 55)

        print("\nRISPOSTA FINALE")
        print(answer)
        return

    print("\nCHUNK RECUPERATI")
    for index, document in enumerate(result["documents"], start=1):
        metadata = document.metadata
        preview = " ".join(document.page_content.split())[:500]
        print(
            f"\n[{index}] {metadata.get('source')} | "
            f"{metadata.get('company')} | anno={metadata.get('fiscal_year')} | "
            f"chunk={metadata.get('chunk_index')}\n{preview}"
        )


if __name__ == "__main__":
    main()
