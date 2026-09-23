"""Frontend interattivo in Streamlit per la pipeline RAG sui bilanci 10-K."""

import json
import os
import re
from pathlib import Path

import streamlit as st

from app.pipeline import (
    answer_from_context,
    interpret_query,
    load_database,
    load_env,
    load_generator,
    retrieve,
    verify_and_correct_answer,
)


def format_financial_markdown(text: str) -> str:
    """Pulisce la formattazione finanziaria: converte i comandi LaTeX in formule leggibili
    ed esegue l'escape di eventuali $ monetari per evitare conflitti di rendering in Streamlit.
    """
    if not text:
        return ""

    # 1. Rimuovi comandi LaTeX di testo: \text{...}, \mathbf{...}, ecc.
    text = re.sub(r"\\(?:text|mathbf|mathit|mathrm)\{([^{}]+)\}", r"\1", text)

    def _clean_frac(match):
        num = match.group(1).strip()
        den = match.group(2).strip()
        if any(op in num for op in ["+", "-", " "]):
            num = f"({num})"
        return f"{num} / {den}"

    # 2. Converti frazioni LaTeX \frac{a}{b} -> (a) / b
    for _ in range(3):
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", _clean_frac, text)

    # 3. Converti simboli matematici LaTeX in simboli standard
    latex_symbols = {
        r"\times": "×",
        r"\cdot": "·",
        r"\approx": "≈",
        r"\pm": "±",
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\neq": "≠",
        r"\%": "%",
        r"\div": "÷",
        r"\rightarrow": "→",
    }
    for sym, repl in latex_symbols.items():
        text = text.replace(sym, repl)

    # 4. Rimuovi blocchi matematici $$...$$ o \$\$...\$\$
    text = re.sub(r"(?:\\?\$){2}\s*(.*?)\s*(?:\\?\$){2}", r"\1", text, flags=re.DOTALL)

    # 5. Rimuovi $ o \$ che racchiudono formule con operatori matematici
    text = re.sub(r"(?<![A-Za-z0-9])\\?\$([^\$\n]*?[=×·/≈±≤≥≠][^\$\n]*?)\\?\$", r"\1", text)

    # Rimuovi \$ o $ residui prima di parentesi (es. '$(13.247' o '$(')
    text = re.sub(r"\\?\$(?=\s*\()", "", text)
    # Rimuovi \$ o $ residui alla fine di una formula (es. '100\$' o '12,96%\$')
    text = re.sub(r"(?<=[0-9%])\\?\$(?!\d)", "", text)

    # 6. Escape dei soli simboli di valuta dollaro ($100M) per non rompere il markdown di Streamlit
    text = re.sub(r"(?<!\\)\$(?=\s*\d)", r"\$", text)

    return text


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = Path("chroma_db_bilanci")
PDF_FOLDER = PROJECT_ROOT / "app" / "data"

# Carica variabili d'ambiente (.env)
load_env()

# Configurazione Pagina Streamlit
st.set_page_config(
    page_title="RAG Bilanci 10-K - Financial Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS per rifinire l'interfaccia
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .chunk-card {
        background-color: #F9FAFB;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
    }
    .chunk-header {
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.4rem;
    }
    .answer-container {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Caricamento indice vettoriale Chroma ed embedding...")
def get_cached_database(db_path_str: str):
    """Carica il database Chroma in cache per evitare ricaricamenti a ogni interazione."""
    return load_database(Path(db_path_str))


# Inizializzazione Session State
if "query_text" not in st.session_state:
    st.session_state.query_text = "Quali sono i ricavi di Apple nel 2024?"
if "run_trigger" not in st.session_state:
    st.session_state.run_trigger = False
if "last_results" not in st.session_state:
    st.session_state.last_results = None


# Sidebar: Impostazioni e Configurazione
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("Impostazioni RAG")

    st.markdown("### ⚙️ Modello Generativo")
    provider = st.selectbox(
        "Provider LLM",
        options=["groq", "local"],
        format_func=lambda x: "Groq (Cloud API)" if x == "groq" else "Locale (Transformers)",
        index=0,
    )

    if provider == "groq":
        env_groq_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input(
            "Groq API Key",
            value=env_groq_key,
            type="password",
            help="Puoi impostarla nel file .env o incollarla qui direttamente.",
        )
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

        model_id = st.selectbox(
            "Modello Groq",
            options=[
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "qwen/qwen3.8-27b",
            ],
            index=0,
        )
    else:
        model_id = st.text_input(
            "Modello Locale Hugging Face",
            value="Qwen/Qwen2.5-3B-Instruct",
        )

    st.markdown("---")
    st.markdown("### 🔍 Parametri Retrieval")
    top_k = st.slider("Top K Chunks da recuperare", min_value=5, max_value=48, value=28, step=1)
    max_context_chars = st.slider(
        "Budget Contesto (Caratteri)",
        min_value=6000,
        max_value=30000,
        value=16000,
        step=1000,
        help="Massimo numero di caratteri del contesto iniettati nel prompt dell'LLM (divisi equamente tra le aziende individuate).",
    )
    retrieval_only = st.checkbox(
        "Solo Retrieval (senza LLM)",
        value=False,
        help="Visualizza i chunk recuperati e la diagnostica senza chiamare il modello generativo.",
    )
    enable_self_correction = st.checkbox(
        "🛡️ Attiva Self-Correction (Guarded)",
        value=True,
        help="Esegue un secondo passaggio LLM per convalidare l'accuratezza dei numeri, la coerenza temporale ed eliminare allucinazioni.",
    )

    st.markdown("---")
    st.markdown("### 🗄️ Stato Indice Chroma")
    
    try:
        vector_db, all_documents = get_cached_database(str(DB_PATH))
        companies = sorted({doc.metadata.get("company") for doc in all_documents if doc.metadata.get("company")})
        years = sorted({doc.metadata.get("fiscal_year") for doc in all_documents if doc.metadata.get("fiscal_year")})
        
        st.success(f"Indice pronto: **{len(all_documents):,}** chunk")
        st.markdown(f"**Aziende:** {', '.join(companies)}")
        st.markdown(f"**Anni fiscali:** {min(years)} - {max(years)}")
    except Exception as exc:
        st.error(f"Errore caricamento indice: {exc}")
        all_documents = []


# Main Dashboard Area
st.markdown('<div class="main-header">📊 RAG Financial Analyst</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interroga i report finanziari annuali (Form 10-K) con interpretazione deterministica, '
    'retrieval con context expansion e risposta vincolata alle fonti.</div>',
    unsafe_allow_html=True,
)


def set_preset_and_run(query: str):
    st.session_state.query_text = query
    st.session_state.run_trigger = True


# Preset Query Buttons
st.markdown("##### 💡 Domande frequenti rapide")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    if st.button("🍎 Ricavi di Apple nel 2024", use_container_width=True):
        set_preset_and_run("Quali sono i ricavi di Apple nel 2024?")
with col_p2:
    if st.button("☁️ Confronto AWS vs Azure 2024", use_container_width=True):
        set_preset_and_run("Compare the revenue and growth rate of Amazon Web Services (AWS) and Microsoft Azure for fiscal year 2024.")
with col_p3:
    if st.button("🔬 Spese R&D Meta e Microsoft 2023", use_container_width=True):
        set_preset_and_run("Who spent more on Research and Development (R&D) in 2023, Meta or Microsoft?")

# Form Input Query
with st.form(key="rag_query_form"):
    query_input = st.text_area(
        "Inserisci la tua domanda finanziaria:",
        value=st.session_state.query_text,
        placeholder="Es. Quali sono stati i ricavi di Apple nel 2024 e come sono ripartiti?",
        height=80,
    )
    submit_btn = st.form_submit_button("🚀 Analizza Bilanci", type="primary", use_container_width=True)

# Verifica se dobbiamo eseguire la query
should_run = submit_btn or st.session_state.run_trigger
if should_run:
    st.session_state.run_trigger = False
    st.session_state.query_text = query_input

    if not query_input.strip():
        st.warning("⚠️ Inserisci una domanda prima di avviare l'analisi.")
    elif not all_documents:
        st.error("❌ L'indice Chroma non è disponibile. Assicurati che sia stato inizializzato.")
    else:
        with st.spinner("Elaborazione in corso: interpretazione query, filtri metadati e retrieval..."):
            try:
                # 1. Carica generatore se necessario
                generator_fn = None
                if not retrieval_only:
                    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
                        st.error("⚠️ GROQ_API_KEY non trovata. Inseriscila nella barra laterale a sinistra.")
                        st.stop()
                    generator_fn = load_generator(provider=provider, model_id=model_id)

                # 2. Interpretazione Query
                interpretation = interpret_query(query_input, all_documents, generate=generator_fn)

                # 3. Retrieval
                retrieval_res = retrieve(
                    vector_db=vector_db,
                    documents=all_documents,
                    interpretation=interpretation,
                    top_k=top_k,
                    max_context_chars=max_context_chars,
                )

                # 4. Generazione Risposta (se non retrieval_only)
                answer = None
                validation_report = None
                if not retrieval_only and generator_fn is not None:
                    with st.spinner("Generazione dell'analisi finanziaria (Bozza)..."):
                        draft_answer = answer_from_context(generator_fn, query_input, retrieval_res["context"])

                    if enable_self_correction:
                        with st.spinner("🛡️ Guarded Generation: Validazione, coerenza temporale e self-correction..."):
                            validation_report = verify_and_correct_answer(
                                generator_fn,
                                query_input,
                                retrieval_res["context"],
                                draft_answer,
                            )
                            answer = validation_report["verified_answer"]
                    else:
                        answer = draft_answer

                # Salva i risultati nel session_state
                st.session_state.last_results = {
                    "query": query_input,
                    "answer": answer,
                    "validation_report": validation_report,
                    "interpretation": interpretation,
                    "retrieval_res": retrieval_res,
                    "retrieval_only": retrieval_only,
                }

            except Exception as exc:
                st.error(f"❌ Si è verificato un errore: {exc}")
                st.exception(exc)


# Render dei Risultati (se presenti in session_state)
if st.session_state.last_results:
    results = st.session_state.last_results
    interpretation = results["interpretation"]
    retrieval_res = results["retrieval_res"]
    answer = results["answer"]
    validation = results.get("validation_report")

    st.markdown("---")

    # Sezione Risposta Finale
    if answer:
        st.markdown("### 📝 Risposta dell'Analista Finanziario")

        # Badge Guarded Generation / Veracity
        if validation:
            score = validation.get("faithfulness_score", 100)
            if validation.get("correction_applied"):
                st.warning(
                    f"🛡️ **Guarded Generation (Self-Correction Applicata)** — "
                    f"Fedeltà Documentale: **{score}%** | Il validatore ha allineato la risposta al contesto."
                )
            elif validation.get("is_grounded", True):
                st.success(
                    f"🛡️ **Guarded Generation (Output Garantito)** — "
                    f"Fedeltà Documentale: **{score}%** | Nessuna allucinazione rilevata. Dati verificati sulle fonti ufficiali."
                )
            else:
                st.error(
                    f"🛡️ **Guarded Generation**: Livello di confidenza limitato ({score}%)."
                )

        st.markdown(format_financial_markdown(answer))
        st.markdown("")

        # Expander Dettagli Self-Correction
        if validation:
            with st.expander("🛡️ Dettagli Guarded Generation & Self-Correction (Auditor)", expanded=False):
                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                with col_v1:
                    st.metric("Faithfulness Score", f"{validation.get('faithfulness_score', 100)}%")
                with col_v2:
                    st.metric("Accuratezza Numerica", "✅ Verificata" if validation.get("numeric_consistency") else "⚠️ Discrepanza")
                with col_v3:
                    st.metric("Coerenza Temporale", "✅ Conforme" if validation.get("temporal_consistency") else "⚠️ Non allineata")
                with col_v4:
                    st.metric("Attribuzione Aziendale", "✅ Corretta" if validation.get("company_consistency") else "⚠️ Confusa")

                st.markdown(f"**Critique / Note dell'Auditor:** {format_financial_markdown(validation.get('critique', ''))}")
                if validation.get("correction_applied"):
                    st.markdown("**Bozza originale prima della Self-Correction:**")
                    st.info(validation.get("draft_answer"))

        st.markdown("")

    # Sezione Diagnostica Retrieval & Metadata
    with st.expander("🔍 Diagnostica Retrieval & Metadati", expanded=(results["retrieval_only"])):
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("Aziende Identificate", str(interpretation.get("companies") or "Tutte"))
        with col_d2:
            st.metric("Anni Fiscali", str(interpretation.get("years") or "Tutti"))
        with col_d3:
            st.metric("Chunk Candidati", retrieval_res["diagnostics"].get("candidate_chunks", 0))
        with col_d4:
            st.metric("Chunk nel Contesto", retrieval_res["diagnostics"].get("context_chunks", 0))

        st.markdown("**Strategia di Retrieval:** `" + retrieval_res["diagnostics"].get("retrieval_strategy", "n/d") + "`")
        st.markdown("**Filtro Metadata Applicato:**")
        st.code(json.dumps(retrieval_res["diagnostics"].get("metadata_filter"), indent=2), language="json")
        st.markdown("**Parole chiave espanse:**")
        st.write(", ".join(interpretation.get("keywords_en", [])))

    # Sezione Chunk di Contesto Recuperati
    with st.expander(f"📚 Chunk di Contesto Recuperati ({len(retrieval_res['documents'])})", expanded=False):
        for idx, doc in enumerate(retrieval_res["documents"], start=1):
            meta = doc.metadata
            st.markdown(
                f"""
                <div class="chunk-card">
                    <div class="chunk-header">
                        [#{idx}] 📄 <b>{meta.get('source')}</b> | 🏢 Azienda: <b>{meta.get('company', '').upper()}</b> | 📅 Anno: <b>{meta.get('fiscal_year')}</b> | 🧩 Chunk #{meta.get('chunk_index')}
                    </div>
                    <div style="color: #374151; white-space: pre-wrap; font-family: monospace; font-size: 0.88rem;">{doc.page_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
