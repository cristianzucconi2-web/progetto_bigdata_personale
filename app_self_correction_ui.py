"""Dashboard di Analisi Comparativa: Prima vs Dopo la Self-Correction.

Eseguibile su una porta dedicata (es. http://localhost:8502) per visualizzare
il processo di revisione contabile e auditing dei fatti in tempo reale.
"""

import difflib
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.pipeline import (
    answer_from_context,
    interpret_query,
    load_database,
    load_env,
    load_generator,
    retrieve,
    verify_and_correct_answer,
)

load_env()

st.set_page_config(
    page_title="Self-Correction Inspector (Prima vs Dopo)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Stile CSS per il confronto affiancato
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .card-before {
        background-color: #FEF2F2;
        border: 2px solid #F87171;
        border-radius: 8px;
        padding: 1.2rem;
        min-height: 260px;
    }
    .card-after {
        background-color: #F0FDF4;
        border: 2px solid #4ADE80;
        border-radius: 8px;
        padding: 1.2rem;
        min-height: 260px;
    }
    .card-header-before {
        font-weight: 700;
        color: #991B1B;
        font-size: 1.15rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-header-after {
        font-weight: 700;
        color: #166534;
        font-size: 1.15rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .auditor-box {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 1.2rem 0;
    }
    .diff-del {
        background-color: #FECACA;
        color: #991B1B;
        text-decoration: line-through;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 600;
    }
    .diff-ins {
        background-color: #BBF7D0;
        color: #166534;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


@st.cache_resource(show_spinner="Caricamento indice vettoriale Chroma (132k chunk)...")
def get_database():
    return load_database(Path("chroma_db_bilanci"))


@st.cache_resource(show_spinner="Inizializzazione generatore LLM Groq...")
def get_generator(model_id: str):
    return load_generator("groq", model_id)


# Barra laterale
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("🛡️ Auditor Config")

    model_id = st.selectbox(
        "Modello LLM (Groq)",
        options=[
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
        ],
        index=0,
    )

    top_k = st.slider("Top K Chunks da recuperare", min_value=10, max_value=40, value=28)
    max_context_chars = st.slider(
        "Budget Contesto (Caratteri)",
        min_value=6000,
        max_value=30000,
        value=16000,
        step=1000,
        help="Massimo numero di caratteri del contesto iniettati nel prompt dell'LLM.",
    )

    st.markdown("---")
    st.markdown("### 🔬 Modalità Operativa")
    mode = st.radio(
        "Scegli cosa testare:",
        options=[
            "🚀 Query Reale End-to-End (ChromaDB + LLM)",
            "🧪 Stress Test / Simulatore Allucinazioni",
        ],
    )

    st.markdown("---")
    st.caption("Esecuzione isolata su porta personalizzata (es. 8502). Non interferisce con l'app principale.")

# Titolo principale
st.markdown('<div class="main-title">🛡️ Financial Auditor: Ispettore Self-Correction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Confronto diretto <b>PRIMA vs DOPO</b>: osserva come il revisore contabile convalida i numeri, '
    'intercetta le allucinazioni e bonifica la risposta finale sui bilanci 10-K.</div>',
    unsafe_allow_html=True,
)

# Caricamento risorse
try:
    vdb, docs = get_database()
    generator = get_generator(model_id)
except Exception as e:
    st.error(f"Errore caricamento database o modello: {e}")
    st.stop()


# =========================================================================
# MODALITÀ 1: QUERY REALE END-TO-END
# =========================================================================
if mode == "🚀 Query Reale End-to-End (ChromaDB + LLM)":
    st.markdown("##### 💡 Casi di Test Rapidi")
    col_t1, col_t2, col_t3 = st.columns(3)

    default_query = "Quanti dipendenti ha Amazon nel 2023?"
    if "audit_query" not in st.session_state:
        st.session_state.audit_query = default_query

    with col_t1:
        if st.button("👥 Amazon: Dipendenti 2023 (Chunk 85)", use_container_width=True):
            st.session_state.audit_query = "Quanti dipendenti ha Amazon nel 2023?"
            st.rerun()
    with col_t2:
        if st.button("💶 Spotify: Ricavi 2023 (Form 20-F in €)", use_container_width=True):
            st.session_state.audit_query = "Quali sono stati i ricavi di Spotify nel 2023?"
            st.rerun()
    with col_t3:
        if st.button("🏬 Walmart vs Target: Ricavi 2023", use_container_width=True):
            st.session_state.audit_query = "Confronta i ricavi di Walmart e Target nel 2023"
            st.rerun()

    query_input = st.text_input("Domanda finanziaria da analizzare:", value=st.session_state.audit_query)

    if st.button("🚀 Esegui Analisi e Confronta Prima/Dopo", type="primary", use_container_width=True):
        if not query_input.strip():
            st.warning("Inserisci una domanda valida.")
        else:
            with st.spinner("1/3 Recupero e partizionamento chunk da ChromaDB..."):
                interp = interpret_query(query_input, docs, generator)
                retrieval_res = retrieve(vdb, docs, interp, top_k=top_k, max_context_chars=max_context_chars)
                context = retrieval_res["context"]

            with st.spinner("2/3 Generazione Bozza non verificata (Analista)..."):
                draft = answer_from_context(generator, query_input, context)

            with st.spinner("3/3 Auditing e Self-Correction in corso..."):
                val = verify_and_correct_answer(generator, query_input, context, draft)

            st.session_state.audit_result = {
                "query": query_input,
                "context": context,
                "draft": draft,
                "val": val,
                "chunks": retrieval_res["documents"],
            }

# =========================================================================
# MODALITÀ 2: SIMULATORE ALLUCINAZIONI / STRESS TEST
# =========================================================================
else:
    st.info(
        "💡 **Stress Test Mode**: Puoi testare direttamente l'Auditor iniettando cifre inventate, valute sbagliate o scambi di aziende "
        "per vedere se la Self-Correction è in grado di individuarle ed eliminarle."
    )

    preset_sim = st.selectbox(
        "Carica uno scenario di simulazione predefinito:",
        options=[
            "Amazon 2023 con Allucinazione (+12% crescita e $85B costo personale)",
            "Spotify 2023 con Valuta Sbagliata ($ invece di €)",
            "Walmart vs Target con Inversione Dati di Utile",
            "Personalizzato",
        ],
    )

    if preset_sim == "Amazon 2023 con Allucinazione (+12% crescita e $85B costo personale)":
        sim_query = "Quanti dipendenti ha Amazon nel 2023?"
        sim_context = (
            "[amazon | anno 2023 | chunk 85]\n"
            "Item 1. Business - Human Capital\n"
            "Our employees are critical to our mission of being Earth’s most customer-centric company. "
            "As of December 31, 2023, we employed approximately 1,525,000 full-time and part-time employees. "
            "Additionally, we use independent contractors and temporary personnel to supplement our workforce."
        )
        sim_draft = (
            "Nel 2023 Amazon contava circa 1.525.000 dipendenti a tempo pieno e parziale, "
            "registrando una forte crescita dell'organico del +12% rispetto al 2022 con una spesa complessiva "
            "per il personale pari a $85 miliardi."
        )
    elif preset_sim == "Spotify 2023 con Valuta Sbagliata ($ invece di €)":
        sim_query = "Quali sono i ricavi di Spotify nel 2023?"
        sim_context = (
            "[spotify | anno 2023 | chunk 761]\n"
            "Consolidated statement of operations (in € millions)\n"
            "Revenue: 2023: €13,247 | 2022: €11,727 | 2021: €9,668\n"
            "Operating loss: 2023: €(449) | 2022: €(659)"
        )
        sim_draft = (
            "Nel 2023, i ricavi totali di Spotify sono stati pari a circa $13.247 milioni di dollari "
            "con una perdita operativa di $449 milioni di dollari."
        )
    elif preset_sim == "Walmart vs Target con Inversione Dati di Utile":
        sim_query = "Confronta l'utile netto di Walmart e Target nel 2023"
        sim_context = (
            "[walmart | anno 2023 | chunk 496]\n"
            "Consolidated Statements of Income (in millions)\n"
            "Total revenues: $611,289 | Net income: $11,680\n"
            "[target | anno 2023 | chunk 10]\n"
            "Statements of Operations (in millions)\n"
            "Total revenue: $107,412 | Net earnings: $2,780"
        )
        sim_draft = (
            "Nel 2023, Walmart ha registrato un utile netto pari a $2.780 milioni, "
            "mentre Target ha registrato un utile netto nettamente superiore pari a $11.680 milioni."
        )
    else:
        sim_query = "Test personalizzato"
        sim_context = "Inserisci qui il contesto documentale..."
        sim_draft = "Inserisci qui la bozza da convalidare..."

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        inp_query = st.text_input("Domanda:", value=sim_query)
        inp_context = st.text_area("Contesto Ufficiale (Fonte di Verità):", value=sim_context, height=140)
    with c_s2:
        inp_draft = st.text_area("Bozza Generata con Allucinazione/Errore:", value=sim_draft, height=210)

    if st.button("🛡️ Avvia Revisione Contabile dell'Auditor", type="primary", use_container_width=True):
        with st.spinner("Auditor al lavoro: verifica incrociata su numeri, date, valute ed entità..."):
            val = verify_and_correct_answer(generator, inp_query, inp_context, inp_draft)
            st.session_state.audit_result = {
                "query": inp_query,
                "context": inp_context,
                "draft": inp_draft,
                "val": val,
                "chunks": [],
            }


# =========================================================================
# RENDERING DEI RISULTATI: CONFRONTO PRIMA VS DOPO
# =========================================================================
if "audit_result" in st.session_state and st.session_state.audit_result:
    res = st.session_state.audit_result
    draft = res["draft"]
    val = res["val"]
    verified = val["verified_answer"]
    score = val.get("faithfulness_score", 100)
    correction_applied = val.get("correction_applied", False)

    st.markdown("---")
    st.subheader("📊 Esito Revisione & Certificazione dei Dati")

    # Barra Metriche Auditor
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "Faithfulness Score",
            f"{score}%",
            delta=f"{score - 100}%" if score < 100 else "Massima Fedeltà",
            delta_color="normal" if score == 100 else "inverse",
        )
    with m2:
        st.metric("Accuratezza Numeri", "✅ Verificata" if val.get("numeric_consistency") else "⚠️ Discrepanza")
    with m3:
        st.metric("Coerenza Anno", "✅ Conforme" if val.get("temporal_consistency") else "⚠️ Errore Anno")
    with m4:
        st.metric("Attribuzione Aziende", "✅ Esatta" if val.get("company_consistency") else "⚠️ Scambio Dati")
    with m5:
        if correction_applied:
            st.metric("Stato Correzione", "🛠️ APPLICATA", delta="Bonificata", delta_color="inverse")
        else:
            st.metric("Stato Correzione", "✅ CERTIFICATA", delta="Invariata", delta_color="normal")

    # Box Note Critiche dell'Auditor
    critique = format_financial_markdown(val.get("critique", "Dati conformi alle fonti."))
    if correction_applied:
        st.error(f"**Discrepanze Rilevate dall'Auditor:** {critique}")
    else:
        st.success(f"**Valutazione Auditor:** {critique}")

    # CONFRONTO AFFIANCATO: PRIMA VS DOPO
    st.markdown("### ⚖️ Confronto Diretto: Bozza Iniziale vs Risposta Revisionata")

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown(
            """
            <div class="card-before">
                <div class="card-header-before">
                    <span>🔴 PRIMA: Bozza Iniziale (LLM Standard)</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(format_financial_markdown(draft))
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("⚠️ Non revisionata: può contenere allucinazioni, date errate o valute scambiate.")

    with col_after:
        st.markdown(
            """
            <div class="card-after">
                <div class="card-header-after">
                    <span>🟢 DOPO: Risposta Validata & Bonificata</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(format_financial_markdown(verified))
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("🛡️ Certificata: ogni cifra è provata punto per punto nei documenti ufficiali.")

    # Se c'è stata correzione, mostra il Diff testuale
    if draft.strip() != verified.strip():
        with st.expander("🔍 Dettaglio Modifiche (Cosa è stato tagliato o modificato)", expanded=True):
            st.markdown(
                "Le parti evidenziate mostrano le discrepanze eliminate dall'Auditor rispetto alla fonte ufficiale:"
            )
            col_d_left, col_d_right = st.columns(2)
            with col_d_left:
                st.markdown("**Testo rimosso/corretto (Bozza):**")
                st.code(draft, language=None)
            with col_d_right:
                st.markdown("**Testo bonificato approvato (Finale):**")
                st.code(verified, language=None)

    # Contesto Documentale Utilizzato
    if res.get("context"):
        with st.expander("📄 Contesto Documentale Ufficiale (Fonte di Verità dai Form 10-K)", expanded=False):
            st.text(res["context"])
