# RAG sui report finanziari

Il progetto legge i report PDF, li divide in chunk, li salva in Chroma e usa i
chunk più pertinenti per rispondere a una domanda.

## Struttura

```text
main.py              query e impostazioni da modificare
app/chunks.py        lettura PDF, metadati e chunking
app/embeddings.py    modello di embedding
app/chroma_dao.py    accesso a Chroma
app/pipeline.py      inizializzazione, retrieval e generazione
app/data/            report PDF
```

## Avvio

Installa le dipendenze:

```bash
uv sync
```

Crea un file `.env` nella root del progetto:

```text
GROQ_API_KEY=la-tua-chiave
RAG_LLM_PROVIDER=groq
GROQ_MODEL_ID=openai/gpt-oss-20b
```

Alla prima esecuzione imposta in `main.py`:

```python
INITIALIZE_DB = True
```

Avvia il programma:

```bash
uv run python main.py
```

## Frontend Web (Streamlit)

Per avviare l'interfaccia web interattiva:

```bash
uv run streamlit run app_ui.py
```

oppure:

```bash
streamlit run app_ui.py
```

L'interfaccia ti permette di:
- Inserire domande o selezionare query di esempio rapide
- Visualizzare le risposte dell'analista finanziario formattate con tabelle e KPI
- Ispezionare la diagnostica (aziende, anni, filtri di metadati applicati)
- Leggere tutti i singoli chunk estratti dai PDF utilizzati come contesto

## Frontend Comparativo Self-Correction (Porta 8502)

Per confrontare a video la bozza non verificata vs la risposta bonificata dall'Auditor:

```bash
uv run streamlit run app_self_correction_ui.py --server.port 8502
```

oppure fare doppio clic sul file `avvia_inspector_8502.bat`.
L'interfaccia si aprirà su `http://localhost:8502`.


