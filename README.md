# Data-Centric RAG for Financial Analytics
**Progetto di Big Data — Topic 2: RAG for Data Analytics (Data-Centric GenAI)**  
*Autore: Cristian Zucconi*

---

## 📁 Contenuto del Repository

All'interno di questo repository sono disponibili tutti i materiali ufficiali del progetto:

1. **Relazione Finale di Progetto** (nella cartella [`relazione/`](relazione/)):
   - Disponibile sia in formato **PDF** (`Relazione_progetto_BigData_Cristian_Zucconi.pdf`) sia in formato modificabile **Word** (`Relazione_progetto_BigData_Cristian_Zucconi.docx`).
2. **Presentazione Ufficiale** (nella cartella [`presentazione_powerpoint/`](presentazione_powerpoint/)):
   - Slide complete in formato PowerPoint (`Financial_RAG_v2.pptx`).
3. **Codice Sorgente Completo**:
   - Pipeline RAG avanzata, algoritmi di ranking gerarchico adattivo, modulo di Guarded Generation con Self-Correction e due dashboard web Streamlit.
4. **Dati e Bilanci Ufficiali** (nella cartella [`app/data/`](app/data/)):
   - Contiene **106 bilanci annuali ufficiali (Form 10-K e Form 20-F)** relativi a **14 primarie multinazionali** (tra cui Apple, Microsoft, Amazon, Meta, Spotify, Target, Mastercard, Salesforce, ecc.).
   - *Nota sui dati*: per evitare di appesantire eccessivamente il repository GitHub e rispettare i limiti di caricamento, sono stati inclusi localmente 106 bilanci PDF di 14 aziende; il corpus integrale completo (comprendente tutte le 22 multinazionali, tra cui Walmart, Tesla, Nvidia, PayPal, ecc.) è archiviato su **AWS S3** (`s3://miei-bilanci-rag-10k`) e i restanti file possono essere forniti o sincronizzati su richiesta se necessario.

---

## Struttura del Progetto

```text
relazione/                 Relazione finale di progetto (PDF e Word)
presentazione_powerpoint/  Presentazione del progetto (PowerPoint)
app/data/                  Report annuali PDF ufficiali (106 bilanci)
app/pipeline.py            Pipeline RAG: retrieval, ranking adattivo e self-correction
app/chunks.py              Lettura PDF, metadati e chunking (800 car. con overlap 100)
app/embeddings.py          Modello di embedding locale (all-MiniLM-L6-v2)
app/chroma_dao.py          Accesso e gestione indici HNSW su ChromaDB
app_ui.py                  Interfaccia web principale Streamlit (porta 8501)
app_self_correction_ui.py  Dashboard Auditor Inspector Prima vs Dopo (porta 8502)
ingest.py                  Script di ingestion incrementale dei PDF nel vector DB
main.py                    Punto di accesso da riga di comando (CLI)
benchmark.py               Benchmark comparativo e valutazione Faithfulness Score
```

---

## Avvio Rapido

### 1. Installazione Dipendenze
Il progetto gestisce l'ambiente e i pacchetti con `uv` su Python 3.12:

```bash
uv sync
```

### 2. Configurazione Ambiente
Crea un file `.env` nella radice del progetto:

```text
GROQ_API_KEY=la-tua-chiave-api
RAG_LLM_PROVIDER=groq
GROQ_MODEL_ID=openai/gpt-oss-20b
```

### 3. Creazione del Database Vettoriale
Per creare e popolare il database ChromaDB locale a partire dai bilanci presenti in `app/data/`:

```bash
python ingest.py
```

### 4. Esecuzione da Riga di Comando (CLI)

```bash
uv run python main.py
```

---

## 🖥️ Interfacce Grafiche Web (Streamlit)

Il sistema include due dashboard distinte:

### 1. Interfaccia Principale di Analisi (Porta 8501)
Permette di porre domande libere in linguaggio naturale, selezionare provider cloud (Groq LPU) o modelli locali, regolare i parametri di retrieval (Top-K e Budget di Contesto) e visualizzare risposte con tabelle comparative e formule matematiche:

```bash
uv run streamlit run app_ui.py
```
oppure:
```bash
streamlit run app_ui.py
```
Accessibile da browser su: `http://localhost:8501`

### 2. Financial Auditor Inspector — Prima vs Dopo (Porta 8502)
Dashboard dedicata a mostrare il funzionamento della **Guarded Generation e Self-Correction**: visualizza affiancate la bozza grezza dell'analista e la versione certificata dall'Auditor contabile con relativo Faithfulness Score (100%), includendo uno Stress Test Simulator di allucinazioni:

```bash
uv run streamlit run app_self_correction_ui.py --server.port 8502
```
oppure facendo doppio clic sul file `avvia_inspector_8502.bat`. Accessibile da browser su: `http://localhost:8502`
