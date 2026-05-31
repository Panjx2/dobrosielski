"""
AI Incident Management Copilot
Hybrid Retrieval AI System: SQL + RAG + LLM dla analizy incydentów IT

Implementacja zgodna z "Zadanie do wykonania" z dokumentu:
- SQL: incidents / users / systems
- RAG: procedury bezpieczeństwa, playbook ransomware, definicja incydentu krytycznego
- LLM: Anthropic Claude Haiku z fallbackiem na mock
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional

import anthropic
import faiss
from flask import Flask, jsonify, request, render_template
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. KONFIGURACJA
# ============================================================
DB_PATH = "incidents.db"
DOCUMENTS_DIR = "documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
ANTHROPIC_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1024"))

_anthropic_client: Optional[anthropic.Anthropic] = None
if os.environ.get("ANTHROPIC_API_KEY"):
    _anthropic_client = anthropic.Anthropic()


# ============================================================
# 2. BAZA DANYCH SQL
# ============================================================

def create_database():
    """Tworzy bazę SQLite z tabelami incidents / users / systems i seed-data."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
        DROP TABLE IF EXISTS incidents;
        DROP TABLE IF EXISTS users;
        DROP TABLE IF EXISTS systems;

        CREATE TABLE incidents (
            incident_id INTEGER PRIMARY KEY,
            server_name TEXT,
            severity    TEXT,
            status      TEXT,
            category    TEXT,
            created_at  TEXT
        );

        CREATE TABLE users (
            user_id    INTEGER PRIMARY KEY,
            username   TEXT,
            department TEXT
        );

        CREATE TABLE systems (
            system_id   INTEGER PRIMARY KEY,
            system_name TEXT,
            owner       TEXT
        );
    """)

    users = [
        (1, "jkowalski", "SOC"),
        (2, "anowak", "IT"),
        (3, "pwisniewski", "DevOps"),
        (4, "mzielinska", "SOC"),
    ]

    systems = [
        (1, "srv-db-01",   "DBA Team"),
        (2, "srv-auth-02", "IAM Team"),
        (3, "srv-web-03",  "Web Team"),
        (4, "srv-api-04",  "Platform Team"),
        (5, "srv-mail-05", "Messaging Team"),
    ]

    incidents = [
        (1,  "srv-db-01",   "critical", "resolved",  "ransomware",     "2026-05-03 02:14:00"),
        (2,  "srv-db-01",   "high",     "resolved",  "performance",    "2026-05-07 11:42:00"),
        (3,  "srv-auth-02", "critical", "escalated", "data_breach",    "2026-05-09 19:05:00"),
        (4,  "srv-auth-02", "medium",   "resolved",  "auth_failure",   "2026-05-12 08:30:00"),
        (5,  "srv-web-03",  "low",      "resolved",  "config",         "2026-05-14 14:22:00"),
        (6,  "srv-db-01",   "high",     "resolved",  "disk_full",      "2026-05-17 03:50:00"),
        (7,  "srv-api-04",  "critical", "resolved",  "outage",         "2026-05-21 09:11:00"),
        (8,  "srv-auth-02", "high",     "open",      "auth_failure",   "2026-05-24 22:00:00"),
        (9,  "srv-mail-05", "medium",   "resolved",  "spam",           "2026-05-26 06:40:00"),
        (10, "srv-db-01",   "critical", "open",      "ransomware",     "2026-05-27 01:30:00"),
        (11, "srv-web-03",  "high",     "resolved",  "ddos",           "2026-04-29 17:18:00"),
        (12, "srv-api-04",  "medium",   "resolved",  "latency",        "2026-04-30 12:05:00"),
        (13, "srv-mail-05", "low",      "resolved",  "config",         "2026-06-01 09:00:00"),
        (14, "srv-auth-02", "critical", "open",      "data_breach",    "2026-06-02 04:15:00"),
        (15, "srv-db-01",   "medium",   "resolved",  "backup_failure", "2026-06-03 23:40:00"),
    ]

    cursor.executemany("INSERT INTO users VALUES (?,?,?)", users)
    cursor.executemany("INSERT INTO systems VALUES (?,?,?)", systems)
    cursor.executemany(
        "INSERT INTO incidents VALUES (?,?,?,?,?,?)", incidents
    )

    conn.commit()
    conn.close()


def execute_sql(query: str) -> Dict[str, Any]:
    """Wykonuje zapytanie SQL i zwraca kolumny + wiersze."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description] if cursor.description else []

    conn.close()

    return {"columns": columns, "rows": rows}


# ============================================================
# 3. SQL AGENT (reguły NL -> SQL)
# ============================================================

import re


def generate_sql(question: str, rag_context: str = "") -> Optional[str]:
    """Mapuje pytanie w języku naturalnym na zapytanie SQL."""

    q = question.lower()

    # Wzorzec: "Ile incydentów w [miesiącu]?"
    month_pattern = r'(?:ile|ilu|liczba|ileż).*(?:incydent|zdarzen).*(?:w|za).*(?:stycz|lut|mar|kwiec|maj|czerwc|lip|sierp|wrze|paź|list|grud)'
    if re.search(month_pattern, q):
        month_map = {
            "stycz": "01", "lut": "02", "mar": "03", "kwiec": "04",
            "maj": "05", "czerwc": "06", "lip": "07", "sierp": "08",
            "wrze": "09", "paź": "10", "list": "11", "grud": "12"
        }
        for month_name, month_num in month_map.items():
            if month_name in q:
                return f"""
                SELECT COUNT(*) AS liczba_incydentow
                FROM incidents
                WHERE strftime('%Y-%m', created_at) = '2026-{month_num}';
                """

    # Wzorzec: "Ile krytycznych w [miesiącu]?"
    critical_month_pattern = r'(?:ile|ilu|liczba).*(?:krytyczn).*(?:w|za).*(?:stycz|lut|mar|kwiec|maj|czerwc|lip|sierp|wrze|paź|list|grud)'
    if re.search(critical_month_pattern, q):
        month_map = {
            "stycz": "01", "lut": "02", "mar": "03", "kwiec": "04",
            "maj": "05", "czerwc": "06", "lip": "07", "sierp": "08",
            "wrze": "09", "paź": "10", "list": "11", "grud": "12"
        }
        for month_name, month_num in month_map.items():
            if month_name in q:
                return f"""
                SELECT COUNT(*) AS liczba_krytycznych
                FROM incidents
                WHERE severity = 'critical'
                  AND strftime('%Y-%m', created_at) = '2026-{month_num}';
                """

    # Wzorzec: "Ile ransomware?"
    if re.search(r'(?:ile|ilu|liczba).*ransomware', q):
        return """
        SELECT COUNT(*) AS liczba_ransomware
        FROM incidents
        WHERE category = 'ransomware';
        """

    # Wzorzec: "Które serwery mają najwięcej awarii?"
    if re.search(r'(?:które|jaki|najwi(?:e|ę)cej).*serwer.*awari', q):
        return """
        SELECT server_name, COUNT(*) AS liczba_awarii
        FROM incidents
        GROUP BY server_name
        ORDER BY liczba_awarii DESC
        LIMIT 5;
        """

    # Wzorzec: "Pokaż otwarte incydenty"
    if re.search(r'(?:pokaż|pokaz|lista).*otwart', q):
        return """
        SELECT incident_id, server_name, severity, category, created_at
        FROM incidents
        WHERE status = 'open'
        ORDER BY 
            CASE severity
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'medium' THEN 3
                WHEN 'low' THEN 4
            END,
            created_at DESC;
        """

    return None

# ============================================================
# 4. DOKUMENTY DLA RAG
# ============================================================

DEFAULT_DOCS = {
    "incident_response.txt": """
Procedura reakcji na incydent IT (Incident Response).

1. Wykrycie incydentu przez system monitoringu lub zgłoszenie użytkownika.
2. Klasyfikacja incydentu według poziomu ważności (low / medium / high / critical).
3. Powiadomienie zespołu SOC oraz właściciela systemu.
4. Izolacja zagrożonego systemu od reszty sieci, jeżeli to konieczne.
5. Analiza logów i identyfikacja źródła incydentu.
6. Wdrożenie działań naprawczych zgodnie z procedurą recovery.
7. Dokumentacja incydentu i post-mortem.

Czas reakcji:
- critical: 15 minut,
- high: 1 godzina,
- medium: 4 godziny,
- low: 24 godziny.
""",
    "recovery_procedures.txt": """
Procedura odtwarzania usług (Recovery Procedure).

Standardowe kroki recovery dla serwera produkcyjnego:
1. Restart usługi i weryfikacja, czy problem jest powtarzalny.
2. Weryfikacja logów aplikacyjnych i systemowych.
3. Odtworzenie backupu z ostatniego znanego sprawnego stanu.
4. Walidacja integralności danych po przywróceniu.
5. Eskalacja incydentu do zespołu SOC, jeżeli przyczyna nie została ustalona.
6. Aktualizacja statusu incydentu w systemie ticketowym.

RTO (Recovery Time Objective): 4 godziny dla systemów krytycznych.
RPO (Recovery Point Objective): maksymalnie 1 godzina utraty danych.
""",
    "security_policy.txt": """
Polityka bezpieczeństwa informacji.

Każdy pracownik zobowiązany jest do:
- przestrzegania zasady najmniejszych uprawnień (least privilege),
- stosowania uwierzytelniania wieloskładnikowego (MFA) dla wszystkich kont produkcyjnych,
- raportowania podejrzanych zdarzeń do zespołu SOC w ciągu 1 godziny,
- nieudostępniania haseł i tokenów dostępu osobom trzecim,
- używania wyłącznie zatwierdzonego oprogramowania.

Wszystkie incydenty bezpieczeństwa muszą być logowane w systemie SIEM
i objęte procedurą response.
""",
    "ransomware_playbook.txt": """
Playbook obsługi ataku ransomware.

Krok 1: Natychmiastowa izolacja zainfekowanych hostów od sieci
(odłączenie kabla / odpięcie z VLAN-u).

Krok 2: Powiadomienie zespołu SOC oraz CISO. Atak ransomware jest
domyślnie traktowany jako incydent krytyczny.

Krok 3: Zachowanie próbek złośliwego oprogramowania i obrazu dysku
do analizy forensic. NIE wyłączać hosta - utracimy zawartość RAM.

Krok 4: Identyfikacja wariantu ransomware (Hash, IOC, behawior).

Krok 5: Sprawdzenie integralności backupów offline. NIE odtwarzać
backupów online, jeśli istnieje ryzyko, że są zaszyfrowane.

Krok 6: Odtworzenie systemów z czystych backupów po potwierdzeniu,
że wektor infekcji został zamknięty.

Krok 7: NIE płacić okupu. Zgłosić incydent organom ścigania
oraz - jeśli dotyczy danych osobowych - do UODO w ciągu 72h.

Krok 8: Post-incident review, aktualizacja reguł EDR/SIEM.
""",
    "critical_incident_definition.txt": """
Definicja incydentu krytycznego (Critical Incident).

Incydent jest klasyfikowany jako krytyczny, jeżeli spełnia
co najmniej jeden z poniższych warunków:

- powoduje niedostępność kluczowej usługi biznesowej,
- powoduje utratę lub uszkodzenie danych produkcyjnych,
- stanowi naruszenie bezpieczeństwa (data breach, unauthorized access),
- jest atakiem ransomware lub innym malware o szerokim zasięgu,
- prowadzi do wycieku danych osobowych lub poufnych,
- dotyczy systemu objętego wymaganiami regulacyjnymi (np. RODO, PCI DSS).

Każdy incydent krytyczny musi być natychmiast eskalowany do SOC
oraz raportowany w cotygodniowym przeglądzie zarządu.
""",
}


def create_documents():
    """Tworzy katalog `documents/` i zapisuje przykładowe dokumenty .txt."""

    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    for filename, content in DEFAULT_DOCS.items():
        path = os.path.join(DOCUMENTS_DIR, filename)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as file:
                file.write(content.strip() + "\n")


# ============================================================
# 5. CHUNKING
# ============================================================

def chunk_text(text: str) -> List[str]:
    """Dzieli dokument na nakładające się fragmenty (sliding window)."""

    chunks = []
    start = 0
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += step

    return chunks


def load_documents() -> List[Dict[str, Any]]:
    """Wczytuje wszystkie .txt z DOCUMENTS_DIR i dzieli na chunki."""

    documents = []

    for filename in sorted(os.listdir(DOCUMENTS_DIR)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(DOCUMENTS_DIR, filename)
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        for chunk_id, chunk in enumerate(chunk_text(text)):
            documents.append({
                "source": filename,
                "chunk_id": chunk_id,
                "text": chunk,
            })

    return documents


# ============================================================
# 6. RAG ENGINE (FAISS + SentenceTransformer)
# ============================================================

class RagEngine:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = self._build_index(documents)

    def _build_index(self, documents: List[Dict[str, Any]]):
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_model.encode(
            texts, convert_to_numpy=True
        ).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def search(self, question: str) -> List[Dict[str, Any]]:
        question_embedding = self.embedding_model.encode(
            [question], convert_to_numpy=True
        ).astype("float32")

        _, indices = self.index.search(question_embedding, TOP_K)
        return [self.documents[idx] for idx in indices[0]]


# ============================================================
# 7. LLM (Ollama z fallbackiem na mock)
# ============================================================

def call_llm(prompt: str) -> str:
    """Wywołuje Claude Haiku przez Anthropic API. Fallback: mock z pełnym promptem."""

    if _anthropic_client is None:
        return (
            "[MOCK LLM — brak ANTHROPIC_API_KEY]\n"
            f"Model: {ANTHROPIC_MODEL}\n\n"
            f"--- PROMPT ---\n{prompt}"
        )

    try:
        response = _anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip()
        return text or "[Claude zwrócił pustą odpowiedź]"
    except anthropic.APIError as exc:
        return (
            f"[MOCK LLM — Anthropic API niedostępne: {exc.__class__.__name__}]\n"
            f"Model: {ANTHROPIC_MODEL}\n\n"
            f"--- PROMPT ---\n{prompt}"
        )


# ============================================================
# 8. ROUTER
# ============================================================

def route_question(question: str) -> str:
    """Decyduje, czy pytanie wymaga SQL, RAG, czy HYBRID.

    Hybrid = pytanie łączy zapytanie o dane (SQL) z prośbą o procedurę
    lub definicję z dokumentów (RAG).
    """

    q = question.lower()

    data_keywords = (
        "ile", "ilu", "liczba", "najwięcej", "statyst",
        "otwart", "eskalow", "open", "escalat",
        "który", "które", "którzy",
    )
    knowledge_keywords = (
        "procedur", "definicj", "polityka", "playbook",
        "recovery", "rekomend", "kwalifikuje", "spełnia",
    )

    has_data = any(k in q for k in data_keywords)
    has_knowledge = any(k in q for k in knowledge_keywords)

    if has_data and has_knowledge:
        return "hybrid"
    if has_data:
        return "sql"
    return "rag"


# ============================================================
# 9. ODPOWIEDZI
# ============================================================

def answer_with_sql(question: str) -> Dict[str, Any]:
    sql = generate_sql(question)

    if sql is None:
        return {
            "mode": "sql",
            "answer": "Nie udało się wygenerować SQL dla tego pytania.",
            "sql": None,
        }

    result = execute_sql(sql)
    prompt = f"""
Pytanie:
{question}

Wykonane zapytanie SQL:
{sql}

Wynik:
{result}

Przedstaw wynik w języku biznesowym, krótko i konkretnie.
"""
    answer = call_llm(prompt)
    return {"mode": "sql", "sql": sql, "result": result, "answer": answer}


def answer_with_rag(question: str, rag: RagEngine) -> Dict[str, Any]:
    docs = rag.search(question)
    context = "\n\n".join(
        f"Źródło: {doc['source']}\n{doc['text']}" for doc in docs
    )

    prompt = f"""
Pytanie:
{question}

Kontekst z dokumentów:
{context}

Odpowiedz wyłącznie na podstawie powyższych dokumentów. Wskaż źródło.
"""
    answer = call_llm(prompt)
    return {"mode": "rag", "sources": docs, "answer": answer}


def answer_with_hybrid(question: str, rag: RagEngine) -> Dict[str, Any]:
    docs = rag.search(question)
    rag_context = "\n\n".join(
        f"Źródło: {doc['source']}\n{doc['text']}" for doc in docs
    )

    sql = generate_sql(question, rag_context)
    sql_result = execute_sql(sql) if sql else None

    prompt = f"""
Pytanie:
{question}

Kontekst RAG (procedury, definicje):
{rag_context}

{f"Wynik SQL ({sql}):{sql_result}" if sql else "Brak danych SQL — odpowiadaj wyłącznie na podstawie dokumentów."}

Połącz wiedzę z dokumentów z danymi z bazy i sformułuj odpowiedź
w języku naturalnym dla analityka SOC.
"""
    answer = call_llm(prompt)
    return {
        "mode": "hybrid",
        "sources": docs,
        "sql": sql,
        "result": sql_result,
        "answer": answer,
    }


# ============================================================
# 10. FLASK API
# ============================================================

app = Flask(__name__)


INDEX_HTML = """<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>AI Incident Management Copilot</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; }
  textarea { width: 100%; height: 5rem; font: inherit; padding: .5rem; box-sizing: border-box; }
  button { padding: .5rem 1rem; font: inherit; cursor: pointer; }
  .examples button { margin: .15rem; font-size: .85rem; }
  .out { margin-top: 1rem; padding: 1rem; background: #f4f4f4; border-radius: 6px;
         white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: .9rem; }
  .route { display: inline-block; padding: .1rem .5rem; border-radius: 4px;
           background: #2563eb; color: white; font-size: .8rem; }
  .answer { background: #fff; border: 1px solid #ddd; padding: 1rem; margin-top: .5rem; border-radius: 6px; }
  details { margin-top: .5rem; }
</style>
</head>
<body>
<h1>AI Incident Management Copilot</h1>
<p>Model: <code id="model">…</code></p>

<div class="examples">
  <strong>Przykłady:</strong><br>
  <button onclick="setQ('Ile incydentów krytycznych mamy w maju?')">krytyczne w maju</button>
  <button onclick="setQ('Które serwery mają najwięcej awarii?')">top serwery</button>
  <button onclick="setQ('Pokaż otwarte incydenty')">otwarte</button>
  <button onclick="setQ('Jak reagować na atak ransomware?')">ransomware playbook</button>
  <button onclick="setQ('Co kwalifikuje incydent jako krytyczny?')">definicja krytycznego</button>
  <button onclick="setQ('Ile mamy otwartych incydentów i jaka jest procedura reakcji?')">hybrid</button>
</div>

<p><textarea id="q" placeholder="Zadaj pytanie po polsku…"></textarea></p>
<p><button onclick="ask()">Zapytaj</button></p>

<div id="result"></div>

<script>
fetch('/health').then(r => r.json()).then(d => document.getElementById('model').textContent = d.model);

function setQ(t) { document.getElementById('q').value = t; }

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const result = document.getElementById('result');
  result.innerHTML = '<p>⏳ Myślę…</p>';
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q}),
    });
    const data = await r.json();
    const resp = data.response || {};
    const parts = [
      `<p><span class="route">${data.route}</span></p>`,
      `<div class="answer">${(resp.answer || '').replace(/</g,'&lt;')}</div>`,
    ];
    if (resp.sql)     parts.push(`<details><summary>SQL</summary><div class="out">${resp.sql}</div></details>`);
    if (resp.result)  parts.push(`<details><summary>Wynik SQL</summary><div class="out">${JSON.stringify(resp.result, null, 2)}</div></details>`);
    if (resp.sources) parts.push(`<details><summary>Źródła RAG (${resp.sources.length})</summary><div class="out">${JSON.stringify(resp.sources, null, 2)}</div></details>`);
    result.innerHTML = parts.join('');
  } catch (e) {
    result.innerHTML = `<p style="color:red">Błąd: ${e}</p>`;
  }
}

document.getElementById('q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask();
});
</script>
</body>
</html>"""


def initialize_system() -> RagEngine:
    if not os.path.exists(DB_PATH):
        create_database()
    create_documents()
    return RagEngine(load_documents())


rag_engine = initialize_system()


@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": ANTHROPIC_MODEL})


@app.route("/incidents", methods=["GET"])
def api_incidents():
    try:
        severity = request.args.get("severity", "all")
        status = request.args.get("status", "all")

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        query = "SELECT * FROM incidents WHERE 1=1"
        params = []

        if severity and severity != "all":
            query += " AND severity = ?"
            params.append(severity)
        if status and status != "all":
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        incidents = [dict(row) for row in rows]
        return jsonify(incidents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.get_json() or {}).get("question", "")
    route = route_question(question)

    if route == "sql":
        response = answer_with_sql(question)
    elif route == "rag":
        response = answer_with_rag(question, rag_engine)
    else:
        response = answer_with_hybrid(question, rag_engine)

    return jsonify({"question": question, "route": route, "response": response})


@app.route("/documents", methods=["GET"])
def documents():
    try:
        documents_meta = []
        if not os.path.exists(DOCUMENTS_DIR):
            return jsonify([])

        for filename in sorted(os.listdir(DOCUMENTS_DIR)):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(DOCUMENTS_DIR, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks_count = len(chunk_text(content))
            preview = content[:200].strip()
            if len(content) > 200:
                preview += "..."

            documents_meta.append({
                "name": filename,
                "chunks": chunks_count,
                "size": os.path.getsize(filepath),
                "preview": preview,
                "modified": os.path.getmtime(filepath)
            })

        return jsonify(documents_meta)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sql", methods=["POST"])
def sql_endpoint():
    question = (request.get_json() or {}).get("question", "")
    return jsonify(answer_with_sql(question))


@app.route("/rag", methods=["POST"])
def rag_endpoint():
    question = (request.get_json() or {}).get("question", "")
    return jsonify(answer_with_rag(question, rag_engine))


# ============================================================
# 11. START
# ============================================================

if __name__ == "__main__":
    print("API key loaded:", bool(os.environ.get("ANTHROPIC_API_KEY")))
    app.run(host="0.0.0.0", port=5000, debug=True)
