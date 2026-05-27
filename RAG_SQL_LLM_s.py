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
from flask import Flask, jsonify, request
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

def generate_sql(question: str, rag_context: str = "") -> Optional[str]:
    """Mapuje pytanie w języku naturalnym na zapytanie SQL."""

    q = question.lower()

    if ("ile" in q or "ilu" in q) and "krytyczn" in q and "maj" in q:
        return """
        SELECT COUNT(*) AS liczba_incydentow_krytycznych
        FROM incidents
        WHERE severity = 'critical'
          AND strftime('%Y-%m', created_at) = '2026-05';
        """

    if "najwięcej" in q and ("awari" in q or "incydent" in q or "serwer" in q):
        return """
        SELECT server_name, COUNT(*) AS failures
        FROM incidents
        GROUP BY server_name
        ORDER BY failures DESC
        LIMIT 3;
        """

    if "otwart" in q or "open" in q:
        return """
        SELECT incident_id, server_name, severity, created_at
        FROM incidents
        WHERE status = 'open'
        ORDER BY created_at DESC;
        """

    if "ransomware" in q and ("ile" in q or "ilu" in q or "liczba" in q):
        return """
        SELECT COUNT(*) AS ransomware_incidents
        FROM incidents
        WHERE category = 'ransomware';
        """

    if "eskalow" in q or "escalat" in q:
        return """
        SELECT incident_id, server_name, severity, created_at
        FROM incidents
        WHERE status = 'escalated';
        """

    if "według ważności" in q or "severity" in q or "wagi" in q:
        return """
        SELECT severity, COUNT(*) AS liczba
        FROM incidents
        GROUP BY severity
        ORDER BY liczba DESC;
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

{f"Wynik SQL ({sql}):\n{sql_result}" if sql else "Brak danych SQL — odpowiadaj wyłącznie na podstawie dokumentów."}

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


def initialize_system() -> RagEngine:
    if not os.path.exists(DB_PATH):
        create_database()
    create_documents()
    return RagEngine(load_documents())


rag_engine = initialize_system()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": ANTHROPIC_MODEL})


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
    app.run(host="0.0.0.0", port=5000, debug=True)
