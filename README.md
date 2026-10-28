# python-aznw-api

LLM-Service für den KI-Chat auf [benjamin-milcic.dev](https://benjamin-milcic.dev). Er beantwortet Fragen über
Benjamin Milčić, seine Arbeit, seine Projekte und seine Webseite – in jeder Sprache – mit **RAG** über eine
Wissensbasis und **Tool Calling** für Live-Daten (Rezepte, Gästebuch, Witze, Wetter, …).

```
Angular (aznw-routes)  ──►  NestJS (nest-aznw-api)  ──►  python-aznw-api  ──►  LiteLLM  ──►  OpenAI
  /chat                       /chat/stream                    │   ▲
                              Verlauf pro Session             │   └── Wissensbasis: Markdown, Portfolio, Lebenslauf
                                    ▲                         │       (aus aznw-routes, Embeddings über Cloudflare)
                                    └──── Tools holen Daten ──┘
```

- Nur das NestJS-Backend ruft diese API auf. Sie ist von außen nicht erreichbar.
- Alles, was mit dem LLM zu tun hat, liegt hier. Die Daten der Webseite bleiben bei NestJS, den Gesprächsverlauf
  hält NestJS im Speicher (es wird nichts dauerhaft gespeichert).

| Teil | Technik |
|---|---|
| API | FastAPI + Uvicorn, Python 3.12 mit [uv](https://docs.astral.sh/uv/) |
| LLM | OpenAI über LiteLLM, `gpt-5.4-mini` (Reasoning `low`), Spracheingabe mit Whisper bei Groq |
| Embeddings | Cloudflare-Worker `/embed` (Workers AI, `@cf/baai/bge-m3`, mehrsprachig) |
| Vektorsuche | im Speicher (die Wissensbasis hat nur wenige hundert Abschnitte), Embeddings als Cache-Datei |

## Lokal starten

Voraussetzung: [uv](https://docs.astral.sh/uv/getting-started/installation/) – es lädt das passende Python selbst.

```powershell
# einmalig
uv sync
copy .env.example .env      # dann .env ausfüllen (siehe unten)

# starten (http://127.0.0.1:8000)
uv run uvicorn app.main:app --reload --port 8000
```

Für den ganzen Chat lokal außerdem:

```powershell
# NestJS ohne Datenbank, nur mit den Chat-Modulen (Port 3000)
cd ..\nest-aznw-api
npm run start:chat-dev

# Frontend (http://localhost:4200/chat)
cd ..\aznw-routes
npm start
```

Die Tools greifen auch lokal auf die **Produktiv-Daten** zu (`NEST_API_URL=https://api.benjamin-milcic.dev`),
die Wissensbasis kommt aus dem lokalen Checkout `../aznw-routes` (`KNOWLEDGE_SOURCE`).

Tests (ohne Netzwerk und ohne LLM, die APIs werden gemockt):

```powershell
uv run pytest
```

Wissensbasis prüfen, z.B. nach dem Bearbeiten der Markdown-Dateien:

```powershell
uv run python -m app.rag.check ../aznw-routes
```

## Konfiguration (`.env`)

| Variable | Bedeutung |
|---|---|
| `OPENAI_API_KEY` | API-Key von [OpenAI](https://platform.openai.com/api-keys), am besten aus einem eigenen Projekt mit Prepaid-Guthaben |
| `LLM_MODEL` | LiteLLM-Modellname, Standard `openai/gpt-5.4-mini` |
| `REASONING_EFFORT` | Wie lange das Modell vor der Antwort nachdenkt, Standard `low` (Denk-Tokens zählen als Ausgabe) |
| `MAX_COMPLETION_TOKENS` | Obergrenze pro Modellaufruf (Antwort und Reasoning), Standard 4000 |
| `DAILY_BUDGET_USD` | Tagesbudget aller Besucher zusammen in US-Dollar, Standard 2. Ist es aufgebraucht, antwortet der Chat bis Mitternacht (UTC) mit `daily_limit`. `0` schaltet es ab |
| `GROQ_API_KEY` | API-Key von [Groq](https://console.groq.com/keys), nur für die Spracheingabe |
| `TRANSCRIPTION_MODEL_FAST` / `_QUALITY` | Whisper-Modelle der Spracheingabe (Standard: `large-v3-turbo` / `large-v3`) |
| `HISTORY_TOKEN_BUDGET` | Token-Budget des Gesprächsverlaufs, ab dem ältere Nachrichten zusammengefasst werden. Standard 4000 (dazu kommen pro Anfrage etwa 3.500 Tokens für Prompt, Tools und Wissen – der Verlauf geht bei jeder Frage und Tool-Runde mit und kostet jedes Mal) |
| `NEST_API_URL` | Adresse von nest-aznw-api. Lokal `https://api.benjamin-milcic.dev`, auf dem Server `http://127.0.0.1:3000` |
| `INTERNAL_API_KEY` | Gemeinsamer Schlüssel mit nest-aznw-api. Wird für interne NestJS-Endpoints (Witze) mitgeschickt und von `/chat/stream` und `/transcribe` verlangt |
| `KNOWLEDGE_SOURCE` | Wissensbasis: lokaler Pfad (`../aznw-routes`) oder `github:benjaminmilcic/aznw-routes@new` |
| `KNOWLEDGE_REFRESH_MINUTES` | Wie oft die Wissensbasis auf Änderungen geprüft wird, Standard 15 |
| `EMBED_URL` / `WORKER_SECRET` | Embedding-Endpoint des Cloudflare-Workers und sein Secret (Header `X-Worker-Secret`) |
| `RAG_TOP_K` | Anzahl Wissensabschnitte pro Frage, Standard 6 |
| `MAX_TOOL_ROUNDS` | Höchstzahl Tool-Runden pro Antwort, Standard 5 |

`.env` steht in `.gitignore`. Das Repository ist öffentlich – echte Werte dürfen nie committet werden.

## Wie es funktioniert

- **Wissensbasis (RAG):** `app/rag/` lädt die Markdown-Dateien aus `aznw-routes/ai-chat-knowledge-base`, die
  Portfolio-Projekte (`portfolio.json` mit den englischen Texten) und den englischen Lebenslauf (PDF). Jeder
  `## `-Abschnitt wird ein eigener Chunk, mit Seitentitel, Link und Quellcode-Links davor. Die Embeddings werden nach
  dem Hash des Textes gecacht (`.state/` lokal, `/var/lib/python-aznw-api` auf dem Server), geänderte Dateien kosten
  also nur ihre neuen Abschnitte. Auf dem Server wird GitHub alle 15 Minuten auf Änderungen geprüft – pushen genügt.
- **Suche:** Zu jeder Frage die ähnlichsten Abschnitte (Kosinus-Ähnlichkeit, bge-m3 ist mehrsprachig – eine
  kroatische Frage findet den englischen Text). Bei Rückfragen wird zusätzlich mit der vorigen Frage und Antwort
  gesucht.
- **Prompt** (`app/chat.py`): nur Fragen zu Benjamin und seiner Webseite, nur Fakten aus Wissen und Tools, immer
  E-Mail und Kontaktformular bei Kontakt oder Wissenslücken, Antwort in der Sprache der Frage. Dazu eine Liste aller
  Seiten mit Links (aus dem Frontmatter).
- **Links** (`app/links.py`): Links zur Webseite sind relative Pfade, die das Frontend mit dem Angular-Router öffnet.
  Erfundene Pfade (`/about`, `/#kontakt`, `/gimmicks/wetter`) korrigiert `LinkFixer` direkt im Token-Stream.
- **Tools** (`app/tools/`): holen Live-Daten bei jeder Frage, z.B. neue Gästebucheinträge sofort.
- **Verlauf:** Wird der Gesprächsverlauf größer als `HISTORY_TOKEN_BUDGET`, fasst das Modell alles bis auf die letzte
  Frage und Antwort in höchstens 150 Wörtern zusammen (Event `summary`). NestJS ersetzt die alten Nachrichten in
  seinem Verlauf durch die Zusammenfassung, das Gespräch geht mit „Zusammenfassung + neue Nachrichten“ weiter. Danach
  liegt der Verlauf weit unter dem Budget – erneut zusammengefasst wird erst, wenn er wieder zu groß ist (bei
  typischen Antworten etwa alle 8 Fragen). Die alte Zusammenfassung fließt dann in die neue ein.
- **Limits:** Ist das Minutenlimit des OpenAI-Projekts erreicht, wartet die API die genannte Zeit und versucht es
  erneut. Klappt es trotzdem nicht, kommt ein Fehler mit `code: "rate_limit"`, das Frontend zeigt „Probier es in einer
  Minute nochmal“. Ist das Guthaben bei OpenAI aufgebraucht (`insufficient_quota`), kommt `code: "error"`.
- **Tagesbudget** (`app/budget.py`): LiteLLM berechnet die Kosten jedes Modellaufrufs (zwischengespeicherte Eingabe
  zum günstigeren Preis). Die Summe des Tages (UTC) steht in `daily-budget.json` im State-Verzeichnis und übersteht
  damit Neustarts. Ist `DAILY_BUDGET_USD` erreicht, kommt vor jedem Modellaufruf `code: "daily_limit"` mit den Minuten
  bis Mitternacht UTC. Das fängt Besucher ab, die die Limits pro IP in NestJS umgehen.

## Endpoints

### `POST /chat/stream`

Header `x-internal-api-key`, Body `{"message": "...", "history": [{role, content}], "location": {latitude, longitude}}`
(`location` optional, nur wenn der Besucher seinen Standort geteilt hat). Antwort als Server-Sent Events:

| Event | Bedeutung |
|---|---|
| `{"summary": {...}}` | ältere Nachrichten wurden zusammengefasst (zuerst) |
| `{"tool": "search_recipes"}` | ein Tool wird gerade aufgerufen |
| `{"token": "..."}` | Textstück der Antwort (Markdown) |
| `{"action": "request_location"}` | das Frontend soll anbieten, den Standort zu teilen |
| `{"usage": {inputTokens, outputTokens, totalTokens, cost}}` | Tokens und der Preis, den sie in einem Bezahlplan hätten |
| `[DONE]` | Ende |
| `{"error": "...", "code": "rate_limit" \| "error"}` | Fehler |

### `POST /transcribe`

Spracheingabe: multipart `audio` und `quality` (`fast` | `quality`) → `{"text": "..."}`.

### `GET /health`

Status, Commit, Modell, Wissensbasis (`knowledgePages`, `knowledgeChunks`, `knowledgeError`) und Tools.

## Tools

| Tool | Datenquelle | Verhalten im Chat |
|---|---|---|
| `get_availability` | NestJS `GET /availability` | ob und ab welchem Quartal Benjamin verfügbar ist |
| `search_recipes` | NestJS `GET /recipes` | Rezepte suchen oder einen Vorschlag machen, mit Link und Foto |
| `get_random_joke` | NestJS `GET /jokes/random` (intern) | zufälliger Witz in der Sprache der Frage. Auf der Webseite brauchen die Witze ein Login, im Chat nicht |
| `find_guestbook_entries` | NestJS `GET /guestbook` | fragt erst, ob Einträge gezeigt werden sollen, dann nur Namen und Datum |
| `get_guestbook_entry` | NestJS `GET /guestbook` | ganzer Eintrag mit Bildern, nur auf Nachfrage |
| `get_weather` | Open-Meteo (ohne Key), Ortsname über NestJS | aktuelles Wetter am geteilten Standort oder in einer genannten Stadt |
| `get_moorhuhn_highscores` | NestJS `GET /moorhuhn` | Highscore-Liste als Tabelle |

Ein neues Tool ist eine Datei in `app/tools/` mit `SCHEMA` (Beschreibung für das Modell) und `async def run(args,
context)`, eingetragen in `app/tools/__init__.py`. Wann das Modell es benutzen soll, steht im System-Prompt in
`app/chat.py`, die Ladeanzeige („Sucht Rezepte …“) in den i18n-Dateien von aznw-routes unter `chat.tools`.

## Deployment

Jeder Push auf `main` startet `.github/workflows/deploy.yml`:

1. Tests (`uv run pytest`)
2. Prüfen, ob `uv` auf dem Server installiert ist
3. Code per `rsync` auf den Server kopieren (mit dem Commit in `BUILD_SHA`)
4. `uv sync --frozen --no-dev` auf dem Server
5. Dienst neu starten und Health-Check, bis `/health` den neuen Commit meldet – das prüft zugleich, dass der
   Dienst-Benutzer `pyapi` seinen Code lesen und starten kann

### GitHub-Secrets (Repository → Settings → Secrets and variables → Actions)

| Secret | Inhalt |
|---|---|
| `DEPLOY_HOST` | Adresse des Servers |
| `DEPLOY_HOST_KEY` | öffentlicher Host-Key des Servers, z.B. `ssh-ed25519 AAAA…`, passend zu `DEPLOY_HOST` (einmal prüfen, dann eintragen) |
| `DEPLOY_SSH_KEY` | privater SSH-Key des Deploy-Benutzers |

Der Deploy-Benutzer steht als normale Variable im Workflow – als Secret würde GitHub jedes Vorkommen dieses kurzen
Wortes in den Logs durch `***` ersetzen.

### Einmalige Einrichtung auf dem Server

```bash
# Dienst-Benutzer und Verzeichnis (setgid, damit neue Dateien der Gruppe pyapi gehören)
sudo useradd --system --no-create-home --shell /usr/sbin/nologin pyapi
sudo install -d -o deploy -g pyapi -m 2750 /opt/python-aznw-api

# uv systemweit nach /usr/local/bin (lädt das Python 3.12 für das Projekt selbst ins Projektverzeichnis)

# Secrets – außerhalb des Deploy-Verzeichnisses, nur für root lesbar
sudo install -d -m 700 /etc/python-aznw-api
sudo nano /etc/python-aznw-api/api.env     # OPENAI_API_KEY, GROQ_API_KEY, INTERNAL_API_KEY, WORKER_SECRET
sudo chmod 600 /etc/python-aznw-api/api.env

# sudo-Rechte für den Deploy-Benutzer
sudo nano /etc/sudoers.d/93-deploy-python    # Inhalt siehe unten
sudo chmod 440 /etc/sudoers.d/93-deploy-python
sudo visudo -c -f /etc/sudoers.d/93-deploy-python   # unbedingt prüfen: eine kaputte Datei legt sudo lahm
```

Inhalt von `/etc/sudoers.d/93-deploy-python`:

```
deploy ALL=(root) NOPASSWD: /usr/bin/systemctl start python-aznw-api, \
                            /usr/bin/systemctl stop python-aznw-api, \
                            /usr/bin/systemctl restart python-aznw-api, \
                            /usr/bin/systemctl reset-failed python-aznw-api
```

sudo vergleicht die komplette Kommandozeile – der Workflow ruft die Befehle deshalb genau so auf, ohne
`.service`-Endung und ohne `|| true` (eine fehlende Regel soll auffallen, nicht verschluckt werden).
`systemctl status` und `is-active` brauchen kein sudo, der Workflow ruft sie ohne auf.

Nach dem **ersten** Deploy (Code liegt in `/opt/python-aznw-api`) den Dienst einrichten:

```bash
sudo cp /opt/python-aznw-api/deploy/python-aznw-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now python-aznw-api
curl http://127.0.0.1:8000/health
```

Beim allerersten Lauf schlägt der Schritt „Restart service“ fehl, weil der Dienst noch nicht existiert. Nach der
Einrichtung den Workflow einfach erneut starten (Actions → „Run workflow“).

Die API hört nur auf `127.0.0.1:8000` und bekommt **keinen** Apache-vHost. NestJS erreicht sie direkt auf dem Server.
