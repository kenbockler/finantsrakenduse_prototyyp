# Finantsprototüüp paigaldusjuhend ja kasutusjuhend

## Eeldused
- **Python 3.12**
- **Ollama** — Koos eelpaigaldatud keelemudeliga `ollama pull qwen3:30b-instruct`

## Paigaldamine (Linux/WSL)
```bash
# kui oled prototüübi juurkaustas, käivita:
python3.12 -m venv .venv_py_312
source .venv_py_312/bin/activate
python -m pip install -r requirements.txt
```
**Esmakäivitus** tõmbab Hugging Face’ist embedding-mudeli (salvestub kausta `embedding_mudelid`) ning loob DuckDB ja Chroma indeksi — võib võtta kaua aega ja vajab ruumi kettal.

## Kasutamine
Veebiliidese kasutamise juhend:

```bash
streamlit run kasutajaliides_streamlit/app.py
```

Andmed: `finantsrakendus/seaded/seaded.json` väli `csv_tee` viitab sünteetilisele CSV-failile.