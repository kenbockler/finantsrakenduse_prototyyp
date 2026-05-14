"""Ollama HTTP klient (urllib) ja käsurea "ollama stop" abiline.

Eesmärk: edastab Ollama serveris töötavale LLM-ile "POST /api/chat" päringu ja tagastab vastuse teksti.
Sõnumid antakse Ollama formaadis: [{"role": "system"|"user"|"assistant", "content": "..."}].
Lisaks: "sulge_ollama_mudel" kutsub Ollama CLI-d "ollama stop", et vabastada GPU/RAM pärast sessiooni.
"""

# Käesolev kood on loodud autori ja tehisintellekti koostöös loovkoodi meetodil.
# Arendusprotsessis kasutati Anthropic Claude Opus 4.7 suurt keelemudelit.
# Autor määratles arenduse eesmärgid, jagas ülesanded väiksemateks osadeks,
# hindas pakutud lahendusi, testis koodi ning tegi vajalikud parandused.
# Suurt keelemudelit kasutati süntaksi kontrollimiseks, lahendusvariantide
# pakkumiseks, koodi struktuuri parandamiseks, autori kirjutatud koodi
# kiiremaks ja efektiivsemaks muutmiseks ning kommentaaride sõnastamiseks.
# Tehisintellekti kasutamine ei asendanud autori sisulist otsustamist,
# kontrollimist ega vastutust loodud lahenduse eest.

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request


# Saadab Ollama chat-päringu ja tagastab mudeli vastuse teksti.
# HTTP-, võrgu- või vastusekuju vea korral tõstab selge RuntimeError-i.
# Tagastab mittetühja vastuse sisu.
def ollama_chat(
    base_url: str,
    mudel: str,
    messages: list[dict],
    temperature: float = 0.2,
    timeout_s: int = 300,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": mudel,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=int(timeout_s)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"Ollama HTTP viga: {e.code} {e.reason}. {body}") from e
    except Exception as e:
        raise RuntimeError(f"Ollama päring ebaõnnestus: {e}") from e

    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ootamatu Ollama vastuskuju: {data!r}")
    return content


# Peatab Ollama mudeli, kui see on laaditud.
# Mudeli peatamise ebaõnnestumine ei katkesta programmi tööd.
def sulge_ollama_mudel(mudel: str) -> None:
    m = (mudel or "").strip()
    if not m:
        return
    try:
        subprocess.run(
            ["ollama", "stop", m],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return
    except Exception:
        return
