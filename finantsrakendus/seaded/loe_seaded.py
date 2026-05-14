"""Loeb rakenduse seaded failist "finantsrakendus/seaded/seaded.json".

Eesmärk: hoida prototüübi parameetrid ühes seadistuste failis, et vältida koodis kõvakoderingut.
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
from dataclasses import dataclass
from pathlib import Path


# Projektijuur: "loe_seaded.py" asub kaustas finantsrakendus/seaded/
JUURKAUST = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Seaded:
    ollama_base_url: str
    llm_mudel: str
    embedding_mudel: str
    csv_tee: Path


# Loeb ja valideerib "seaded.json"
# TAGASTAB: Seaded objekt; vea korral tõstab ValueError selge sõnumiga.
def loe_seaded() -> Seaded:
    tee = Path(__file__).resolve().parent / "seaded.json"
    data = json.loads(tee.read_text(encoding="utf-8"))

    ollama = str(data.get("ollama_base_url") or "").strip()
    llm = str(data.get("llm_mudel") or "").strip()
    emb = str(data.get("embedding_mudel") or "").strip()
    csv_str = str(data.get("csv_tee") or "").strip()
    if not ollama:
        raise ValueError("seaded.json: 'ollama_base_url' on kohustuslik ja mittetühi.")
    if not llm:
        raise ValueError("seaded.json: 'llm_mudel' on kohustuslik ja mittetühi.")
    if not emb:
        raise ValueError("seaded.json: 'embedding_mudel' on kohustuslik ja mittetühi.")
    if not csv_str:
        raise ValueError("seaded.json: 'csv_tee' on kohustuslik ja mittetühi (tee projektijuurest).")

    # csv_tee on seadetes suhteline tee projektijuurest; teisendame absoluutseks.
    csv_tee = (JUURKAUST / csv_str).resolve()

    return Seaded(
        ollama_base_url=ollama,
        llm_mudel=llm,
        embedding_mudel=emb,
        csv_tee=csv_tee,
    )
