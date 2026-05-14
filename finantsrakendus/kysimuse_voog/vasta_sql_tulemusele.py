"""LLM-i abil lõppvastus - tabeli faktiline kokkuvõte kasutajale

Eesmärk:
    Pärast SQL-i päringut saadab see moodul tabeli (list[dict]) LLM-ile
    koos rangelt kitsendatud reeglitega: räägi AINULT andmetest tabelis,
    ära tee finantsnõuandmist, ära viita üldistele maailma-teadmistele.
    Tagastab eestikeelse loomuliku teksti, mille "main_kysimuse_voog"
    prindib tabeli ette, et kasutaja näeks faktilist kokkuvõtet enne
    tabeli printimist.
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
from datetime import date
from typing import Any

from finantsrakendus.kysimuse_voog.ollama_klient import ollama_chat
from finantsrakendus.seaded.loe_seaded import Seaded
from finantsrakendus.seaded.loe_sql_mallid import SqlMall

# LLM-i temperature: 0.2 annab veidi varieeruvuse (loomulik tekst), aga ei lähe
# fantaseerima nagu kõrge temperatuur (0.7+).
TEMPERATUUR = 0.2

# Time-out sekundites - lokaalne mudel võib pikemate vastuste puhul natuke aega võtta.
TIMEOUT_S = 120


# Koostab lühikese LLM-i süsteemiviiba, mis kirjeldab soovitud käitumist.
# Kuupäev ja andmestiku ajavahemik aitavad perioode õigesti tõlgendada.
# Tagastab süsteemiviiba stringina.
def _systeemiviiba(tana: date, andmete_vahemik: tuple[str, str]) -> str:
    andmed_alates, andmed_kuni = andmete_vahemik
    read = [
        f"Tänane kuupäev on {tana.isoformat()}.",
        f"Andmestik sisaldab tehinguid ajavahemikul {andmed_alates} kuni {andmed_kuni}.",
        "Sa oled finantsanalüüsi assistent. Sulle antakse SQL-tulemus tabelina ja kasutaja küsimus.",
        "Vasta eesti keeles 3-5 lausega, mis kirjeldavad faktiliselt tabeli sisu.",
        "",
        "Vastuse kuju:",
        "- Esimene lause ütleb, mitu rida leiti ja millises ajavahemikus (kui kuupäevad on tabelis).",
        "- Too esile 1-3 silmatorkavat rida (suurim, sagedasim vms) konkreetsete arvudega.",
        "- Kui sama vastaspool kordub 3+ korda või on muu nähtav muster, märgi see lühidalt.",
        "",
        "Kui parameetrites on 'suund', kasuta esimeses lauses sõnastust:",
        "- 'suund'='DESC' -> 'Siin on välja toodud N kõige suuremat kulutust ajavahemikus X kuni Y' (1 rida -> 'kõige suurim kulutus').",
        "- 'suund'='ASC'  -> 'Siin on välja toodud N kõige väiksemat kulutust ajavahemikus X kuni Y' (1 rida -> 'kõige väikseim kulutus').",
        "",
        "Aja sõnastus:",
        "- Räägi konkreetsete kuupäevade või kuu nimega ('mais 2026', '7. mail 2026', '2026-05-01 kuni 2026-05-09').",
        "- ÄRA kasuta hindavaid fraase nagu 'eelmine kuu', 'käesolev kuu', 'möödunud kuu', 'eelnev periood' - need viitavad valesti suhtelisele asukohale.",
        "",
        "Summad vormista kujul '12,34 €' (kaks komakohta, koma kümnendmärk, € lõpus).",
        "Lõpeta vastus viimase fakti-lausega - ära lisa kokkuvõtvat märkust ('kõik info on tabelis', 'rohkem ei tea' vms).",
    ]
    return "\n".join(read)


# Koostab LLM-ile sisendsõnumi koos küsimuse, malli, parameetrite ja tabeliga.
# Tabel antakse edasi JSON-ina; default=str teisendab nt Decimal- ja date-väärtused.
# Tagastab LLM-i sisendsõnumi tekstina.
def _kasutaja_sonum(
    kusimus: str,
    mall: SqlMall,
    parameetrid: dict[str, Any],
    read: list[dict],
    n_kokku: int,
) -> str:
    read_json = json.dumps(read, ensure_ascii=False, default=str, indent=2)
    return (
        f"Kasutaja küsimus: {kusimus!r}\n"
        f"Valitud mall: {mall.id}\n"
        f"Parameetrid: {parameetrid}\n"
        f"Tabelis kokku ridu: {n_kokku}\n"
        f"Allpool on kogu tabel JSON-is (read sama järjekorras kui kasutajale kuvatud väljundis):\n"
        f"{read_json}\n"
    )


# Kutsub Ollama kaudu LLM-i ja tagastab eestikeelse kokkuvõtte SQL-tulemusest.
# Andmestiku ajavahemik ja tänane kuupäev aitavad vältida eksitavaid suhtelisi ajaväljendeid.
# Kui LLM-kõne ebaõnnestub, tagastab tühja stringi; tabeli kuvamine saab jätkuda.
# Tagastab vastuse teksti või tühja stringi.
def vasta_sql_tulemusele(
    seaded: Seaded,
    kusimus: str,
    mall: SqlMall,
    parameetrid: dict[str, Any],
    tabel: list[dict],
    andmete_vahemik: tuple[str, str],
) -> str:
    n_kokku = len(tabel)
    sonumid = [
        {"role": "system", "content": _systeemiviiba(date.today(), andmete_vahemik)},
        {"role": "user", "content": _kasutaja_sonum(kusimus, mall, parameetrid, tabel, n_kokku)},
    ]
    try:
        return ollama_chat(
            base_url=seaded.ollama_base_url,
            mudel=seaded.llm_mudel,
            messages=sonumid,
            temperature=TEMPERATUUR,
            timeout_s=TIMEOUT_S,
        ).strip()
    except RuntimeError as e:
        # LLM-i viga ei katkesta töövoogu; kasutajale jääb tabel siiski nähtavaks.
        print(f"[hoiatus: kokkuvõtet ei õnnestunud koostada: {e}]", flush=True)
        return ""
