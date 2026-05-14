"""Ettevalmistuse peavoog: CSV -> DuckDB -> otsingu_tekst -> Chroma -> Ollama kontroll.

Käivitamine:
    source .venv_py_312/bin/activate
    .venv_py_312/bin/python -m finantsrakendus.ettevalmistus.main_ettevalmistus
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
from urllib.error import URLError
from urllib.request import Request, urlopen

import duckdb

from finantsrakendus.ettevalmistus.embeddi_tehingud import arvuta_ja_salvesta
from finantsrakendus.ettevalmistus.genereeri_otsingu_tekst import taida_otsingu_tekst
from finantsrakendus.ettevalmistus.laadi_tehingud_duckdb import (
    JUURKAUST,
    TABEL_TEHINGUD,
    laadi_tehingud_kettale,
)
from finantsrakendus.seaded.loe_seaded import Seaded, loe_seaded


@dataclass(frozen=True)
class AndmeTeed:
    csv_tee: Path
    duckdb_tee: Path
    chroma_kaust: Path


# Loeb seaded ja kontrollib, et CSV fail on olemas.
# TAGASTAB: Seaded objekt (csv_tee on absoluutne Path).
def loe_seaded_ja_kontrolli_csv() -> Seaded:
    seaded = loe_seaded()
    csv_tee = seaded.csv_tee.resolve()
    if not csv_tee.is_file():
        raise FileNotFoundError(f"CSV puudub: {csv_tee}")
    return seaded


# Arvutab DuckDB faili tee (CSV nime järgi) ja fikseeritud Chroma püsikausta tee.
# TAGASTAB: AndmeTeed (absoluutsed teed).
def arvuta_sihtkohad(csv_tee: Path) -> AndmeTeed:
    nimi = csv_tee.stem
    duckdb_tee = (JUURKAUST / "finantsrakendus/andmed/duckdb_tehingud" / f"{nimi}.duckdb").resolve()
    chroma_kaust = (JUURKAUST / "finantsrakendus/andmed/chroma_tehingud").resolve()
    return AndmeTeed(csv_tee=csv_tee.resolve(), duckdb_tee=duckdb_tee, chroma_kaust=chroma_kaust)


# Laeb CSV DuckDB-sse kui .duckdb fail puudub; kui olemas, ei tee midagi.
# vaikne=True jätab "juba olemas" teate ära, aga päris töö (uus laadimine) prinditakse alati.
# TAGASTAB: sisestatud ridade arv (uus laadimine) või None (jäi vahele).
def tee_duckdb_kui_vajalik(csv_tee: Path, duckdb_tee: Path, vaikne: bool = False) -> int | None:
    if duckdb_tee.exists():
        if not vaikne:
            print(f"Ettevalmistus: DuckDB andmed juba olemas -> {duckdb_tee.name}", flush=True)
        return None
    duckdb_tee.parent.mkdir(parents=True, exist_ok=True)
    n = laadi_tehingud_kettale(csv_tee, duckdb_tee)
    print(f"Ettevalmistus: laaditud {n} rida -> {duckdb_tee}", flush=True)
    return n


# Loendab read, kus veerg "otsingu_tekst" on tühi või ainult tühikud.
# TAGASTAB: ridade arv.
def _loenda_tuhjad_otsingu_tekstid(duckdb_tee: Path) -> int:
    with duckdb.connect(str(duckdb_tee)) as con:
        r = con.execute(
            f"""
            SELECT count(*) FROM {TABEL_TEHINGUD}
            WHERE otsingu_tekst IS NULL OR trim(otsingu_tekst) = ''
            """
        ).fetchone()
    return int(r[0]) if r else 0


# Täidab veeru "otsingu_tekst", kui tühje ridu on; muidu jätab vahele.
# vaikne=True jätab "juba täidetud" teate ära; uuenduse korral prinditakse alati.
# TAGASTAB: uuendatud ridade arv või 0 kui midagi polnud vaja.
def tee_otsingu_tekst_kui_vajalik(duckdb_tee: Path, vaikne: bool = False) -> int:
    tuhjad = _loenda_tuhjad_otsingu_tekstid(duckdb_tee)
    if tuhjad == 0:
        if not vaikne:
            print("Ettevalmistus: otsingu_tekst juba täidetud, UPDATE-i pole vaja.", flush=True)
        return 0
    n = taida_otsingu_tekst(duckdb_tee)
    print(f"Ettevalmistus: otsingu_tekst uuendatud {n} rida -> {duckdb_tee.name}", flush=True)
    return n


# Indekseerib Chroma-sse, kui püsikaust puudub; kui olemas, ei tee midagi.
# vaikne=True jätab "juba olemas" teate ära; uue indekseerimise korral prinditakse alati.
# TAGASTAB: indekseeritud ridade arv või None (jäi vahele).
def tee_chroma_kui_vajalik(
    duckdb_tee: Path, chroma_kaust: Path, embedding_mudel: str, vaikne: bool = False
) -> int | None:
    if chroma_kaust.exists():
        if not vaikne:
            print(f"Ettevalmistus: Vektoriseeritud andmed juba olemas -> {chroma_kaust}", flush=True)
        return None
    chroma_kaust.parent.mkdir(parents=True, exist_ok=True)
    n = arvuta_ja_salvesta(
        duckdb_tee=duckdb_tee,
        chroma_kaust=chroma_kaust,
        mudeli_id=embedding_mudel,
    )
    print(f"Ettevalmistus: Chroma indekseeritud {n} rida -> {chroma_kaust}", flush=True)
    return n


# Kontrollib Ollama API kaudu, et mudel "llm_mudel" oleks alla tõmmatud.
# TAGASTAB: None; kui mudelit pole, tõstab RuntimeError.
def kontrolli_ollama_llm(ollama_base_url: str, llm_mudel: str) -> None:
    base = ollama_base_url.rstrip("/")
    url = f"{base}/api/tags"
    req = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
    except URLError as e:
        raise RuntimeError(
            f"Ollama ei vasta ({url}): {e}. Paigaldusjuhend: llm_mudelid/Ollama_mudelid.md"
        ) from e
    data = json.loads(raw)
    nimed = {str(m.get("name", "")).strip() for m in (data.get("models") or [])}
    if llm_mudel not in nimed:
        raise RuntimeError(
            f"Ollama mudel {llm_mudel!r} puudub. Olemas: {sorted(nimed)!r}. "
            f"Käivita: ollama pull {llm_mudel}. Mudelite nimekiri: llm_mudelid/Ollama_mudelid.md"
        )


# Käivitab kogu ettevalmistuse järjekorras (idempotentne: olemasolevaid faile ei kirjuta üle ilma vajaduseta).
# vaikne=True (kasutatakse "main.py"-st) varjab "juba olemas" / "valmis" teated;
# päris töö (uus laadimine, uus indekseerimine) prinditakse alati, et kasutaja näeks viivituse põhjust.
# TAGASTAB: AndmeTeed (CSV, DuckDB, Chroma absoluutsed teed).
def tee_ettevalmistus(vaikne: bool = False) -> AndmeTeed:
    seaded = loe_seaded_ja_kontrolli_csv()
    teed = arvuta_sihtkohad(seaded.csv_tee)
    tee_duckdb_kui_vajalik(teed.csv_tee, teed.duckdb_tee, vaikne=vaikne)
    tee_otsingu_tekst_kui_vajalik(teed.duckdb_tee, vaikne=vaikne)
    tee_chroma_kui_vajalik(teed.duckdb_tee, teed.chroma_kaust, seaded.embedding_mudel, vaikne=vaikne)
    kontrolli_ollama_llm(seaded.ollama_base_url, seaded.llm_mudel)
    if not vaikne:
        print("Ettevalmistus valmis: (DuckDB, otsingu_tekst, Chroma, Ollama LLM mudel).", flush=True)
    return teed


# Käsurea käivituspunkt: kasutatav iseseisvalt ettevalmistuse jooksutamiseks.
# TAGASTAB: None.
def main() -> None:
    tee_ettevalmistus()


if __name__ == "__main__":
    main()
