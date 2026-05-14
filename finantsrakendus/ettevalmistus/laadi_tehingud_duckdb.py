"""Laeb pangaväljavõtte CSV-faili DuckDB andmebaasi.

Käivitamine:
    source .venv_py_312/bin/activate
    python -m finantsrakendus.ettevalmistus.laadi_tehingud_duckdb
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

import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import duckdb

# Konstandid
JUURKAUST = Path(__file__).resolve().parents[2]
CSV_FAILI_TEE = JUURKAUST / "finantsrakendus/andmed/OPUS_4.7_SYNT_ANDMED/synth_2024_2026.csv"
DUCKDB_FAILI_TEE = JUURKAUST / "finantsrakendus/andmed/duckdb_tehingud" / f"{CSV_FAILI_TEE.stem}.duckdb"
TABEL_TEHINGUD = "tehingud"

# Kontroll, et kohustuslik tekstiväli ei oleks tühi pärast tühikute eemaldamist.
# TAGASTAB: puhastatud string; tühja sisendi korral tõstab ValueError.
def _kohustuslik(s: str | None, nimi: str) -> str:
    t = (s or "").strip()
    if not t:
        raise ValueError(f"{nimi} puudub")
    return t

# CSV välja tühi string teisendatakse None-iks, DuckDB jaoks sobivaks.
# TAGASTAB: mitte-tühi string või None.
def _voi_none(s: str | None) -> str | None:
    t = (s or "").strip()
    return t or None

# CSV SUMMA stringi (punkt või koma kümnendikuna) teisendamine Decimal-iks DuckDB jaoks.
# TAGASTAB: Decimal väärtus; vigase sisendi korral tõstab ValueError.
def _parsi_summa(s: str) -> Decimal:
    try:
        return Decimal((s or "").strip().replace(",", "."))
    except InvalidOperation as e:
        raise ValueError(f"Vigane summa: {s!r}") from e

# CSV väljade teisendamine tuple-ks.
# TAGASTAB: 9-elemendiline tuple (viimane element otsingu_tekst on None, sest selle täidame hiljem eraldi tööriistaga).
def rida_tuplesiks(rida: dict[str, str]) -> tuple:
    return (
        _kohustuslik(rida.get("TEHINGU_ID"), "TEHINGU_ID"),
        datetime.strptime(_kohustuslik(rida.get("KUUPÄEV"), "KUUPÄEV"), "%d.%m.%Y").date(),
        _voi_none(rida.get("SAAJA_MAKSJA_KONTO")),
        _kohustuslik(rida.get("SAAJA_MAKSJA_NIMI"), "SAAJA_MAKSJA_NIMI"),
        _parsi_summa(_kohustuslik(rida.get("SUMMA"), "SUMMA")),
        _voi_none(rida.get("VIITENUMBER")),
        _kohustuslik(rida.get("SELGITUS"), "SELGITUS"),
        _kohustuslik(rida.get("VALUUTA"), "VALUUTA"),
        None,
    )

# Loeb CSV ja kirjutab tabeli "tehingud" DuckDB faili (vanem tabel kustutatakse, fail ja kaust luuakse vajadusel).
# TAGASTAB: sisestatud ridade arv.
def laadi_tehingud_kettale(csv_tee: Path, duckdb_tee: Path) -> int:
    with csv_tee.open(encoding="utf-8-sig", newline="") as f:
        kirjed = [rida_tuplesiks(r) for r in csv.DictReader(f, delimiter=";", quotechar='"')]
    if not kirjed:
        raise ValueError("CSV on tühi")

    duckdb_tee.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(duckdb_tee)) as con:
        con.execute(f"DROP TABLE IF EXISTS {TABEL_TEHINGUD}")
        con.execute(
            f"""
            CREATE TABLE {TABEL_TEHINGUD} (
                tehingu_id VARCHAR NOT NULL PRIMARY KEY,
                kuupaev DATE NOT NULL,
                vastaspool_konto VARCHAR,
                vastaspool_nimi VARCHAR NOT NULL,
                summa DECIMAL(18, 2) NOT NULL,
                viitenumber VARCHAR,
                selgitus VARCHAR NOT NULL,
                valuuta VARCHAR NOT NULL,
                otsingu_tekst VARCHAR
            )
            """
        )
        con.executemany(
            f"INSERT INTO {TABEL_TEHINGUD} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            kirjed,
        )
    return len(kirjed)


# Käsurea käivituspunkt; kasutab mooduli vaikimisi teid CSV_FAILI_TEE ja DUCKDB_FAILI_TEE.
# TAGASTAB: None; trükib konsooli laaditud ridade arvu ja DuckDB faili tee.
def main() -> None:
    n = laadi_tehingud_kettale(CSV_FAILI_TEE.resolve(), DUCKDB_FAILI_TEE.resolve())
    print(f"Laaditud {n} rida tabelisse {TABEL_TEHINGUD!r} -> {DUCKDB_FAILI_TEE}")


# Lubab faili käivitada otse, mitte ainult -m moodulina.
if __name__ == "__main__":
    main()
