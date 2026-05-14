"""SQL "SELECT"-päringute valideerimine ja käivitamine DuckDB-s.

Eesmärk: kasutaja küsimuse põhjal LLM-i poolt genereeritud SQL-päring peab
    enne käivitamist olema ohutu ja õige - ainult "SELECT", tabel "tehingud" ja kindla
    "LIMIT"-iga. See moodul teeb valideerimise ja read-only käivituse.

Käivitamine:
    Iseseisvalt ei käivitata; kasutatakse "main_kysimuse_voog.py"-st.
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

import re
from pathlib import Path

import duckdb

# Tabeli nimi, mille üle "SELECT" on lubatud (ainus tabel "tehingud").
TABEL = "tehingud"
# Turvalisuse mõttes määrame maksimaalse ridade arvu, mille SQL-päring tohib tagastada.
# 1000 rida on isikliku finantsrakenduse prototüübi jaoks mõistlik piir: katab paar
# aastat tehinguid. Kui SQL-mall ise "LIMIT"-it ei sisalda, lisatakse see piir automaatselt päringule.
MAX_RIDASID = 1000
# Keelatud SQL märksõnad (kõik, mis võivad andmeid muuta või lekitada).
KEELATUD_SONAD = {
    "insert", "update", "delete", "drop", "alter", "truncate",
    "attach", "detach", "copy", "export", "import",
    "pragma", "install", "load", "set", "use", "create", "replace",
    "vacuum", "checkpoint", "analyze",
}


# Eemaldab "--"-rea ja "/* ... */"-blokk-kommentaarid; vähendab tühikute kordusi.
# TAGASTAB: puhastatud SQL string.
def _puhasta_sql(sql: str) -> str:
    s = re.sub(r"--[^\n]*", " ", sql)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    return re.sub(r"\s+", " ", s).strip()


# Kontrollib SQL-i ohutust ja täiendab vajadusel "LIMIT"-iga.
# Reeglid: ainult üks "SELECT" lause, "FROM tehingud" peab esinema, keelatud sõnad puuduvad,
# "LIMIT" peab olema (kui pole, lisatakse "max_ridasid").
# TAGASTAB: puhastatud ja vajadusel täiendatud SQL stringi.
def valideeri_select(sql: str, max_ridasid: int = MAX_RIDASID) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL on tühi.")

    s = _puhasta_sql(sql)

    # Kontrollime, et tegu on ühe lausega
    if s.endswith(";"):
        s = s[:-1].rstrip()
    if ";" in s:
        raise ValueError("Lubatud on ainult üks SQL-lause (mitu semikooloniga eraldatud lauset on keelatud).")

    s_lower = s.lower()

    # Peab algama "select"-iga
    if not s_lower.startswith("select"):
        raise ValueError("Lubatud on ainult 'SELECT'-päring.")

    # Keelatud märksõnad
    for sona in KEELATUD_SONAD:
        if re.search(rf"\b{re.escape(sona)}\b", s_lower):
            raise ValueError(f"Keelatud SQL märksõna: {sona!r}.")

    # Peab viitama tabelile "tehingud".
    if not re.search(rf"\bfrom\s+{re.escape(TABEL)}\b", s_lower):
        raise ValueError(f"SQL peab sisaldama 'FROM {TABEL}'.")

    # "LIMIT" kontroll; kui puudub, lisame ülempiiri.
    limit_otsing = re.search(r"\blimit\s+(\d+)\b", s_lower)
    if limit_otsing is None:
        s = f"{s} LIMIT {int(max_ridasid)}"
    else:
        n = int(limit_otsing.group(1))
        if n > max_ridasid:
            raise ValueError(f"'LIMIT {n}' ületab lubatud ülempiiri {max_ridasid}.")

    return s


# Avab DuckDB faili read-only ühenduses, käivitab valideeritud SQL-i ja tagastab read.
# Iga rida on dict (veeru nimi -> väärtus).
# TAGASTAB: list[dict].
def kaivita_select(duckdb_tee: Path, sql: str, max_ridasid: int = MAX_RIDASID) -> list[dict]:
    if not duckdb_tee.is_file():
        raise FileNotFoundError(f"DuckDB fail puudub: {duckdb_tee}")

    puhas_sql = valideeri_select(sql, max_ridasid=max_ridasid)
    try:
        with duckdb.connect(str(duckdb_tee), read_only=True) as con:
            res = con.execute(puhas_sql)
            veerud = [d[0] for d in (res.description or [])]
            read_toored = res.fetchall()
    except duckdb.Error as e:
        raise RuntimeError(f"DuckDB viga SQL-i käivitamisel: {e}") from e

    return [dict(zip(veerud, rida)) for rida in read_toored]
