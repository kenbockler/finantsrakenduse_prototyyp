"""Loeb SQL-mallide kirjelduse failist "finantsrakendus/seaded/sql_mallid.json".

Eesmärk: hoida SQL-mallid eraldi failis, et neid saaks lihtsalt kohandada,
    ilma koodi muutmata. Iga mall vastab ühele küsimusetüübile ja sisaldab
    parameetreid, mille väärtused tulevad LLM-ilt. Toetatud parameetri-tüübid:
    "int" / "float" - arvulised, vahemikuga;
    "kuupaev"       - string YYYY-MM-DD, vahemikuga;
    "valik"         - string suletud lubatud väärtuste hulgast (näit "DESC"/"ASC").
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
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Lubatud parameetrite tüübid mallides; iga tüüp tagab erineva turvakontrolli.
LUBATUD_TUUBID = {"int", "float", "kuupaev", "valik"}

# Kuupäeva-formaadi regex YYYY-MM-DD
KUUPAEV_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Parameeter:
    nimi: str
    tuup: str
    # "vaikimisi" hoitakse alati stringina; arvulised parsitakse hiljem float-iks/int-iks.
    vaikimisi: str
    # "int"/"float"/"kuupaev" jaoks: ülempiirid (kuupaeva puhul YYYY-MM-DD string).
    # "valik" jaoks: None (lubatud väärtuste loend on "lubatud_vaartused").
    min_vaartus: str | None = None
    max_vaartus: str | None = None
    # Ainult "valik" jaoks - lubatud stringide loend (näiteks ("DESC", "ASC")).
    lubatud_vaartused: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SqlMall:
    id: str
    kirjeldus: str
    sql: str
    parameetrid: tuple[Parameeter, ...]


# Tagastab JSON-faili tee
def _mallid_fail() -> Path:
    return Path(__file__).resolve().parent / "sql_mallid.json"


# Kontrollib, et string on YYYY-MM-DD ja vastab kalendrile (näiteks 2024-02-30 ei sobi)
# TAGASTAB: True või False
def _on_kuupaev(s: str) -> bool:
    if not KUUPAEV_RE.match(s):
        return False
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


# Loeb arvulise parameetri ("int" või "float") JSON-objektist
# TAGASTAB: Parameeter objekti
def _loe_arv_parameeter(id_: str, p_nimi: str, p_def: dict, tuup: str) -> Parameeter:
    try:
        vaikimisi = float(p_def["vaikimisi"])
        min_v = float(p_def["min"])
        max_v = float(p_def["max"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"sql_mallid.json: mall {id_!r} parameeter {p_nimi!r} - 'vaikimisi'/'min'/'max' peavad olema arvud."
        ) from e
    if not (min_v <= vaikimisi <= max_v):
        raise ValueError(
            f"sql_mallid.json: mall {id_!r} parameeter {p_nimi!r} - vaikimisi {vaikimisi} pole vahemikus [{min_v}, {max_v}]."
        )
    return Parameeter(
        nimi=p_nimi,
        tuup=tuup,
        vaikimisi=str(vaikimisi),
        min_vaartus=str(min_v),
        max_vaartus=str(max_v),
    )


# Loeb kuupäeva-parameetri
# TAGASTAB: Parameeter objekti
def _loe_kuupaev_parameeter(id_: str, p_nimi: str, p_def: dict) -> Parameeter:
    vaikimisi = str(p_def.get("vaikimisi") or "")
    min_v = str(p_def.get("min") or "")
    max_v = str(p_def.get("max") or "")
    for nimi, vaartus in (("vaikimisi", vaikimisi), ("min", min_v), ("max", max_v)):
        if not _on_kuupaev(vaartus):
            raise ValueError(
                f"sql_mallid.json: mall {id_!r} parameeter {p_nimi!r} {nimi} pole kehtiv YYYY-MM-DD: {vaartus!r}."
            )
    if not (min_v <= vaikimisi <= max_v):
        raise ValueError(
            f"sql_mallid.json: mall {id_!r} parameeter {p_nimi!r} - vaikimisi {vaikimisi} pole vahemikus [{min_v}, {max_v}]."
        )
    return Parameeter(
        nimi=p_nimi,
        tuup="kuupaev",
        vaikimisi=vaikimisi,
        min_vaartus=min_v,
        max_vaartus=max_v,
    )


# Loeb valiku-tüüpi parameetri
# TAGASTAB: Parameeter objekti
def _loe_valik_parameeter(id_: str, p_nimi: str, p_def: dict) -> Parameeter:
    lubatud_raw = p_def.get("lubatud")
    if not isinstance(lubatud_raw, list) or not lubatud_raw:
        raise ValueError(
            f"sql_mallid.json: mall {id_!r} parameeter {p_nimi!r} - 'lubatud' peab olema mittetühi list."
        )
    lubatud = tuple(str(x) for x in lubatud_raw)
    vaikimisi = str(p_def.get("vaikimisi") or "")
    if vaikimisi not in lubatud:
        raise ValueError(
            f"sql_mallid.json: mall {id_!r} parameeter {p_nimi!r} - vaikimisi {vaikimisi!r} pole lubatud väärtus {list(lubatud)!r}."
        )
    # Lubatud on ainult tähed, numbrid, alakriips
    for v in lubatud:
        if not re.fullmatch(r"[A-Za-z0-9_]+", v):
            raise ValueError(
                f"sql_mallid.json: mall {id_!r} parameeter {p_nimi!r} - lubatud väärtus {v!r} sisaldab keelatud sümboleid (lubatud A-Z, a-z, 0-9, _)."
            )
    return Parameeter(
        nimi=p_nimi,
        tuup="valik",
        vaikimisi=vaikimisi,
        lubatud_vaartused=lubatud,
    )


# Valideerib ühe mallikirjelduse JSON-objektist; kontrollib id, sql ja parameetrid.
# TAGASTAB: SqlMall.
def _loe_mall(obj: dict) -> SqlMall:
    id_ = str(obj.get("id") or "").strip()
    if not id_:
        raise ValueError("sql_mallid.json: mallil puudub mittetühi 'id'.")
    kirjeldus = str(obj.get("kirjeldus") or "").strip()
    if not kirjeldus:
        raise ValueError(f"sql_mallid.json: mallil {id_!r} puudub mittetühi 'kirjeldus'.")
    sql = str(obj.get("sql") or "").strip()
    if not sql:
        raise ValueError(f"sql_mallid.json: mallil {id_!r} puudub mittetühi 'sql'.")

    params_obj = obj.get("parameetrid") or {}
    if not isinstance(params_obj, dict):
        raise ValueError(f"sql_mallid.json: mallil {id_!r} on 'parameetrid' vales kujus (peab olema objekt).")

    params: list[Parameeter] = []
    for p_nimi, p_def in params_obj.items():
        if not isinstance(p_def, dict):
            raise ValueError(f"sql_mallid.json: mall {id_!r} parameeter {p_nimi!r} pole objekt.")
        tuup = str(p_def.get("tüüp") or "").strip()
        if tuup not in LUBATUD_TUUBID:
            raise ValueError(
                f"sql_mallid.json: mall {id_!r} parameeter {p_nimi!r} tüüp {tuup!r} ei ole lubatud (lubatud: {sorted(LUBATUD_TUUBID)})."
            )
        if tuup in ("int", "float"):
            params.append(_loe_arv_parameeter(id_, p_nimi, p_def, tuup))
        elif tuup == "kuupaev":
            params.append(_loe_kuupaev_parameeter(id_, p_nimi, p_def))
        else:
            params.append(_loe_valik_parameeter(id_, p_nimi, p_def))

    # Kontroll, et SQL sisaldab kõiki nimetatud parameetri-kohti "{p_nimi}".
    for p in params:
        if "{" + p.nimi + "}" not in sql:
            raise ValueError(
                f"sql_mallid.json: mall {id_!r} SQL ei sisalda kohatäidet {{{p.nimi}}}."
            )
    return SqlMall(id=id_, kirjeldus=kirjeldus, sql=sql, parameetrid=tuple(params))


# Loeb ja valideerib kõik mallid; kontrollib id-de unikaalsust.
# TAGASTAB: tuple SqlMall objektidest järjekorras nagu JSON-failis.
def loe_sql_mallid() -> tuple[SqlMall, ...]:
    tee = _mallid_fail()
    data = json.loads(tee.read_text(encoding="utf-8"))
    arr = data.get("mallid")
    if not isinstance(arr, list) or not arr:
        raise ValueError("sql_mallid.json: 'mallid' peab olema mittetühi list.")

    mallid = tuple(_loe_mall(o) for o in arr)
    nimed = [m.id for m in mallid]
    if len(set(nimed)) != len(nimed):
        raise ValueError(f"sql_mallid.json: mallide id-d peavad olema unikaalsed; said: {nimed}")
    return mallid
