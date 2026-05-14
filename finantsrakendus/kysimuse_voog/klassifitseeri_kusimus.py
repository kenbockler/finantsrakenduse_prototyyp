"""Klassifitseerib kasutaja küsimuse SQL-malli järgi (LLM kaudu) ja täidab parameetrid.

Eesmärk:
    Annab LLM-ile kasutaja küsimuse + 5 SQL-malli kirjelduse + tänase kuupäeva.
    LLM tagastab JSON-i {"malli_id": ..., "parameetrid": {...}}. Kood valideerib
    JSON-i, kontrollib parameetrite tüüpi ja vahemikku ning tagastab valmis SQL-i.
    Toetatud parameetri-tüübid: int / float (vahemikuga), kuupaev (YYYY-MM-DD,
    vahemikuga), valik (suletud lubatud väärtuste hulgast).
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
from datetime import date, timedelta

from finantsrakendus.kysimuse_voog.ollama_klient import ollama_chat
from finantsrakendus.seaded.loe_seaded import Seaded
from finantsrakendus.seaded.loe_sql_mallid import Parameeter, SqlMall


@dataclass(frozen=True)
class KlassifikatsiooniTulemus:
    mall: SqlMall
    parameetrid: dict[str, object]
    valmis_sql: str


# Eraldi erind, kui LLM ei suuda kasutaja küsimust ühelegi mallile sobitada.
class KusimusEiSobi(ValueError):
    pass


# Tagastab antud kuu viimase päeva
# Järgmise kuu esimese päeva miinus üks päev
# TAGASTAB: date
def _kuu_viimane_paev(d: date) -> date:
    if d.month == 12:
        return d.replace(day=31)
    return d.replace(month=d.month + 1, day=1) - timedelta(days=1)


# Arvutab kõik suhtelised ajavahemikud absoluutseteks kuupäeva-paarideks (alates, kuni).
# LLM saab kasutada valmis kuupäevi, sest LLM ise ei suuda korrektselt seda teha.
# TAGASTAB: dict[str, tuple[alates_iso, kuni_iso]].
def _arvuta_ajavahemikud(tana: date) -> dict[str, tuple[str, str]]:
    eile = tana - timedelta(days=1)
    selle_n_alg = tana - timedelta(days=tana.weekday())
    eelm_n_lopp = selle_n_alg - timedelta(days=1)
    eelm_n_alg = eelm_n_lopp - timedelta(days=6)
    selle_kuu_alg = tana.replace(day=1)
    eelm_kuu_lopp = selle_kuu_alg - timedelta(days=1)
    eelm_kuu_alg = eelm_kuu_lopp.replace(day=1)
    selle_aasta_alg = tana.replace(month=1, day=1)
    eelm_aasta = tana.year - 1
    eelm_aasta_alg = date(eelm_aasta, 1, 1)
    eelm_aasta_lopp = date(eelm_aasta, 12, 31)
    return {
        "täna": (tana.isoformat(), tana.isoformat()),
        "eile": (eile.isoformat(), eile.isoformat()),
        "see nädal": (selle_n_alg.isoformat(), tana.isoformat()),
        "eelmine nädal": (eelm_n_alg.isoformat(), eelm_n_lopp.isoformat()),
        "see kuu": (selle_kuu_alg.isoformat(), tana.isoformat()),
        "eelmine kuu": (eelm_kuu_alg.isoformat(), eelm_kuu_lopp.isoformat()),
        "see aasta": (selle_aasta_alg.isoformat(), tana.isoformat()),
        "eelmine aasta": (eelm_aasta_alg.isoformat(), eelm_aasta_lopp.isoformat()),
    }


# Vormistab ühe parameetri kirjelduse LLM-i süsteemiviiba jaoks
# TAGASTAB: stringi
def _parameetri_kirjeldus(p: Parameeter) -> str:
    if p.tuup in ("int", "float"):
        return f"{p.nimi} ({p.tuup}, vaikimisi {p.vaikimisi}, vahemik {p.min_vaartus}..{p.max_vaartus})"
    if p.tuup == "kuupaev":
        return (
            f"{p.nimi} (kuupaev YYYY-MM-DD, vaikimisi {p.vaikimisi}, "
            f"vahemik {p.min_vaartus}..{p.max_vaartus})"
        )
    # valik
    lubatud = list(p.lubatud_vaartused or ())
    return f"{p.nimi} (valik {lubatud}, vaikimisi {p.vaikimisi!r})"


# Koostab LLM-ile süsteemiviiba: tänane kuupäev + andmestiku ajaraam + mallid + JSON-vorming.
# "andmete_vahemik" on (alates_iso, kuni_iso) - reaalsed esimese ja viimase tehingu kuupäevad.
# TAGASTAB: süsteemiviiba tekst.
def _systeemiviiba(
    mallid: tuple[SqlMall, ...], tana: date, andmete_vahemik: tuple[str, str]
) -> str:
    andmed_alates, andmed_kuni = andmete_vahemik
    read = [
        f"Tänane kuupäev on {tana.isoformat()}.",
        f"Andmestik sisaldab tehinguid ajavahemikul {andmed_alates} kuni {andmed_kuni}.",
        "Sa oled klassifitseerija. Kasutaja küsib oma finantstehingute kohta.",
        "Vali ALLPOOL toodud mallidest TÄPSELT ÜKS, mis sobib kõige paremini.",
        "",
        "Lubatud mallid (id ja kirjeldus):",
    ]
    for m in mallid:
        params_kirj = "; ".join(_parameetri_kirjeldus(p) for p in m.parameetrid)
        read.append(f"- id={m.id!r}; {m.kirjeldus} Parameetrid: {params_kirj}")
    read.append("")
    read.append("Ajavahemiku reeglid (parameetrid 'alates_kuupaev' ja 'kuni_kuupaev'):")
    read.append(
        f"- Kui kasutaja ei maini ajavahemikku, kasuta andmestiku piire: "
        f"alates_kuupaev={andmed_alates}, kuni_kuupaev={andmed_kuni}."
    )
    read.append("")
    read.append("Suhteliste ajavahemike VALMIS KUUPÄEVAD - kasuta neid täpselt sellisel kujul")
    read.append("(ÄRA arvuta ise, ÄRA muuda aastat, KOPEERI siit):")
    vahemikud = _arvuta_ajavahemikud(tana)
    for nimi, (alates, kuni) in vahemikud.items():
        read.append(f"- '{nimi}': alates={alates}, kuni={kuni}")
    read.append("")
    read.append("Konkreetsed kuud arvuta kalendrijärgi:")
    read.append(f"- 'mai 2024' -> alates=2024-05-01, kuni=2024-05-31")
    read.append(f"- Kuu nimi ilma aastata (näit 'mai', 'jaanuar') -> selle aasta antud kuu.")
    read.append(f"  Kuna tänane kuupäev on {tana.isoformat()}, kasutab 'mai' aastat {tana.year}.")
    read.append("")
    read.append("Tagasta AINULT JSON kujul:")
    read.append('{"malli_id": "<üks ülaltoodud id>", "parameetrid": {"<nimi>": <väärtus>, ...}}')
    read.append("Kuupäevad ja valikud antakse stringidena, arvud arvudena.")
    read.append('Kui ükski mall ei sobi, tagasta {"malli_id": null, "parameetrid": {}}.')
    read.append("Ära lisa selgitusi ega koodibloki tähiseid - ainult puhas JSON.")
    return "\n".join(read)


# Eraldab LLM vastusest JSON-objekti (esimene "{...}" sulgudega blokk).
# Vea korral tõstab ValueError.
# TAGASTAB: parsitud dict.
def _parsi_json(vastus: str) -> dict:
    s = vastus.strip()
    # Eemaldame markdown koodibloki (```json ... ```) kui mudel selle lisab.
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Otsime esimese "{" kuni viimase "}" - kõige robustsem mustritolerantsi nimel.
    algus = s.find("{")
    lopp = s.rfind("}")
    if algus < 0 or lopp <= algus:
        raise ValueError(f"LLM vastus ei sisalda JSON-objekti: {vastus!r}")
    try:
        return json.loads(s[algus : lopp + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM vastuse JSON ei parsi: {e}; vastus: {vastus!r}") from e


# Valideerib arvulise (int/float) parameetri väärtuse ja tagastab puhastatud arvu.
# TAGASTAB: int või float.
def _valideeri_arv(p: Parameeter, toores) -> int | float:
    if toores is None:
        toores = p.vaikimisi
    try:
        n = float(toores)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Parameeter {p.nimi!r}: ei ole arv ({toores!r}).") from e
    min_v = float(p.min_vaartus) if p.min_vaartus is not None else float("-inf")
    max_v = float(p.max_vaartus) if p.max_vaartus is not None else float("inf")
    if not (min_v <= n <= max_v):
        raise ValueError(
            f"Parameeter {p.nimi!r}={n} pole vahemikus [{min_v}, {max_v}]."
        )
    return int(n) if p.tuup == "int" else n


# Valideerib kuupäeva-parameetri
# Kui LLM ei tagasta kuupäeva, kasutatakse DuckDB-st loetud tegeliku andmestiku algus- või lõppkuupäeva
# TAGASTAB: kuupäeva stringina YYYY-MM-DD
def _valideeri_kuupaev(
    p: Parameeter, toores, andmestiku_piir_kuupaev: str | None = None
) -> str:
    if toores is None:
        if andmestiku_piir_kuupaev is None:
            raise ValueError(
                f"Parameeter {p.nimi!r}: kuupäev puudub ja andmestiku piirkuupäeva pole antud."
            )
        return andmestiku_piir_kuupaev
    s = str(toores).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise ValueError(f"Parameeter {p.nimi!r}: kuupäev pole YYYY-MM-DD ({toores!r}).")
    try:
        date.fromisoformat(s)
    except ValueError as e:
        raise ValueError(f"Parameeter {p.nimi!r}: kuupäev pole kehtiv ({toores!r}).") from e
    if p.min_vaartus and s < p.min_vaartus:
        raise ValueError(f"Parameeter {p.nimi!r}={s} on enne lubatud miinimumi {p.min_vaartus}.")
    if p.max_vaartus and s > p.max_vaartus:
        raise ValueError(f"Parameeter {p.nimi!r}={s} on pärast lubatud maksimumi {p.max_vaartus}.")
    return s


# Valideerib valiku-parameetri - väärtus peab kuuluma lubatud väärtuste hulka
# TAGASTAB: lubatud string.
def _valideeri_valik(p: Parameeter, toores) -> str:
    if toores is None:
        return p.vaikimisi
    s = str(toores).strip()
    lubatud = p.lubatud_vaartused or ()
    if s not in lubatud:
        raise ValueError(
            f"Parameeter {p.nimi!r}={s!r} pole lubatud väärtus {list(lubatud)!r}."
        )
    return s


# Valideerib ühe parameetri vastavalt tüübile
# TAGASTAB: int / float / str (sõltuvalt tüübist).
def _valideeri_parameeter(p: Parameeter, toores, andmete_vahemik: tuple[str, str]) -> object:
    if p.tuup in ("int", "float"):
        return _valideeri_arv(p, toores)
    if p.tuup == "kuupaev":
        andmestiku_piir_kuupaev: str | None = None
        if p.nimi == "alates_kuupaev":
            andmestiku_piir_kuupaev = andmete_vahemik[0]
        elif p.nimi == "kuni_kuupaev":
            andmestiku_piir_kuupaev = andmete_vahemik[1]
        return _valideeri_kuupaev(p, toores, andmestiku_piir_kuupaev)
    return _valideeri_valik(p, toores)


# Asendab SQL-mallis kohatäited valideeritud parameetrite väärtustega.
# TAGASTAB: valmis SQL string
def _taida_sql(mall: SqlMall, parameetrid: dict[str, object]) -> str:
    return mall.sql.format(**parameetrid)


# Klassifitseerib kasutaja küsimuse
# Kui LLM ei suuda valida (malli_id == null), saab vastava teate.
# TAGASTAB: KlassifikatsiooniTulemus (mall, valideeritud parameetrid, valmis SQL).
def klassifitseeri_kusimus(
    seaded: Seaded,
    mallid: tuple[SqlMall, ...],
    kusimus: str,
    andmete_vahemik: tuple[str, str],
) -> KlassifikatsiooniTulemus:
    sonumid = [
        {"role": "system", "content": _systeemiviiba(mallid, date.today(), andmete_vahemik)},
        {"role": "user", "content": kusimus},
    ]
    vastus = ollama_chat(
        base_url=seaded.ollama_base_url,
        mudel=seaded.llm_mudel,
        messages=sonumid,
        temperature=0.1,
        timeout_s=60,
    )
    obj = _parsi_json(vastus)

    malli_id = obj.get("malli_id")
    if not malli_id:
        raise KusimusEiSobi(
            "Vabandust, see küsimus jääb prototüübi 5 toetatud küsimusetüübist välja. "
            "Proovi sõnastust muuta."
        )

    leitud = next((m for m in mallid if m.id == malli_id), None)
    if leitud is None:
        lubatud = [m.id for m in mallid]
        raise ValueError(f"LLM tagastas tundmatu malli_id={malli_id!r}; lubatud: {lubatud}.")

    toored_params = obj.get("parameetrid") or {}
    if not isinstance(toored_params, dict):
        raise ValueError(f"LLM tagastas 'parameetrid' vales kujus: {toored_params!r}")

    valideeritud: dict[str, object] = {}
    for p in leitud.parameetrid:
        valideeritud[p.nimi] = _valideeri_parameeter(p, toored_params.get(p.nimi), andmete_vahemik)

    valmis_sql = _taida_sql(leitud, valideeritud)
    return KlassifikatsiooniTulemus(mall=leitud, parameetrid=valideeritud, valmis_sql=valmis_sql)
