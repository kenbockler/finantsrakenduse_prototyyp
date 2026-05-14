"""Sarnasusotsing: lähtetehing -> Chroma top-K -> DuckDB read -> LLM kokkuvõte.

Eesmärk:
    Pärast SQL-faasi kuvab kasutajaliides vastusetabeli rea numbritega ja
    kasutaja valib järelotsingu jaoks rea. Sõltuvalt malli tüübist on kaks teed:
    1) Detailmall ("suurimad_kulutused"): igal real on konkreetne "tehingu_id".
       Loeme DuckDB-st selle rea "otsingu_tekst", embedime selle ja küsime Chromast
       top-K sarnaseid tehinguid. Lähte-tehing ise filtreeritakse vastusest välja.
    2) Agregeeritud mallid ("korduvad_maksed", "topeltmaksed",
       "sagedased_vaikesed_maksed", "pusimakse_hinnatous"): rida kirjeldab
       vastaspoolt (mitte üksiktehingut). Embedime jooksvalt "vastaspool_nimi"
       teksti samal kujul, nagu DuckDB-s veerg "otsingu_tekst" on normaliseeritud
       (lower + ainult tähed/tühikud), ja küsime Chromast sarnaseid tehinguid.
       Sel juhul "ei ole" lähte-tehingut, mida filtreerida.

    Mõlemal juhul loeme Chroma "tehingu_id"-de järgi täisread DuckDB-st ja LLM
    koostab eestikeelse faktilise kokkuvõtte ainult antud andmete põhjal.

    Lõputöö metoodika seisukohast: see on ainus koht, kus SQL + Chroma + LLM
    kohtuvad ühes voos - täidab uurimisküsimust "SQL + semantiline + LLM".
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
from pathlib import Path
from typing import Any

import duckdb

from finantsrakendus.ettevalmistus.embeddi_tehingud import KOGUMIK
from finantsrakendus.ettevalmistus.laadi_tehingud_duckdb import TABEL_TEHINGUD
from finantsrakendus.kysimuse_voog.chroma_otsing import otsi as chroma_otsi
from finantsrakendus.kysimuse_voog.embedding_kusimus import embedi_yhe_teksti
from finantsrakendus.kysimuse_voog.ollama_klient import ollama_chat
from finantsrakendus.seaded.loe_seaded import Seaded

# Mitu sarnast tehingut Chroma-st päringuks võtame.
TOP_K = 20

# LLM-i temperature: 0.2 annab loomulikku eesti keelset teksti, ilma fantaseerimata.
TEMPERATUUR = 0.2

# Time-out sekundites - sarnasusotsingu kokkuvõte on lühike, kuid mudeli alglaadimine
# võib esimesel kõnel olla aeglane.
TIMEOUT_S = 120

# Sama normaliseerimine, mida kasutab "ettevalmistus/genereeri_otsingu_tekst.py"
# DuckDB UPDATE-is. Hoiame jooksva embeddingu sisendi konsistentsena indeksis
# olevate "otsingu_tekst" väärtustega - muidu samale vastaspoolele võib semantiline
# distance olla suurem kui peaks.
_LUBAMATUD_RE = re.compile(r"[^a-zäöõüšž ]+")
_TYHIKUD_RE = re.compile(r" +")


# Normaliseerib teksti samal kujul nagu DuckDB veerus "otsingu_tekst".
# TAGASTAB: normaliseeritud string (lowercase, ainult tähed ja üksikud tühikud).
def _normaliseeri(tekst: str) -> str:
    s = (tekst or "").lower()
    s = _LUBAMATUD_RE.sub(" ", s)
    s = _TYHIKUD_RE.sub(" ", s)
    return s.strip()


# Loeb DuckDB-st ühe tehingu "tehingu_id" järgi (sh "otsingu_tekst" embeddinguks
# ja täisinfo LLM-i lähte-tehingu kirjelduseks).
# TAGASTAB: dict täisinfoga või None, kui sellise id-ga rida pole.
def _loe_tehing_id_jargi(duckdb_tee: Path, tehingu_id: str) -> dict | None:
    with duckdb.connect(str(duckdb_tee), read_only=True) as con:
        res = con.execute(
            f"""
            SELECT tehingu_id, kuupaev, vastaspool_nimi, summa, selgitus, otsingu_tekst
            FROM {TABEL_TEHINGUD}
            WHERE tehingu_id = ?
            """,
            [tehingu_id],
        )
        veerud = [d[0] for d in (res.description or [])]
        rida = res.fetchone()
    if rida is None:
        return None
    return dict(zip(veerud, rida))


# Loeb DuckDB-st mitme "tehingu_id" kõik veerud korraga (parameetriline IN-list,
# DuckDB toetab "list parameter"-it ohutult ilma string-asenduseta).
# "tehingu_id" loetakse SQL-ist ainult selleks, et Chroma sarnasusjärjestust säilitada;
# kasutajale kuvamise jaoks eemaldame selle, sest see ei anna kasutajale infoväärtust.
# TAGASTAB: list[dict] järjekorras, nagu Chroma sarnasus tagastas (ilma "tehingu_id" võtmeta).
def _loe_tehingud_idide_jargi(duckdb_tee: Path, tehingu_idid: list[str]) -> list[dict]:
    if not tehingu_idid:
        return []
    with duckdb.connect(str(duckdb_tee), read_only=True) as con:
        res = con.execute(
            f"""
            SELECT tehingu_id, kuupaev, vastaspool_nimi, summa, selgitus
            FROM {TABEL_TEHINGUD}
            WHERE tehingu_id IN (SELECT * FROM UNNEST(?))
            """,
            [tehingu_idid],
        )
        veerud = [d[0] for d in (res.description or [])]
        read = res.fetchall()
    indeks = {str(r[0]): dict(zip(veerud, r)) for r in read}
    jarjestatud: list[dict] = []
    loetud_id: set[str] = set()
    for tid in tehingu_idid:
        if tid in loetud_id:
            continue
        rida = indeks.get(tid)
        if rida is None:
            continue
        loetud_id.add(tid)
        jarjestatud.append(rida)
    for rida in jarjestatud:
        rida.pop("tehingu_id", None)
    return jarjestatud


# LLM-i süsteemiviiba sarnasusotsingu kokkuvõtteks. Hoiame lühike ja faktiline,
# nagu samm 4 puhul (variant A) - räägi ainult andmetest, ära lisa metameetilist lauset.
# TAGASTAB: süsteemiviiba string.
def _systeemiviiba() -> str:
    read = [
        "Sa oled finantsanalüüsi assistent. Kasutaja valis vastusetabelist ühe rea (lähte-element) ja küsis semantiliselt sarnaseid tehinguid.",
        "Vasta eesti keeles 2-4 lausega, mis kirjeldavad faktiliselt seoseid.",
        "",
        "Vastuse kuju:",
        "- Esimene lause kirjeldab lähte-elementi (kui see on üksik tehing - vastaspool, summa, kuupäev; kui see on agregaat - ainult vastaspoole nimi).",
        "- Edasi: mis seob sarnased tehingud lähte-elemendiga (sama vastaspool, sarnane teenuseliik, sarnased märksõnad selgituses).",
        "- Kui näed silmatorkavat mustrit (näiteks kõik samalt vastaspoolelt, summad varieeruvad palju), märgi see.",
        "",
        "Summad vormista kujul '12,34 €' (kaks komakohta, koma kümnendmärk, € lõpus).",
        "Lõpeta vastus viimase fakti-lausega - ära lisa kokkuvõtvat märkust ('kõik info on tabelis' vms).",
    ]
    return "\n".join(read)


# Vormistab kasutaja-rolli sõnumi: lähte-element + sarnaste tehingute loend.
# TAGASTAB: kasutaja-sõnumi tekst.
def _kasutaja_sonum(lahte: dict[str, Any], sarnased: list[dict]) -> str:
    # Eemaldame tehnilised väljad: "otsingu_tekst" on sisemine vektorotsingu väli ja
    # "tehingu_id" ei anna kasutajale ega LLM-i selgitusele lisaväärtust.
    lahte_sisu = {k: v for k, v in lahte.items() if k not in {"otsingu_tekst", "tehingu_id"}}
    lahte_json = json.dumps(lahte_sisu, ensure_ascii=False, default=str, indent=2)
    sarnased_json = json.dumps(sarnased, ensure_ascii=False, default=str, indent=2)
    return (
        f"Lähte-element (kasutaja valis selle):\n{lahte_json}\n\n"
        f"Sarnased tehingud Chroma top-K järjekorras (lähimast kaugemani), {len(sarnased)} rida:\n"
        f"{sarnased_json}\n"
    )


# LLM-iga kokkuvõte sarnasusotsingust. Vea korral tagastab tühja stringi ja
# prinditakse hoiatus - kasutaja näeb ikkagi sarnaste tabelit.
# TAGASTAB: vastuse tekst või tühi string.
def _vasta_sarnastest(seaded: Seaded, lahte: dict, sarnased: list[dict]) -> str:
    sonumid = [
        {"role": "system", "content": _systeemiviiba()},
        {"role": "user", "content": _kasutaja_sonum(lahte, sarnased)},
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
        print(f"[hoiatus: sarnasusotsingu kokkuvõtet ei õnnestunud koostada: {e}]", flush=True)
        return ""


# Pea-funktsioon. Täpselt üks järgnevatest peab olema antud:
#   - "tehingu_id": detailmalli puhul (üksiktehing) - loeme DuckDB-st rea ja embedime
#     selle "otsingu_tekst"; lähte-tehingu ise filtreerime Chroma vastuses välja.
#   - "vastaspool_nimi": agregeeritud mallide puhul - embedime jooksvalt nime
#     (sama normaliseerimine kui "otsingu_tekst" indeksis); lähte-elementi DuckDB-s
#     ei lookupita ja Chroma vastusest midagi välja ei filtreerita.
# Kasutamine: kutsutakse "main_kysimuse_voog"-st pärast SQL-faasi.
# TAGASTAB: tuple (LLM kokkuvõte või tühi string, sarnaste tehingute list).
def tee_jarelotsing(
    seaded: Seaded,
    duckdb_tee: Path,
    chroma_kaust: Path,
    *,
    tehingu_id: str | None = None,
    vastaspool_nimi: str | None = None,
) -> tuple[str, list[dict]]:
    if (tehingu_id is None) == (vastaspool_nimi is None):
        raise ValueError(
            "tee_jarelotsing: täpselt üks parameeter peab olema antud "
            "('tehingu_id' VÕI 'vastaspool_nimi')."
        )

    # 1) Lähte-tekst + lähte-element LLM-i jaoks.
    if tehingu_id is not None:
        lahte = _loe_tehing_id_jargi(duckdb_tee, tehingu_id)
        if lahte is None:
            raise ValueError(f"Tehingut id={tehingu_id!r} ei leitud andmebaasist.")
        otsingu_tekst = (lahte.get("otsingu_tekst") or "").strip()
        if not otsingu_tekst:
            raise ValueError(
                f"Tehingul id={tehingu_id!r} puudub 'otsingu_tekst' "
                "(kontrolli, et ettevalmistus on läbi käinud)."
            )
        valja_filtreeritav_id: str | None = str(lahte["tehingu_id"])
    else:
        nimi = (vastaspool_nimi or "").strip()
        if not nimi:
            raise ValueError("Vastaspoole nimi on tühi.")
        otsingu_tekst = _normaliseeri(nimi)
        if not otsingu_tekst:
            raise ValueError(
                f"Vastaspoole nimest {nimi!r} ei jäänud normaliseerimisel midagi alles."
            )
        # Agregaadi puhul ei ole konkreetset lähte-tehingut, mida Chroma vastusest välja võtta.
        lahte = {"vastaspool_nimi": nimi}
        valja_filtreeritav_id = None

    # 2) Embedime lähte-teksti (CPU lazy load - esimene kõne ~30 s, edasi vahetu).
    print(f"[embedin lähte-teksti: {otsingu_tekst!r}]", flush=True)
    vektor = embedi_yhe_teksti(seaded.embedding_mudel, otsingu_tekst)

    # 3) Chroma top-K. Kui filtreerime lähte-tehingu välja, küsime ühe rohkem,
    # et kasutajale jääks ikka TOP_K rida.
    n_kysida = TOP_K + 1 if valja_filtreeritav_id else TOP_K
    print(f"[Chroma top-{TOP_K} sarnasusotsing ...]", flush=True)
    kandidaadid = chroma_otsi(
        chroma_kaust=chroma_kaust,
        kogumik=KOGUMIK,
        kusimus_embedding=vektor,
        top_k=n_kysida,
    )

    # 4) Eemaldame lähte-tehingu enda (kui see on määratletud); võtame kuni TOP_K ülejäänud.
    # Korduvad tehingu_id-d (Chroma või kogumiku anomaalia) eemaldame — muidu kuvatakse rohkem
    # ridu kui tegelikult erinevaid tulemusi.
    sarnaste_idid: list[str] = []
    loetud_id: set[str] = set()
    for k in kandidaadid:
        tid = str(k["tehingu_id"])
        if tid == valja_filtreeritav_id:
            continue
        if tid in loetud_id:
            continue
        loetud_id.add(tid)
        sarnaste_idid.append(tid)
        if len(sarnaste_idid) >= TOP_K:
            break
    if not sarnaste_idid:
        raise ValueError("Sarnaseid tehinguid ei leitud (Chroma kogumik on tühi).")

    # 5) DuckDB-st loeme sarnaste tehingute täisread.
    sarnased = _loe_tehingud_idide_jargi(duckdb_tee, sarnaste_idid)

    # 6) LLM kokkuvõte (variant A - faktiline).
    print("[koostan sarnasusotsingu kokkuvõtet ...]", flush=True)
    kokkuvote = _vasta_sarnastest(seaded, lahte, sarnased)

    return kokkuvote, sarnased
