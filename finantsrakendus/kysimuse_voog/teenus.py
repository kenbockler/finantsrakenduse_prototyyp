"""Küsimuse voo ärilogiika ilma terminali I/O-ta (CLI, Streamlit jms adapterid).

Teenusefunktsioonid ei kasuta print/input; valikuline progress-callback
võimaldab adapteritel kuvada sama etappide teateid mis varem konsoolis.
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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb

from finantsrakendus.ettevalmistus.laadi_tehingud_duckdb import TABEL_TEHINGUD
from finantsrakendus.ettevalmistus.main_ettevalmistus import AndmeTeed
from finantsrakendus.kysimuse_voog.duckdb_paring import kaivita_select
from finantsrakendus.kysimuse_voog.jarelotsing import tee_jarelotsing
from finantsrakendus.kysimuse_voog.klassifitseeri_kusimus import (
    KlassifikatsiooniTulemus,
    klassifitseeri_kusimus,
)
from finantsrakendus.kysimuse_voog.vasta_sql_tulemusele import vasta_sql_tulemusele
from finantsrakendus.seaded.loe_seaded import Seaded
from finantsrakendus.seaded.loe_sql_mallid import SqlMall

# Veerud, mis sisaldavad raha-summat (Decimal/float) - tabelis vormindame "12,34 €".
SUMMA_VEERUD = frozenset({"summa", "min_summa", "max_summa", "summa_kokku", "keskmine", "endine", "uus"})

# Veerud, mis sisaldavad protsenti - tabelis vormindame "5,0 %" (üks komakoht).
PROTSENDI_VEERUD = frozenset({"muutus_pct"})

# Tehnilised veerud, mida kasutajale ei kuvata ega LLM-ile ei edastata.
TEHNILISED_VEERUD = frozenset({"tehingu_id"})

# Mall, mille tulemusena tagastatakse üksikud (mitte-agregeeritud) tehingud.
DETAIL_MALLID = frozenset({"suurimad_kulutused"})


@dataclass(frozen=True)
class KusimuseTulemus:
    mall: SqlMall
    parameetrid: dict[str, object]
    read: list[dict]
    kokkuvote: str | None

    @property
    def on_detail_mall(self) -> bool:
        return self.mall.id in DETAIL_MALLID


@dataclass(frozen=True)
class JarelotsinguTulemus:
    kokkuvote: str | None
    read: list[dict]


def loe_andmete_ajavahemik(duckdb_tee: Path) -> tuple[str, str]:
    with duckdb.connect(str(duckdb_tee), read_only=True) as con:
        rida = con.execute(
            f"SELECT min(kuupaev), max(kuupaev) FROM {TABEL_TEHINGUD}"
        ).fetchone()
    if rida is None or rida[0] is None or rida[1] is None:
        raise RuntimeError(f"DuckDB tabel {TABEL_TEHINGUD!r} on tühi - ei saa leida andmestiku ajaraami.")
    return (rida[0].isoformat(), rida[1].isoformat())


def vorminda_lahter(veeru_nimi: str, vaartus: object) -> str:
    if vaartus is None:
        return ""
    if veeru_nimi in SUMMA_VEERUD:
        return f"{float(vaartus):.2f}".replace(".", ",") + " €"
    if veeru_nimi in PROTSENDI_VEERUD:
        return f"{float(vaartus):.1f}".replace(".", ",") + " %"
    return str(vaartus)


def vorminda_read_kuvatavateks(read: list[dict]) -> list[dict[str, str]]:
    """Tagastab read sõne-väärtustega; lisab veeru '#' (1..n); peidab tehnilised veerud."""
    if not read:
        return []
    nahtavad_veerud = [v for v in read[0].keys() if v not in TEHNILISED_VEERUD]
    return [
        {"#": str(i + 1), **{v: vorminda_lahter(v, r[v]) for v in nahtavad_veerud}}
        for i, r in enumerate(read)
    ]


def prindi_tabel_konsooli(read: list[dict]) -> None:
    """ASCII-tabel konsooli (sama väljund mis varem _kuva_tabel)."""
    if not read:
        print("(tulemusi ei leitud)\n", flush=True)
        return
    vormindatud = vorminda_read_kuvatavateks(read)
    veerud = list(vormindatud[0].keys())
    laiused = {v: max(len(v), max(len(r[v]) for r in vormindatud)) for v in veerud}
    pais = " | ".join(v.ljust(laiused[v]) for v in veerud)
    print(pais)
    print("-" * len(pais))
    for r in vormindatud:
        print(" | ".join(r[v].ljust(laiused[v]) for v in veerud))
    print(f"\n({len(read)} rida)\n", flush=True)


def tootle_kusimus(
    seaded: Seaded,
    teed: AndmeTeed,
    mallid: tuple[SqlMall, ...],
    kusimus: str,
    andmete_vahemik: tuple[str, str],
    progress: Callable[[str], None] | None = None,
) -> KusimuseTulemus:
    if progress:
        progress("[klassifitseerin küsimust ...]")
    tulemus: KlassifikatsiooniTulemus = klassifitseeri_kusimus(
        seaded, mallid, kusimus, andmete_vahemik
    )
    read = kaivita_select(teed.duckdb_tee, tulemus.valmis_sql)
    read_llm_jaoks = [
        {k: v for k, v in r.items() if k not in TEHNILISED_VEERUD} for r in read
    ]
    if progress:
        progress("[koostan kokkuvõtet ...]")
    kokkuvote = vasta_sql_tulemusele(
        seaded=seaded,
        kusimus=kusimus,
        mall=tulemus.mall,
        parameetrid=tulemus.parameetrid,
        tabel=read_llm_jaoks,
        andmete_vahemik=andmete_vahemik,
    )
    return KusimuseTulemus(
        mall=tulemus.mall,
        parameetrid=tulemus.parameetrid,
        read=read,
        kokkuvote=kokkuvote,
    )


def tee_jarelotsing_realjargi(
    seaded: Seaded,
    teed: AndmeTeed,
    read: list[dict],
    mall: SqlMall,
    rea_nr_1_pohine: int,
) -> JarelotsinguTulemus:
    """rea_nr_1_pohine: 1..len(read). Tõstab ValueError vale rea korral."""
    if not read:
        raise ValueError("Tabel on tühi — sarnasusotsingut ei saa teha.")
    n = len(read)
    if not (1 <= rea_nr_1_pohine <= n):
        raise ValueError(f"Rea number peab olema vahemikus 1..{n}, saadi {rea_nr_1_pohine}.")
    valitud = read[rea_nr_1_pohine - 1]
    on_detail = mall.id in DETAIL_MALLID
    if on_detail:
        kokkuvote, sarnased = tee_jarelotsing(
            seaded=seaded,
            duckdb_tee=teed.duckdb_tee,
            chroma_kaust=teed.chroma_kaust,
            tehingu_id=str(valitud["tehingu_id"]),
        )
    else:
        kokkuvote, sarnased = tee_jarelotsing(
            seaded=seaded,
            duckdb_tee=teed.duckdb_tee,
            chroma_kaust=teed.chroma_kaust,
            vastaspool_nimi=str(valitud["vastaspool_nimi"]),
        )
    return JarelotsinguTulemus(kokkuvote=kokkuvote, read=sarnased)
