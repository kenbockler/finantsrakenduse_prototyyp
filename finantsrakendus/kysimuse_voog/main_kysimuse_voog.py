"""Küsimuse voo peavoog: küsib kasutajalt küsimuse ja töötleb seda.

Märkus:
    Iseseisvalt seda moodulit ei käivitata. Ainus käivituspunkt on
    "finantsrakendus/main.py" ("python -m finantsrakendus.main").
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

from finantsrakendus.ettevalmistus.main_ettevalmistus import AndmeTeed
from finantsrakendus.kysimuse_voog.klassifitseeri_kusimus import KusimusEiSobi
from finantsrakendus.kysimuse_voog.teenus import (
    loe_andmete_ajavahemik,
    prindi_tabel_konsooli,
    tee_jarelotsing_realjargi,
    tootle_kusimus,
)
from finantsrakendus.seaded.loe_seaded import Seaded
from finantsrakendus.seaded.loe_sql_mallid import SqlMall, loe_sql_mallid


def _on_lopetus(sisend: str) -> bool:
    s = sisend.strip().lower()
    return s == "" or s in {"exit", "välju", "lõpeta"}


def _tootle_kusimust(
    seaded: Seaded,
    teed: AndmeTeed,
    mallid: tuple[SqlMall, ...],
    kusimus: str,
    andmete_vahemik: tuple[str, str],
) -> tuple[list[dict], SqlMall]:
    def _progress(msg: str) -> None:
        print(msg, flush=True)

    tulemus = tootle_kusimus(
        seaded, teed, mallid, kusimus, andmete_vahemik, progress=_progress
    )
    print(f"\nMall: {tulemus.mall.id}")
    print(f"Parameetrid: {tulemus.parameetrid}\n", flush=True)
    if tulemus.kokkuvote:
        print(f"\n{tulemus.kokkuvote}\n", flush=True)
    prindi_tabel_konsooli(tulemus.read)
    return tulemus.read, tulemus.mall


def _paku_jarelotsingut(
    seaded: Seaded, teed: AndmeTeed, read: list[dict], mall: SqlMall
) -> None:
    if not read:
        return

    n = len(read)
    while True:
        try:
            sisend = input(
                f"Sarnasusotsing> sisesta tabeli rea number 1..{n} (tühi rida = uus küsimus): "
            )
        except (EOFError, KeyboardInterrupt):
            print("", flush=True)
            return

        tekst = sisend.strip()
        if not tekst:
            return
        try:
            indeks = int(tekst)
        except ValueError:
            print(f"Vigane rea number {tekst!r}; oodatud täisarv vahemikus 1..{n}.\n", flush=True)
            continue
        if not (1 <= indeks <= n):
            print(f"Rea number {indeks} pole vahemikus 1..{n}.\n", flush=True)
            continue

        try:
            jt = tee_jarelotsing_realjargi(seaded, teed, read, mall, indeks)
        except (ValueError, RuntimeError) as e:
            print(f"Sarnasusotsing ebaõnnestus: {e}\n", flush=True)
            continue

        if jt.kokkuvote:
            print(f"\n{jt.kokkuvote}\n", flush=True)
        prindi_tabel_konsooli(jt.read)


def kaivita_kysimuse_voog(teed: AndmeTeed, seaded: Seaded) -> None:
    mallid = loe_sql_mallid()
    andmete_vahemik = loe_andmete_ajavahemik(teed.duckdb_tee)

    print("\nTere! See on isiklik finantskulutuste analüüsimise süsteemi prototüüp.")
    print("See võimaldab vabas tekstis pärida huvipakkuvaid tehinguid ning selgitada neid andmepõhiselt.\n")

    print(
        f"Kasutusel: andmestik={teed.csv_tee.name} ({andmete_vahemik[0]} kuni {andmete_vahemik[1]}); "
        f"LLM={seaded.llm_mudel}; embedding={seaded.embedding_mudel}\n",
        flush=True,
    )

    print("See prototüüp on häälestatud vastama viiele küsimustüübile:")
    print("  1. Leia suurimad või väiksemad kulutused (Vali ajaperiood ja soovitud kulutuste arv)")
    print("  2. Leia korduvaid makseid (Vali ajaperiood)")
    print("  3. Leia võimalikud topeltmaksed (Vali ajaperiood)")
    print("  4. Leia tehingud, kus teed sageli palju väikeseid kulutusi. (Vali ajaperiood, summa ülempiir)")
    print("  5. Leia püsimaksed, mille hind on tõusnud. (Vali ajaperiood ja muutuse lävi protsentides)\n")

    print("Sisesta küsimus (lõpetamiseks tühi rida või 'exit').\n", flush=True)

    while True:
        try:
            sisend = input("Küsimus> ")
        except (EOFError, KeyboardInterrupt):
            print("\nLõpetan.")
            return

        if _on_lopetus(sisend):
            print("Lõpetan.")
            return

        kusimus = sisend.strip()
        try:
            read, mall = _tootle_kusimust(seaded, teed, mallid, kusimus, andmete_vahemik)
        except KusimusEiSobi as e:
            print(f"\n{e}\n", flush=True)
            continue
        except (ValueError, RuntimeError) as e:
            print(f"Viga: {e}\n", flush=True)
            continue

        _paku_jarelotsingut(seaded, teed, read, mall)
