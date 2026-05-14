"""
Streamlit-põhine kasutajaliides finantsprototüübile.

Käivitamine projekti juurkaustast (virtuaalkeskkond peab olema aktiivne, Ollama peab töötama):

streamlit run kasutajaliides_streamlit/app.py

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

import sys
from pathlib import Path

# Tagab käivitamise ka siis, kui töökataloog pole projekti juur.
_JUUR = Path(__file__).resolve().parents[1]
if str(_JUUR) not in sys.path:
    sys.path.insert(0, str(_JUUR))

import pandas as pd
import streamlit as st

from finantsrakendus.ettevalmistus.main_ettevalmistus import tee_ettevalmistus
from finantsrakendus.kysimuse_voog.klassifitseeri_kusimus import KusimusEiSobi
from finantsrakendus.kysimuse_voog.teenus import (
    JarelotsinguTulemus,
    KusimuseTulemus,
    loe_andmete_ajavahemik,
    tee_jarelotsing_realjargi,
    tootle_kusimus,
    vorminda_read_kuvatavateks,
)
from finantsrakendus.seaded.loe_seaded import loe_seaded
from finantsrakendus.seaded.loe_sql_mallid import loe_sql_mallid

# Tekst, mis kuvatakse enne Ollama/LLM-i koostatud sõnalist kokkuvõtet.
_LLM_KOKKUVOTE_PEALDIS = "Järgnev tekst on keelemudeli (LLM) koostatud vastus."


def _puhasta_uue_kusimuse_jaoks() -> None:
    """Tühjendab vastused; küsimuse välja tühjendatakse järgmisel käivitusel (enne text_area)."""
    for k in (
        "viimane_kusimuse_tulemus",
        "viimane_read",
        "viimane_mall",
        "viimane_jarelotsing",
        "viimane_teade",
    ):
        st.session_state.pop(k, None)
    st.session_state["_kusimus_reset_jargmine_kord"] = True
    st.session_state["tulemus_versioon"] = st.session_state.get("tulemus_versioon", 0) + 1


def _rakenda_kusimuse_välja_reset_kui_vajalik() -> None:
    """Streamlit ei luba muuta key=kusimus_sisend pärast text_area loomist — tühjendame võtme enne vidinat."""
    if st.session_state.pop("_kusimus_reset_jargmine_kord", False):
        st.session_state.pop("kusimus_sisend", None)


NÄIDISKÜSIMUSED: tuple[tuple[str, str], ...] = (
    (
        "Suurimad kulutused",
        "Mis olid minu 10 suurimat kulutust aastatel 2024-01-01 kuni 2025-12-31?",
    ),
    (
        "Korduvad maksed",
        "Leia korduvad maksed perioodil 2024-01-01 kuni 2025-06-30.",
    ),
    (
        "Topeltmaksed",
        "Kas mul on võimalikke topeltmakseid perioodil 2024-06-01 kuni 2025-12-31?",
    ),
    (
        "Palju väikseid oste",
        "Leia kohad, kus olen teinud palju väikseid oste (alla 15 euro) perioodil 2024-01-01 kuni 2025-12-31.",
    ),
    (
        "Tõusnud püsikulud",
        "Millised püsikulud on tõusnud rohkem kui 10% aastatel 2024-01-01 kuni 2025-12-31?",
    ),
)


def _tagab_sessiooni() -> None:
    if st.session_state.get("_ctx_ok"):
        return
    if st.session_state.get("_boot_error"):
        return
    with st.status("Ettevalmistus — esmakordsel käivitamisel võib see võtta mitu minutit (vektorid, mudelid)...", expanded=True) as status:
        status.write("Laen andmeid, kontrollin Ollama ühendust...")
        try:
            teed = tee_ettevalmistus(vaikne=True)
            seaded = loe_seaded()
            mallid = loe_sql_mallid()
            andmed = loe_andmete_ajavahemik(teed.duckdb_tee)
        except Exception as e:
            status.update(label="Ettevalmistus ebaõnnestus", state="error")
            st.session_state["_boot_error"] = str(e)
            return
        st.session_state["_teed"] = teed
        st.session_state["_seaded"] = seaded
        st.session_state["_mallid"] = mallid
        st.session_state["_andmete_vahemik"] = andmed
        st.session_state["_ctx_ok"] = True
        status.update(label="Ettevalmistus valmis", state="complete")


def main() -> None:
    st.set_page_config(page_title="Isiklike kulutuste analüüsimise prototüüp", layout="wide")
    st.title("Isiklike kulutuste analüüsimise prototüüp")
    st.caption(
        "Uuri oma kulutusi lihtsate küsimuste abil. Süsteem koostab andmepõhise kokkuvõtte viie ettevalmistatud küsimusetüübi piires."
    )

    _tagab_sessiooni()
    if st.session_state.get("_boot_error"):
        st.error(f"Käivitus ebaõnnestus: {st.session_state['_boot_error']}")
        st.info("Kontrolli README juhiseid: Ollama, mudel `seaded.json`-is, CSV tee.")
        if st.button("Proovi uuesti"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        return

    teed = st.session_state["_teed"]
    seaded = st.session_state["_seaded"]
    mallid = st.session_state["_mallid"]
    andmed = st.session_state["_andmete_vahemik"]

    _rakenda_kusimuse_välja_reset_kui_vajalik()

    with st.expander("Teave andmestiku ja mudelite kohta", expanded=True):
        st.markdown(
            f"- **CSV fail:** `{teed.csv_tee.name}`  \n"
            f"- **Andmete kuupäevad:** {andmed[0]} — {andmed[1]}  \n"
            f"- **LLM (Ollama):** `{seaded.llm_mudel}`  \n"
            f"- **Embedding:** `{seaded.embedding_mudel}`"
        )

    with st.expander("Toetatavad küsimusetüübid", expanded=True):
        st.caption("Klõpsa küsimusel, et kopeerida näidisküsimus väljale.")
        for pealkiri, tekst in NÄIDISKÜSIMUSED:
            if st.button(pealkiri, key=f"näidis_{pealkiri}"):
                st.session_state["kusimus_sisend"] = tekst
                st.rerun()

    if "kusimus_sisend" not in st.session_state:
        st.session_state["kusimus_sisend"] = ""

    kusimus = st.text_area(
        "Sinu küsimus",
        height=120,
        key="kusimus_sisend",
        placeholder="Kirjuta küsimus eesti keeles, näiteks suurimate kulutuste või korduvate maksete kohta valitud ajaga.",
    )

    if st.button("Saada küsimus", type="primary"):
        k = (kusimus or "").strip()
        if not k:
            st.warning("Palun kirjuta küsimus.")
        else:
            st.session_state.pop("viimane_teade", None)
            try:
                with st.spinner("Klassifitseerin küsimust, käivitan päringu ja koostan kokkuvõtet — see võib võtta pisut aega..."):
                    tulemus = tootle_kusimus(
                        seaded, teed, mallid, k, andmed, progress=None
                    )
            except KusimusEiSobi as e:
                st.session_state.pop("viimane_kusimuse_tulemus", None)
                st.session_state.pop("viimane_read", None)
                st.session_state.pop("viimane_mall", None)
                st.session_state.pop("viimane_jarelotsing", None)
                st.session_state["viimane_teade"] = ("info", str(e))
            except (ValueError, RuntimeError) as e:
                st.session_state.pop("viimane_kusimuse_tulemus", None)
                st.session_state.pop("viimane_read", None)
                st.session_state.pop("viimane_mall", None)
                st.session_state.pop("viimane_jarelotsing", None)
                st.session_state["viimane_teade"] = ("error", f"Viga: {e}")
            else:
                st.session_state["viimane_kusimuse_tulemus"] = tulemus
                st.session_state["viimane_read"] = tulemus.read
                st.session_state["viimane_mall"] = tulemus.mall
                st.session_state.pop("viimane_jarelotsing", None)
                st.session_state["tulemus_versioon"] = st.session_state.get("tulemus_versioon", 0) + 1

    teade = st.session_state.get("viimane_teade")
    if teade:
        tüüp, sisu = teade
        if tüüp == "info":
            st.info(sisu)
        else:
            st.error(sisu)

    tulemus_kuvaks: KusimuseTulemus | None = st.session_state.get("viimane_kusimuse_tulemus")
    if tulemus_kuvaks is not None:
        st.subheader("Viimane vastus")
        t = tulemus_kuvaks
        st.caption(f"Mall: **{t.mall.id}** · Parameetrid: `{t.parameetrid}`")
        if t.kokkuvote:
            st.caption(_LLM_KOKKUVOTE_PEALDIS)
            st.markdown(t.kokkuvote)
        if not t.read:
            st.warning("(Tulemusi ei leitud.)")
        else:
            df = pd.DataFrame(vorminda_read_kuvatavateks(t.read))
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height="content",
            )
            st.caption(f"Ridu: {len(t.read)}")

        if st.button("Uus küsimus", key="uus_kusimus_peale_vastust", type="primary"):
            _puhasta_uue_kusimuse_jaoks()
            st.rerun()

    read = st.session_state.get("viimane_read")
    mall = st.session_state.get("viimane_mall")
    if read and mall and len(read) > 0:
        st.divider()
        st.subheader("Sarnasusotsing")
        st.caption(
            "Vali eelmise tabeli rea number (1 = esimene rida). Valitud tehingu vektoriesituse põhjal "
            "leitakse Chroma vektorbaasist 20 sellele semantiliselt kõige sarnasemat tehingut."
        )
        n = len(read)
        ver = st.session_state.get("tulemus_versioon", 0)
        rea_valik = st.number_input(
            "Rea number",
            min_value=1,
            max_value=n,
            value=1,
            step=1,
            key=f"jarelotsing_rea_{ver}",
        )
        if st.button("Otsi sarnaseid tehinguid", key=f"jarelotsing_nupp_{ver}"):
            try:
                with st.spinner("Otsin sarnaseid tehinguid..."):
                    jt = tee_jarelotsing_realjargi(seaded, teed, read, mall, int(rea_valik))
            except (ValueError, RuntimeError) as e:
                st.session_state.pop("viimane_jarelotsing", None)
                st.error(str(e))
            else:
                st.session_state["viimane_jarelotsing"] = jt

        jt_kuvaks: JarelotsinguTulemus | None = st.session_state.get("viimane_jarelotsing")
        if jt_kuvaks is not None:
            st.markdown("**Sarnasusotsingu tulemus**")
            if jt_kuvaks.kokkuvote:
                st.caption(_LLM_KOKKUVOTE_PEALDIS)
                st.markdown(jt_kuvaks.kokkuvote)
            if not jt_kuvaks.read:
                st.warning("(Sarnaseid ei leitud.)")
            else:
                st.dataframe(
                    pd.DataFrame(vorminda_read_kuvatavateks(jt_kuvaks.read)),
                    use_container_width=True,
                    hide_index=True,
                    height="content",
                )
                st.caption(f"Sarnaseid ridu: {len(jt_kuvaks.read)}")

            if st.button("Uus küsimus", key="uus_kusimus_peale_jarelotsingut", type="primary"):
                _puhasta_uue_kusimuse_jaoks()
                st.rerun()


if __name__ == "__main__":
    main()
