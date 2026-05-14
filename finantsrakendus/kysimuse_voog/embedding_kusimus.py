"""Teksti vektoriseerimine (kasutaja küsimus või valitud tehingu "otsingu_tekst") vektoriks.

Tööloogika:
    1) Lazy import ja lazy load: "sentence_transformers" ja embedding-mudel
       laetakse alles esimese "embedi_yhe_teksti" kutse ajal. Kui järelotsingut
       ei kasutata, ei laadita ka "torch"-i ega mudelit mällu.
    2) Embedding-mudel töötab CPU peal. GPU jääb Ollama LLM-i jaoks, mida
       kasutatakse igal küsimusel. Kahe suure mudeli korraga GPU-s hoidmine
       võib mälu täis teha. CPU peal on vektoriseerimine aeglasem, kuid
       järelotsingu jaoks piisavalt kiire.
    3) Mudel laaditakse mällu ainult esimesel kasutamisel. Pärast seda
       kasutatakse sama mudelit uuesti, et järgmised kutsed oleksid kiiremad.

Märkus:
    Eraldi käivitamiseks pole; importimine ja kutsumine "kysimuse_voog/" mooduliahelast.
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

from typing import TYPE_CHECKING

from finantsrakendus.ettevalmistus.laadi_tehingud_duckdb import JUURKAUST

# Import on vajalik ainult tüübivihjete jaoks.
# Programmi käivitamisel seda ei tehta, et vältida raske teegi varajast laadimist.
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Hoiab laetud SentenceTransformer objekti kuni protsessi lõpuni
_mudel_cache: "SentenceTransformer | None" = None


# Laeb (kui veel laadimata) embedding-mudeli "CPU peal" ja tagastab cache'itud objekti.
# TAGASTAB: SentenceTransformer.
def _hangi_mudel(mudeli_id: str) -> "SentenceTransformer":
    global _mudel_cache
    if _mudel_cache is not None:
        return _mudel_cache

    # Alles siin teeme raske impordi - oluline lazy-load mustrist!
    from sentence_transformers import SentenceTransformer

    cache_kaust = (JUURKAUST / "embedding_mudelid").resolve()
    cache_kaust.mkdir(parents=True, exist_ok=True)

    print(f"[laen embedding-mudelit {mudeli_id!r} (CPU) ...]", flush=True)
    _mudel_cache = SentenceTransformer(
        mudeli_id,
        cache_folder=str(cache_kaust),
        device="cpu",
        # Qwen mudel kasutab oma transformer-koodi (mitte standardset HF Transformer-it) - HF nõuab kinnitust.
        model_kwargs={"trust_remote_code": True},
        # Qwen mudelikaardi soovitus: tokenizer paddib vasakult, et viimane token oleks kindel positsioon.
        processor_kwargs={"padding_side": "left"},
    )
    print("[embedding-mudel valmis, edaspidi kiired kõned]\n", flush=True)
    return _mudel_cache


# Embedib teksti vektoriks
# TAGASTAB: list[float] (Qwen3-Embedding-8B puhul 4096 numbrit).
def embedi_yhe_teksti(mudeli_id: str, tekst: str) -> list[float]:
    t = (tekst or "").strip()
    if not t:
        raise ValueError("Embedimiseks antud tekst on tühi.")
    mudel = _hangi_mudel(mudeli_id)
    vektor = mudel.encode([t], batch_size=1, convert_to_numpy=True, show_progress_bar=False)[0]
    return vektor.tolist()
