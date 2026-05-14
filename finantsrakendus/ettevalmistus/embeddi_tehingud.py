"""Arvutab tehingute "otsingu_tekst" Qwen3-Embedding-8B vektoriteks ja salvestab ChromaDB-sse.

Mida see fail teeb:
    Loeb DuckDB tabelist "tehingud" iga rea välja "otsingu_tekst", arvutab sellest
    sentence-transformers mudeliga "Qwen/Qwen3-Embedding-8B" vektori (4096 numbrit)
    ja salvestab vektorid ChromaDB kogumikku koos ainsa identifikaatoriga "tehingu_id".
    Täisinfo (kuupaev, summa, vastaspool jms) jääb DuckDB-sse - kui hiljem semantiline
    otsing tagastab "tehingu_id"-d, laeme täisread DuckDB-st.

Nõue: CUDA-toega GPU. Mudelifailid hoitakse projektikaustas "embedding_mudelid/"

Käivitamine:
    source .venv_py_312/bin/activate
    python -m finantsrakendus.ettevalmistus.embeddi_tehingud
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

# Standardteegid: failiteed ja tüübivihjed (TYPE_CHECKING - importi ei laadita run-timeil).
from pathlib import Path
from typing import TYPE_CHECKING

# Kerge tipuga import - duckdb on väike (~60 ms) ja vajame seda igal jooksul.
import duckdb

# Projekti seest: laadurist samad teed/tabelinimi mis CSV->DuckDB sammus.
from finantsrakendus.ettevalmistus.laadi_tehingud_duckdb import (
    DUCKDB_FAILI_TEE,
    JUURKAUST,
    TABEL_TEHINGUD,
)

# Tüübivihjed jooksvalt EI lae raskeid tegijaid (torch, sentence_transformers, chromadb);
# TYPE_CHECKING on alati False run-timeil, kuid IDE/pyright loevad selle.
if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import SentenceTransformer


# Kuhu Hugging Face embedding-mudeli failid laetakse (sentence-transformers cache); .gitignore'is.
EMBEDDING_MUDELID_KAUST = JUURKAUST / "embedding_mudelid"

# Chroma püsikaust - kogu vektor-indeks ühes kohas; .gitignore'is.
CHROMA_KAUST = JUURKAUST / "finantsrakendus/andmed/chroma_tehingud"

# Mitu teksti korraga GPU-le saadetakse encode-misel: suurem = kiirem, aga rohkem VRAM-i; 16 sobib alguseks.
PARTII_SUURUS = 16

# Chroma kogumiku nimi - üks fikseeritud kogumik kõigi tehingute embeddingute jaoks.
KOGUMIK = "tehingud"


# Loeb DuckDB tabelist "tehingud" kõik read.
# TAGASTAB: list tuple-itest (tehingu_id, otsingu_tekst) "tehingu_id" järjekorras.
def loe_otsingu_tekstid(duckdb_tee: Path) -> list[tuple[str, str]]:
    con = duckdb.connect(str(duckdb_tee))
    read = con.execute(
        f"""
        SELECT tehingu_id, otsingu_tekst
        FROM {TABEL_TEHINGUD}
        WHERE otsingu_tekst IS NOT NULL AND trim(otsingu_tekst) != ''
        ORDER BY tehingu_id
        """
    ).fetchall()
    con.close()
    # ChromaDB nõuab ID-d sõnena; otsingu_tekst niikuinii sõne.
    return [(str(r[0]), str(r[1])) for r in read]


# Laeb embedding-mudeli. Eelistab CUDA-t, kui see on saadaval; muidu kasutab CPU-d.
# Lazy import: torch ja sentence_transformers laetakse alles siis, kui mudel päriselt vaja.
# TAGASTAB: SentenceTransformer objekt, valmis "encode" kutseteks.
def lae_mudel(mudeli_id: str) -> SentenceTransformer:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_MUDELID_KAUST.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(
        mudeli_id,
        cache_folder=str(EMBEDDING_MUDELID_KAUST),
        device=device,
        # Qwen mudel kasutab oma transformer-koodi (mitte standardset HF Transformer-it) - HF nõuab kinnitust.
        model_kwargs={"trust_remote_code": True},
        # Qwen mudelikaardi soovitus: tokenizer paddib vasakult, et viimane token oleks kindel positsioon.
        processor_kwargs={"padding_side": "left"},
    )


# Kodeerib teksti listi vektoriteks etteantud sentence-transformers mudeliga.
# TAGASTAB: list vektoritest (iga vektor on list float-idest pikkusega 4096 Qwen3-Embedding-8B puhul).
def embeddi(model: SentenceTransformer, tekstid: list[str]) -> list[list[float]]:
    # batch_size=PARTII_SUURUS: mudel ise jagab sisendi 16-kaupa GPU peal.
    # convert_to_numpy=True: tagastab numpy massiivi (mille tolist() teeb tavaliseks Pythoni listiks).
    vektorid = model.encode(
        tekstid,
        batch_size=PARTII_SUURUS,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vektorid.tolist()


# Avab Chroma püsikliendi (kettal asuv DB) ja tagastab kogumiku, kuhu vektorid pannakse.
# Lazy import: chromadb laetakse alles siis, kui Chroma päriselt vaja.
# TAGASTAB: chromadb Collection objekt - sellel saab kutsuda .upsert(...), .query(...) jne.
def ava_kogumik(chroma_kaust: Path) -> chromadb.Collection:
    import chromadb

    # Loo vajadusel ema-kaust; PersistentClient ise loob lõppkausta ja chroma.sqlite3 faili.
    chroma_kaust.parent.mkdir(parents=True, exist_ok=True)
    klient = chromadb.PersistentClient(path=str(chroma_kaust))
    # hnsw:space=cosine: koosinussarnasus on embedding-mudelite jaoks standard (vaikimisi oleks l2/eukleidiline).
    return klient.get_or_create_collection(name=KOGUMIK, metadata={"hnsw:space": "cosine"})


# Pea-funktsioon: loeb DuckDB-st tehingud, embedib mudeliga, salvestab ChromaDB-sse.
# TAGASTAB: indekseeritud (Chroma-sse pandud) ridade arv.
def arvuta_ja_salvesta(duckdb_tee: Path, chroma_kaust: Path, mudeli_id: str) -> int:
    # 1) Loe DuckDB-st kõik (tehingu_id, otsingu_tekst) paarid.
    read = loe_otsingu_tekstid(duckdb_tee)
    if not read:
        raise ValueError("DuckDB-s pole ühtegi rida 'otsingu_tekst'-iga; käivita enne 'genereeri_otsingu_tekst'.")

    # 2) Eralda paarid kahe paralleelse listi: ID-d ja tekstid (sama indeksi peal vastavad).
    tehingu_id_loend = [r[0] for r in read]
    tekstid = [r[1] for r in read]

    # 3) Lae mudel GPU-le ja embedi kõik tekstid (mudel ise jagab partiideks PARTII_SUURUS järgi).
    mudel = lae_mudel(mudeli_id)
    vektorid = embeddi(mudel, tekstid)

    # 4) Ava Chroma kogumik ja salvesta kõik korraga; upsert kirjutab üle, kui ID juba olemas.
    kogumik = ava_kogumik(chroma_kaust)
    kogumik.upsert(ids=tehingu_id_loend, embeddings=vektorid)

    return len(tehingu_id_loend)


# Käsurea käivituspunkt: kasutab vaikimisi DuckDB ja Chroma teid + seadetes määratud mudeli ID-d.
# Lazy import: loe_seaded laetakse alles siis, kui käsurea käivitus tegelikult toimub.
# TAGASTAB: None; trükib indekseeritud ridade arvu, mudeli ja Chroma kausta info.
def main() -> None:
    from finantsrakendus.seaded.loe_seaded import loe_seaded

    seaded = loe_seaded()
    n = arvuta_ja_salvesta(
        duckdb_tee=DUCKDB_FAILI_TEE,
        chroma_kaust=CHROMA_KAUST,
        mudeli_id=seaded.embedding_mudel,
    )
    print(f"Embeddingud: {n} rida indekseeritud, mudel {seaded.embedding_mudel!r}, Chroma {CHROMA_KAUST}")



if __name__ == "__main__":
    main()

