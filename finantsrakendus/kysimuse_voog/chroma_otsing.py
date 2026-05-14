"""ChromaDB semantiline otsing tehingute embeddingute kogumikus.

Eeldab, et tehingute embeddingud on eelnevalt arvutatud ja salvestatud koodiga
"finantsrakendus/ettevalmistus/embeddi_tehingud.py". Chroma kogumikus hoitakse iga
tehingu kohta ainult "tehingu_id" ja 4096-mõõtmeline vektor; täielik tehinguinfo
(kuupäev, summa, vastaspool jne) loetakse hiljem DuckDB-st "tehingu_id" järgi.
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

from pathlib import Path

import chromadb


# Otsib Chroma kogumikust top-K kõige sarnasemat vektorit etteantud küsimuse vektori suhtes.
# TAGASTAB: listina sõnastikud { "tehingu_id", "distance" } sarnasusjärjestuses (lähimast kaugemani).
def otsi(
    chroma_kaust: Path,
    kogumik: str,
    kusimus_embedding: list[float],
    top_k: int,
) -> list[dict]:
    klient = chromadb.PersistentClient(path=str(chroma_kaust))
    col = klient.get_collection(name=kogumik)

    # include=["distances"]: muud väljad (documents/metadatas) on Chromast välja jäetud, sest
    # neid sinna ei salvestata; ID-d tulevad vastusesse niikuinii (pole include-is).
    res = col.query(
        query_embeddings=[kusimus_embedding],
        n_results=top_k,
        include=["distances"],
    )

    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    return [
        {"tehingu_id": ids[i], "distance": dists[i] if i < len(dists) else None}
        for i in range(len(ids))
    ]
