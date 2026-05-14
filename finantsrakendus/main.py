"""Rakenduse käivituspunkt: ühendab ettevalmistuse ja küsimuse voo.

Käivitamine:
    source .venv_py_312/bin/activate
    python -m finantsrakendus.main

    Lõpetamisel (tühi rida, "exit", Ctrl+C) kutsutakse "ollama stop <llm_mudel>",
    et Ollama vabastaks GPU/RAM-i (kui "ollama" CLI on PATH-is).
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

import atexit
import signal
import sys
from pathlib import Path

# Võimaldab käivitamist ka kujul "python finantsrakendus/main.py" (ilma -m).
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finantsrakendus.ettevalmistus.main_ettevalmistus import tee_ettevalmistus
from finantsrakendus.kysimuse_voog.main_kysimuse_voog import kaivita_kysimuse_voog
from finantsrakendus.kysimuse_voog.ollama_klient import sulge_ollama_mudel
from finantsrakendus.seaded.loe_seaded import loe_seaded


# Rakenduse peavoog: 1) ettevalmistus (andmed + Ollama kontroll); 2) küsimuse voog (interaktiivne tsükkel).
# Lõpus: "atexit" ja signaalid kutsuvad "ollama stop" (vt "sulge_ollama_mudel").
def main() -> None:
    seaded = loe_seaded()
    # "atexit" käivitub iga normaalse väljumise korral (sh "sys.exit", erindi tõttu lõpetus).
    atexit.register(sulge_ollama_mudel, seaded.llm_mudel)

    # Ctrl+C / SIGTERM peavad samuti viima atexit-i juurde: tõstame "SystemExit",
    # mis annab kontrolli interpretaatorile ja interpretaator kutsub atexit-i ühe korra.
    def _signal_handler(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    teed = tee_ettevalmistus(vaikne=True)
    kaivita_kysimuse_voog(teed, seaded)


if __name__ == "__main__":
    main()
