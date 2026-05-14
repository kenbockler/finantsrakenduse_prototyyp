"""
Täidab DuckDB tabelis "tehingud" veeru "otsingu_tekst" kasutades väljade "vastaspool_nimi" ja "selgitus" väärtuseid.
Tekst on normaliseeritud: kõik väiketähtedes, erimärgid eemaldatud, ainult üksikud tühikud.
Näide: "swedbank as igapäevane pakett kuutasu"

Käivitamine:
    source .venv_py_312/bin/activate
    python -m finantsrakendus.ettevalmistus.genereeri_otsingu_tekst
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

from pathlib import Path
import duckdb
from finantsrakendus.ettevalmistus.laadi_tehingud_duckdb import DUCKDB_FAILI_TEE, TABEL_TEHINGUD

# Koostab UPDATE SQL-i, mis täidab veeru "otsingu_tekst" vastaspoole ja selgituse põhjal.
# Tekst normaliseeritakse: lowercase, ainult tähed/tühikud, üksikud tühikud, ilma alguse/lõpu tühikuteta, numbrid on eemaldatud.
# TAGASTAB: SQL-string (UPDATE-lause).
def _uuenda_sql(tabel: str) -> str:
    return f"""
    UPDATE {tabel}
    SET otsingu_tekst = trim(regexp_replace(
        regexp_replace(
            lower(vastaspool_nimi || ' ' || selgitus),
            '[^a-zäöõüšž ]+',
            ' ',
            'g'
        ),
        ' +',
        ' ',
        'g'
    ))
    """

# Avab DuckDB ühenduse, käivitab UPDATE ja sulgeb ühenduse.
# TAGASTAB: UPDATE-i poolt muudetud ridade arv.
def taida_otsingu_tekst(duckdb_tee: Path | None = None) -> int:
    tee = (duckdb_tee or DUCKDB_FAILI_TEE).resolve()
    con = duckdb.connect(str(tee))
    muudetud = con.execute(_uuenda_sql(TABEL_TEHINGUD)).fetchone()[0]
    con.close()
    return muudetud

# Käsurea käivituspunkt
# TAGASTAB: None; trükib uuendatud ridade arvu ja DuckDB faili tee.
def main() -> None:
    n = taida_otsingu_tekst()
    print(f"Uuendatud {n} rida: veerg otsingu_tekst tabelis {TABEL_TEHINGUD!r} -> {DUCKDB_FAILI_TEE}")

# Otse käivitusel käivitatakse main() meetod, sest __name__ on siis "__main__".
if __name__ == "__main__":
    main()
