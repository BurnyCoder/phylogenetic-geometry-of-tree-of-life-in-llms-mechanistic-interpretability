"""Tests for the pure helper functions of the species selection rule."""

import pandas as pd

from phylogeo.data.species import (
    assign_stratum,
    common_name,
    head_noun,
    lineage_ranks,
    select_by_quota,
)


def test_common_name_rules():
    from phylogeo.data.species import ensure_wordnet

    ensure_wordnet()
    proper = set()
    assert common_name("Lion", "Panthera leo", proper) == "lion"
    assert common_name("African bush elephant", "Loxodonta africana", proper) == "African bush elephant"
    assert common_name("Eurasian magpie", "Pica pica", proper) == "Eurasian magpie"
    assert common_name("Guinea pig", "Cavia porcellus", proper) == "guinea pig"
    assert common_name("House sparrow", "Passer domesticus", proper) == "house sparrow"
    assert common_name("Komodo dragon", "Varanus komodoensis", proper) == "Komodo dragon"
    assert common_name("Darwin's bark spider", "Caerostris darwini", proper) == "Darwin's bark spider"
    assert common_name("Common Eurasian thing", "Foo bar", proper) == "common Eurasian thing"
    assert common_name("Pacific oyster", "Magallana gigas", proper) == "Pacific oyster"
    assert common_name("West Indian Ocean coelacanth", "Latimeria chalumnae", proper) == "West Indian Ocean coelacanth"
    assert common_name("Basking shark", "Cetorhinus maximus", proper) == "basking shark"
    assert common_name("Lion's mane jellyfish", "Cyanea capillata", proper) == "lion's mane jellyfish"
    assert common_name("Red fox", "Vulpes vulpes", proper) == "red fox"
    assert common_name("Jaguar (animal)", "Panthera onca", proper) == "jaguar"
    # Title equal to scientific name or genus is not a common name.
    assert common_name("Epinephelus ongus", "Epinephelus ongus", proper) is None
    assert common_name("Tyrannosaurus", "Tyrannosaurus rex", proper) is None
    assert common_name("Species 8472", "Foo bar", proper) is None


def test_head_noun():
    assert head_noun("great white shark") == "shark"
    assert head_noun("lion") == "lion"
    assert head_noun("Black-capped chickadee") == "chickadee"


def test_assign_stratum_priority():
    assert assign_stratum({"Metazoa", "Chordata", "Mammalia"}) == "mammals"
    assert assign_stratum({"Metazoa", "Aves", "Archelosauria"}) == "birds"
    assert assign_stratum({"Metazoa", "Squamata", "Lepidosauria"}) == "reptiles"
    assert assign_stratum({"Metazoa", "Annelida"}) == "other"
    assert assign_stratum({"Viridiplantae"}) is None


def test_lineage_ranks_nearest_first():
    lin = [("Panthera", "genus", 1), ("Felidae", "family", 2), ("Carnivora", "order", 3),
           ("Mammalia", "class", 4), ("Chordata", "phylum", 5)]
    r = lineage_ranks(lin)
    assert r == {"phylum": "Chordata", "class": "Mammalia", "order": "Carnivora",
                 "family": "Felidae", "genus": "Panthera"}


def test_select_by_quota_family_cap_and_order():
    rows = []
    for i in range(6):  # 6 species in one family, decreasing sitelinks
        rows.append(dict(sci_name=f"A a{i}", stratum="mammals", sitelinks=100 - i, family="F1", genus="A"))
    rows.append(dict(sci_name="B b", stratum="mammals", sitelinks=50, family="F2", genus="B"))
    df = pd.DataFrame(rows)
    out = select_by_quota(df, {"mammals": 4}, family_cap=3)
    assert len(out) == 4
    assert (out["family"] == "F1").sum() == 3  # cap respected
    assert list(out["sci_name"][:3]) == ["A a0", "A a1", "A a2"]  # highest sitelinks first
    assert "B b" in set(out["sci_name"])
