"""Official scorer IDs must override duplicate names and handle accents."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('name,nba_id', [('Moritz Wagner',1629021),('Hugo González',1642864),('Luka Dončić',1629029),('Tacko Fall',1629605)])
def test_scorer_overrides_are_bound_to_verified_official_ids(name,nba_id):
    path=Path('/app/scripts/import_gamebooks.py')
    if not path.exists(): path=Path(__file__).resolve().parents[2]/'scripts/import_gamebooks.py'
    spec=importlib.util.spec_from_file_location('scorer_exact',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.SCORER_ONLY_IDENTITIES[name][0]==nba_id
    key=module.normalize(name)
    players={1:SimpleNamespace(nba_player_id=999999999),2:SimpleNamespace(nba_player_id=nba_id)}
    assert module.scorer_player_ids(name,{key:{1}},{key:{1,2}},players,{key:nba_id})=={2}
    assert module.scorer_player_ids(name,{key:{1}},{key:{1}},{1:players[1]},{key:nba_id})==set()


def test_verified_scorer_identity_handles_accent_and_legacy_collision():
    path=Path('/app/scripts/import_gamebooks.py')
    if not path.exists():
        path=Path(__file__).resolve().parents[2]/'scripts/import_gamebooks.py'
    spec=importlib.util.spec_from_file_location('scorer_import',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    players={1:SimpleNamespace(nba_player_id=196295054),2:SimpleNamespace(nba_player_id=1643510)}
    key=module.normalize('Karim López')
    names={key:{1,2}}
    official={key:1643510}
    assert module.scorer_player_ids('Karim López',{},names,players,official)=={2}
    assert module.scorer_player_ids('Karim López',{},{},players,official)=={2}
    assert module.scorer_player_ids('Karim Lopez',{},names,players,official)=={2}
    assert module.scorer_player_ids('Karim Lopez',{},names,players,{})=={1,2}
    assert module.scorer_player_ids('Karim López',{},names,{1:players[1]},official)==set()
