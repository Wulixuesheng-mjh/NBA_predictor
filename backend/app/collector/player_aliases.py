"""Explicit official-ID aliases; never fuzzy-match or change team membership."""
PLAYER_ALIASES = {
    1630231: ('KJ Martin','Kenyon Martin Jr.'),
    1642259: ('Alexandre Sarr','Alex Sarr'),
    1630562: ('Matt Hurt','Matthew Hurt'),
    1642267: ('Carlton Carrington','Bub Carrington'),
    1642905: ('Yang Hansen','Hansen Yang'),
}
ALIAS_SOURCES = {
    1630231: 'https://www.nba.com/sixers/news/kj-martin-bio-stats-facts',
    1642259: 'https://www.nba.com/draft/2024/prospects/alexandre-sarr',
    1630562: 'https://www.nba.com/player/1630562/matthew-hurt',
    1642267: 'https://www.nba.com/player/1642267/carlton-carrington/profile',
    1642905: 'https://www.nba.com/player/1642905/hansen-yang',
}

# Explicit NBA identities for scorer tables when imported legacy/non-NBA rows
# share a name. These IDs never infer historic team membership.
SCORER_NAME_IDS = {
    'Georges Niang': 1627777,
    'Bogdan Bogdanovic': 203992,
    'Noa Essengue': 1642855,
    'Jaylin Williams': 1631119,
    'Mario Hezonja': 1626209,
    'Brandon Williams': 1630314,
}
SCORER_ID_SOURCES = {
    1627777: 'https://www.nba.com/player/1627777/georges-niang',
    203992: 'https://www.nba.com/player/203992/bogdan-bogdanovic',
    1642855: 'https://www.nba.com/player/1642855/noa-essengue',
    1631119: 'https://www.nba.com/player/1631119/jaylin-williams',
    1626209: 'https://www.nba.com/player/1626209/mario-hezonja',
    1630314: 'https://www.nba.com/player/1630314/brandon-williams',
}
