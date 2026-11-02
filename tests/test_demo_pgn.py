import chess
import chess.pgn

from fairplay.demo.pgn import focus_player, parse, read_games

OPENING = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5",
           "Bb3", "d6", "c3", "O-O", "h3", "Nb8", "d4", "Nbd7", "Nbd2", "Bb7", "Bc2", "Re8"]


def make_pgn(white="alice", black="bob", game_id="AbCdEfGh", tc="180+0", event="Rated Blitz game",
             clocks=True, site="https://lichess.org/", termination="Normal",
             white_title=None) -> str:
    g = chess.pgn.Game()
    g.headers.update({"Event": event, "Site": site + game_id, "White": white, "Black": black,
                      "Result": "1-0", "UTCDate": "2025.07.10", "UTCTime": "12:00:00",
                      "WhiteElo": "1500", "BlackElo": "1550", "WhiteRatingDiff": "+6",
                      "BlackRatingDiff": "-6", "TimeControl": tc, "Termination": termination})
    if white_title:
        g.headers["WhiteTitle"] = white_title
    node, board = g, chess.Board()
    for i, san in enumerate(OPENING):
        node = node.add_variation(board.parse_san(san))
        board.push(node.move)
        if clocks:
            node.set_clock(180 - 3 * (i // 2))
    return str(g) + "\n\n"


def test_reads_lichess_game_with_clocks():
    focus, parsed, unreadable = parse(make_pgn(), side="w")
    assert unreadable == 0 and len(parsed) == 1
    g = parsed[0].game
    assert g.game_id == "AbCdEfGh" and g.white.account_id == focus
    assert g.moves == OPENING and g.clocks[:2] == [180, 180] and g.result == "1-0"
    assert g.white.rating == 1500 and g.white.rating_diff == 6 and g.tc_base == 180


def test_names_never_survive_parsing():
    _, parsed, _ = parse(make_pgn(white="secretname"), side="w")
    assert "secretname" not in parsed[0].game.model_dump_json()


def test_focus_player_is_the_one_in_every_game():
    text = make_pgn("alice", "bob", "AAAAAAAA") + make_pgn("carol", "alice", "BBBBBBBB")
    assert focus_player(read_games(text)) == "alice"
    focus, parsed, _ = parse(text)
    assert [p.color for p in parsed] == ["w", "b"]
    assert parsed[1].game.black.account_id == focus


def test_mixed_players_need_a_side():
    text = make_pgn("alice", "bob", "AAAAAAAA") + make_pgn("carol", "dave", "BBBBBBBB")
    _, parsed, _ = parse(text)
    assert all(p.game is None and "pick a side" in p.problem for p in parsed)


def test_out_of_scope_games_explain_why():
    cases = {
        make_pgn(site="https://chess.com/game/"): "not a Lichess game",
        make_pgn(tc="-"): "no clock",
        make_pgn(tc="900+10"): "not blitz",
        make_pgn(event="Casual Blitz game"): "unrated",
        make_pgn(white_title="BOT"): "bot game",
        make_pgn(termination="Abandoned"): "abandoned",
    }
    for text, reason in cases.items():
        _, parsed, _ = parse(text, side="w")
        assert parsed[0].game is None and reason in parsed[0].problem


def test_missing_clocks_kept_but_marked():
    _, parsed, _ = parse(make_pgn(clocks=False), side="b")
    assert parsed[0].game.clocks is None and parsed[0].color == "b"


def test_game_links_parse_and_profiles_are_refused():
    import pytest

    from fairplay.demo.lichess import FetchError, game_ids

    text = ("https://lichess.org/AbCdEfGh/black#24 and lichess.org/ZZZZZZZZ1234 "
            "https://lichess.org/AbCdEfGh")
    assert game_ids(text) == ["AbCdEfGh", "ZZZZZZZZ"]
    with pytest.raises(FetchError, match="profile"):
        game_ids("https://lichess.org/@/someone")
    with pytest.raises(FetchError, match="no Lichess"):
        game_ids("https://example.com/AbCdEfGh")


def test_fetch_sends_one_request_and_reports_rate_limits():
    import httpx
    import pytest

    from fairplay.demo.lichess import FetchError, fetch_pgn

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=make_pgn())

    pgn = fetch_pgn(["AbCdEfGh", "BBBBBBBB"], httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(seen) == 1 and seen[0].content == b"AbCdEfGh,BBBBBBBB"
    assert seen[0].url.params["clocks"] == "true" and "[%clk" in pgn
    limited = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(FetchError, match="one minute"):
        fetch_pgn(["AbCdEfGh"], limited)
