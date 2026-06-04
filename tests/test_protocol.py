"""Protocol-parser tests against the exact live reference-server strings."""

from zappy_rl.deploy.protocol import (
    inventory_to_vec,
    parse_broadcast,
    parse_inventory,
    parse_look,
)


def test_parse_inventory_live_string():
    line = "[ food 9, linemate 0, deraumere 0, sibur 0, mendiane 0, phiras 0, thystame 0 ]"
    inv = parse_inventory(line)
    assert inv == {
        "food": 9, "linemate": 0, "deraumere": 0, "sibur": 0,
        "mendiane": 0, "phiras": 0, "thystame": 0,
    }
    assert inventory_to_vec(inv) == [9, 0, 0, 0, 0, 0, 0]


def test_parse_look_live_string():
    line = "[ player food, mendiane,, food linemate ]"
    tiles = parse_look(line)
    assert tiles == [["player", "food"], ["mendiane"], [], ["food", "linemate"]]
    assert len(tiles) == 4  # level-1 look = (1+1)**2 tiles


def test_parse_look_tolerates_no_space_separator():
    # Subject: separator is "a comma followed or not by a space".
    assert parse_look("[player,food,,thystame]") == [["player"], ["food"], [], ["thystame"]]


def test_parse_look_empty_single_tile():
    assert parse_look("[ ]") == [[]]
    assert parse_look("[player]") == [["player"]]


def test_parse_broadcast():
    assert parse_broadcast("message 3, help north") == (3, "help north")
    assert parse_broadcast("0, same tile") == (0, "same tile")
