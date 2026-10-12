"""Playing every deal twice (duplicate poker) without letting information about a deal cross over.

A *duplicate pair* is one :class:`~poker_arena.engine.deals.Deal` played twice with the two players
swapped between button and big blind. Cards are tied to positions, so two identical agents that keep no
memory cancel exactly in every pair. Agents that adapt to **opponent statistics** (the teacher, EBM,
XGBoost, ...) carry a memory of the opponent, which is why the way the two plays are arranged matters:

``single_table`` (legacy)
    Both plays are the consecutive hands of ONE game with shared agents and opponent trackers. The second
    play of a deal is therefore played against an opponent model that already contains the first play of
    the very same deal - the second mover learns how the opponent played the cards it now holds. Identical
    adaptive agents do not cancel; the first-named agent is the first mover of every pair and loses
    (roughly -10 to -13 BB/100 for the teacher). Kept only to reproduce and diagnose old runs.

``two_table`` (default; as in real duplicate poker)
    Two independent tables play the same deals ``d_0 .. d_{n-1}``. Each table has its own agent copies
    and its own opponent trackers (all fresh at the start), and alternates the button every hand like a
    normal heads-up game: at table 0 agent A is on the button for even ``p``, at table 1 for odd ``p``
    (the exact complement of table 0 for the same deal). The pair of deal ``p`` is A's profit at table 0
    plus A's profit at table 1; nothing a table learns about deal ``p`` can reach the other table. Identical
    adaptive agents then cancel *in expectation* (the first-mover advantage of one table is the second-mover
    disadvantage of the other); identical stateless agents still cancel exactly in every pair.

Hands are always yielded in the layout ``[table 0 deal 0, table 1 deal 0, table 0 deal 1, ...]`` (resp.
``[A on button deal 0, B on button deal 0, ...]`` for the single table), so adjacent hands ``2p`` and
``2p + 1`` are always the two plays of deal ``p`` and pair sums are ``hands[0::2] + hands[1::2]`` in both
protocols - but WHO is on the button in hand ``2p`` is no longer always agent A (it alternates with the
parity of ``p`` in ``two_table``): use the ``a_on_button`` flag, never the index parity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from poker_arena.agents.base import Agent
from poker_arena.config import GameConfig
from poker_arena.engine.deals import Deal
from poker_arena.engine.history import HandRecord
from poker_arena.engine.opponent import OpponentTracker
from poker_arena.engine.session import play_hand
from poker_arena.tournament.config import PROTOCOLS, SINGLE_TABLE, TWO_TABLE

HANDS_PER_PAIR = 2
TABLES = 2

#: Builds the two agents (A = seat 0, B = seat 1) of one table, already ``reset`` with their seeds.
#: Called with ``0`` / ``1`` for the two tables of ``two_table`` and with ``None`` for the single table.
TableAgents = Callable[[int | None], tuple[Agent, Agent]]


@dataclass(frozen=True, slots=True)
class DuplicateHand:
    """One played hand of a duplicate session."""

    deal_position: int  # index of the deal within the deals passed in (hand // 2 in layout order)
    a_on_button: bool  # agent A (seat 0) was on the button
    record: HandRecord


def play_duplicate_deals(
    game: GameConfig,
    deals: Sequence[Deal],
    make_agents: TableAgents,
    protocol: str = TWO_TABLE,
    *,
    first_hand_number: int = 0,
) -> Iterator[DuplicateHand]:
    """Play every deal twice and yield the hands in layout order (see the module docstring).

    ``make_agents(table)`` must return NEW agent instances for every call (they must not share mutable
    state, e.g. ``copy.copy`` of a prototype followed by ``reset(seed)``). ``first_hand_number`` offsets
    the hand numbers of the records: ``two_table`` numbers a table's hands ``0, 1, 2...`` (both plays of
    deal ``p`` are hand ``p`` of their table), ``single_table`` numbers all hands consecutively.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    if protocol == TWO_TABLE:
        tables = [(make_agents(t), (OpponentTracker(), OpponentTracker())) for t in range(TABLES)]
        for position, deal in enumerate(deals):
            for table, (agents, trackers) in enumerate(tables):
                a_on_button = (position + table) % 2 == 0
                record = play_hand(
                    game, deal, 0 if a_on_button else 1, agents,
                    hand_number=first_hand_number + position, trackers=trackers,
                )  # fmt: skip
                yield DuplicateHand(position, a_on_button, record)
        return
    agents, trackers = make_agents(None), (OpponentTracker(), OpponentTracker())
    for position, deal in enumerate(deals):
        for leg in range(HANDS_PER_PAIR):  # leg 0: A on the button, leg 1: B on the button
            record = play_hand(
                game, deal, leg, agents,
                hand_number=first_hand_number + HANDS_PER_PAIR * position + leg, trackers=trackers,
            )  # fmt: skip
            yield DuplicateHand(position, leg == 0, record)
