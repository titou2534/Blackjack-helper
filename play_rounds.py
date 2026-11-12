"""
Let the trained model play blackjack and see how it does.

Two tables:
  1. Gymnasium Blackjack-v1: an independent, open-source simulator. Hit/stand
     only (no double or split), blackjack pays 3:2 (natural=True).
  2. Our own full-rules table: double, split, dealer peeks for blackjack.

On each table, the model plays against a baseline "copy the dealer" player
(hit below 17, otherwise stand). Both players see the exact same cards
(same random seed), so the difference is down to the decisions.

Usage:
    python play_rounds.py                 # 1,000 rounds per table
    python play_rounds.py --rounds 100000
"""
import argparse
import random

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd

from blackjack_model import (CARDS, DECK, MODEL_PATH, RULES, add_card, dealer_must_hit,
                             predict_table, state_label)

# ----------------------------------------------------------------------------
# Turn the model into a lookup: (state, dealer card) -> EV for each action
# ----------------------------------------------------------------------------
clf, reg = joblib.load(MODEL_PATH)
TABLE = predict_table(clf, reg)
EV = {(r.state, r.dealer_up, r.action): r.model_ev for r in TABLE.itertuples()}


def model_move(total, soft, up, can_double=False, pair=0):
    """Best action according to the model. Returns 'stand', 'hit', 'double' or 'split'."""
    if total >= 20 or (total == 21):
        return "stand"
    if pair:
        label = "pair A,A" if pair == 1 else f"pair {pair},{pair}"
    elif (not soft and total < 5) or (soft and total < 13):
        return "hit"  # hands not in the table (2,2 / A,A without splitting) are always hits
    else:
        label = state_label(total, soft, 0)
    options = ["stand", "hit"] + (["double"] if can_double else []) + (["split"] if pair else [])
    return max(options, key=lambda a: EV[(label, up, a)])


def dealer_copy_move(total, soft, up, can_double=False, pair=0):
    return "hit" if total < 17 else "stand"


PLAYERS = {"ML model": model_move, "Copy the dealer": dealer_copy_move}


# ----------------------------------------------------------------------------
# Table 1: Gymnasium
# ----------------------------------------------------------------------------
def play_gymnasium(policy, rounds, seed):
    env = gym.make("Blackjack-v1", natural=True, sab=False)
    results = []
    for i in range(rounds):
        obs, _ = env.reset(seed=seed + i)
        done, reward = False, 0.0
        while not done:
            total, up, usable_ace = obs
            move = policy(total, bool(usable_ace), up)
            obs, reward, terminated, truncated, _ = env.step(0 if move == "stand" else 1)
            done = terminated or truncated
        results.append(reward)
    env.close()
    return np.array(results)


# ----------------------------------------------------------------------------
# Table 2: our own full-rules game
# ----------------------------------------------------------------------------
def play_own_table(policy, rounds, seed):
    rng = random.Random(seed)
    draw = lambda: DECK[rng.randrange(13)]
    results = []
    for _ in range(rounds):
        p1, p2, up, hole = draw(), draw(), draw(), draw()
        pt, ps = add_card(*add_card(0, False, p1), p2)
        dt, ds = add_card(*add_card(0, False, up), hole)
        player_bj, dealer_bj = pt == 21, dt == 21
        if dealer_bj:  # dealer peeks: round ends right away
            results.append(0.0 if player_bj else -1.0)
            continue
        if player_bj:
            results.append(RULES["blackjack_pays"])
            continue

        def finish(t, s, can_double):
            """Play out one hand. Returns (final total, bet multiplier)."""
            first = True
            while t < 21:
                move = policy(t, s, up, can_double=can_double and first)
                if move == "stand":
                    break
                t, s = add_card(t, s, draw())
                if move == "double":
                    return t, 2
                first = False
            return t, 1

        pair = p1 if p1 == p2 else 0
        move = policy(pt, ps, up, can_double=True, pair=pair)
        hands = []
        if move == "split":
            for _ in range(2):
                t, s = add_card(*add_card(0, False, pair), draw())
                hands.append((t, 1) if pair == 1 else finish(t, s, can_double=False))
        else:
            if move == "double":
                t, _ = add_card(pt, ps, draw())
                hands.append((t, 2))
            elif move == "stand":
                hands.append((pt, 1))
            else:
                t, s = add_card(pt, ps, draw())
                hands.append(finish(t, s, can_double=False))

        while dealer_must_hit(dt, ds):
            dt, ds = add_card(dt, ds, draw())
        net = 0
        for t, bet in hands:
            if t > 21:
                net -= bet
            elif dt > 21 or t > dt:
                net += bet
            elif t < dt:
                net -= bet
        results.append(float(net))
    return np.array(results)


def summarize(name, table, res):
    return dict(table=table, player=name, rounds=len(res),
                won=int((res > 0).sum()), pushed=int((res == 0).sum()), lost=int((res < 0).sum()),
                net_units=res.sum(), return_pct=100 * res.mean(),
                margin_pct=100 * 1.96 * res.std() / np.sqrt(len(res)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rows = []
    for name, policy in PLAYERS.items():
        rows.append(summarize(name, "Gymnasium (hit/stand)", play_gymnasium(policy, args.rounds, args.seed)))
        rows.append(summarize(name, "Own table (full rules)", play_own_table(policy, args.rounds, args.seed)))
    df = pd.DataFrame(rows).sort_values(["table", "player"], ascending=[True, False])

    print(f"\n{args.rounds:,} rounds per table, $1 bet per round\n")
    for table, g in df.groupby("table", sort=False):
        print(table)
        print(f"  {'player':<17}{'won':>7}{'pushed':>8}{'lost':>7}{'net $':>9}{'return':>9}   (luck margin)")
        for r in g.itertuples():
            print(f"  {r.player:<17}{r.won:>7}{r.pushed:>8}{r.lost:>7}{r.net_units:>+9.1f}"
                  f"{r.return_pct:>+8.2f}%   (+/- {r.margin_pct:.1f}%)")
        print()
    print("Return = average money won per $1 bet. Luck margin = how far the result could")
    print("swing from random luck alone (95% range); it shrinks as you play more rounds.")


if __name__ == "__main__":
    main()
