"""
Compare every logged Lagos Life hand with what the model would have done.

Usage: python check_decisions.py
"""
import csv
import os

import joblib

import blackjack_model as bm

HERE = os.path.dirname(os.path.abspath(__file__))
clf, reg = joblib.load(bm.MODEL_PATH)
TABLE = bm.predict_table(clf, reg)
STATES = set(TABLE.state)


def val(c):
    return 1 if c == "A" else 10 if c in ("J", "Q", "K", "10") else int(c)


def model_pick(t, soft, up, pair, first):
    """Model's move. first=True means double/split are still allowed."""
    if t >= 20 and not pair:
        return "stand"
    if pair:
        label = "pair A,A" if pair == 1 else f"pair {pair},{pair}"
    else:
        label = bm.state_label(t, soft, 0)
    if label not in STATES:
        return "hit"
    g = TABLE[(TABLE.state == label) & (TABLE.dealer_up == up)]
    if not first:
        g = g[g.action.isin(["stand", "hit"])]
    best = g.sort_values("model_ev").iloc[-1]
    return best.action, g.set_index("action")["model_ev"]


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "lagos_life_hands.csv"))))
    agree = total = 0
    costly = []
    for r in rows:
        cards = r["player_cards"].split()
        if not cards or "|" in r["player_cards"]:
            continue  # skip baseline, missed hands and splits
        if "dealer has blackjack" in r["result"]:
            continue  # dealer peeked and ended the hand: no decision was made
        up = val(r["dealer_cards"].split()[0])
        doubled = r["actions"].startswith("double")
        t, s = 0, False
        steps = []
        for i, c in enumerate(cards):
            t, s = bm.add_card(t, s, val(c))
            if i == 0 or t > 21:
                continue
            last = i == len(cards) - 1
            did = "double" if (i == 1 and doubled) else ("stand" if last else "hit")
            pair = val(cards[0]) if i == 1 and val(cards[0]) == val(cards[1]) else 0
            pick = model_pick(t, s, up, pair, first=(i == 1))
            rec, evs = (pick, None) if isinstance(pick, str) else pick
            total += 1
            note = ""
            if did == rec:
                agree += 1
            else:
                gap = (evs[rec] - evs[did]) if evs is not None and did in evs else float("nan")
                note = f" (model: {rec}, cost {gap:.2f}/bet)"
                costly.append((r["hand"], gap))
            steps.append(f"{'soft ' if s else ''}{t}{' pair' if pair else ''}: {did}{note}")
            if did in ("stand", "double"):
                break
        print(f"  #{r['hand']:>3} vs {r['dealer_cards'].split()[0]:<2}  " + " -> ".join(steps)
              + f"   => {r['result'].split(' / ')[0]}")
    if total:
        print(f"\nYou matched the model on {agree} of {total} decisions ({agree / total:.0%}).")


if __name__ == "__main__":
    main()
