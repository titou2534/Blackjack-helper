"""
Append hands captured by the in-browser recorder to lagos_life_hands.csv.

Usage: python import_recorded.py recorded.json
The JSON is the list the recorder keeps in the page (window.__bjLog).
Hands whose round number is already in the CSV are skipped, so it is safe
to import the same capture twice.
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "lagos_life_hands.csv")
RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"}
SUITS = {"♠", "♥", "♦", "♣"}
STAKE = 100_000  # your usual stake; a 3-card hand at 2x this was a double


def naira(s):
    s = s.replace(",", "").replace("₦", "").strip().upper()
    mult = 1
    if s.endswith("K"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    return round(float(s) * mult)


def infer_actions(hands, bets):
    """Best guess at what was pressed, from the cards and bet size."""
    out = []
    base = min(bets) if bets else 0
    for i, h in enumerate(hands):
        n = len(h["cards"])
        if len(hands) > 1:
            out.append("split")
        if (i < len(bets) and bets[i] > base) or (n == 3 and i < len(bets) and bets[i] >= 2 * STAKE):
            out.append("double")
        elif n > 2:
            out.append("hit" + (f" x{n - 2}" if n > 3 else ""))
        else:
            out.append("stand")
    return ", ".join(out)


def parse(entry):
    lines = [l.strip() for l in entry["seg"].split("\n") if l.strip()]
    sections, cur, result = [], None, []
    for l in lines:
        if re.match(r"^(DEALER|YOU)\s*·", l, re.I):  # section header, not "You won ..."
            cur = {"header": l, "cards": []}
            sections.append(cur)
        elif l in RANKS:
            if cur is not None:
                cur["cards"].append(l)
        elif l in SUITS:
            continue
        else:
            result.append(l)  # e.g. "You won · dealer bust · ₦193,000 back", "High-roller tax ..."
    result_line = " / ".join(result)
    dealer = next(s for s in sections if s["header"].upper().startswith("DEALER"))
    hands = [s for s in sections if s["header"].upper().startswith("YOU")]

    bets, statuses = [], []
    for h in hands:
        m = re.match(r"YOU\s*·\s*([^·]+?)\s*·\s*₦([\d.,]+[KkMm]?)\s*(.*)$", h["header"], re.I)
        if m:
            bets.append(naira(m.group(2)))
            statuses.append((m.group(3) or "").strip().lower())
        else:
            statuses.append("")
    payout = re.search(r"₦([\d.,]+[KkMm]?)\s*back", result_line)
    chips = re.search(r"Chips ₦([\d,]+) · Wallet ₦([\d,]+)", entry["chips"])
    tonight = re.search(r"You're (down|up) ₦([\d,]+) tonight · (\d+) rounds", entry["tonight"])
    total = None
    if tonight:
        total = naira(tonight.group(2)) * (-1 if tonight.group(1) == "down" else 1)
    return {
        "time": entry["at"],
        "player_cards": " | ".join(" ".join(h["cards"]) for h in hands),
        "dealer_cards": " ".join(dealer["cards"]),
        "actions": infer_actions(hands, bets),
        "bet": sum(bets) if bets else "",
        "result": result_line,
        "payout": naira(payout.group(1)) if payout else "",
        "chips_after": naira(chips.group(1)) if chips else "",
        "wallet_after": naira(chips.group(2)) if chips else "",
        "tonight_total": total if total is not None else "",
        "rounds_tonight": int(tonight.group(3)) if tonight else "",
        "notes": " | ".join(s for s in statuses if s) + (" (split)" if len(hands) > 1 else ""),
    }


def main():
    entries = json.load(open(sys.argv[1]))
    # The recorder saves every change on the table. When the round counter
    # ticks up to N, the table is showing finished hand N. Later saves with the
    # same counter are either that same hand (numbers settling) or the next
    # hand being dealt. Keep the last save that still shows hand N's cards.
    by_round = {}
    for e in entries:
        try:
            p = parse(e)
        except StopIteration:
            continue
        n = p["rounds_tonight"]
        if n == "":
            continue
        if n not in by_round:
            by_round[n] = p
        elif (p["player_cards"].startswith(by_round[n]["player_cards"].split(" | ")[0][:3])
              and p["dealer_cards"].startswith(by_round[n]["dealer_cards"][:3])
              and len(p["player_cards"]) >= len(by_round[n]["player_cards"])):
            by_round[n] = p
    # hands where the counter jumped by more than one were missed: flag them
    rounds = sorted(by_round)
    missed = [r for a, b in zip(rounds, rounds[1:]) for r in range(a + 1, b)]
    if missed:
        print(f"Warning: rounds {missed} were not captured (cards unknown).")
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    last_round = max(int(r["rounds_tonight"]) for r in rows if r["rounds_tonight"])
    next_hand = max(int(r["hand"]) for r in rows) + 1
    added = []
    for key in sorted(by_round, key=lambda k: (isinstance(k, str), k)):
        p = by_round[key]
        if int(p["rounds_tonight"]) <= last_round:  # already logged, or from before the baseline
            continue
        p["hand"] = next_hand
        next_hand += 1
        added.append(p)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        for p in added:
            w.writerow({k: p.get(k, "") for k in fields})
    print(f"Added {len(added)} new hand(s); skipped {len(by_round) - len(added)} already logged.")
    for p in added:
        print(f"  #{p['hand']}: you {p['player_cards']} vs dealer {p['dealer_cards']} -> "
              f"{p['result']}  (chips {p['chips_after']:,})" if p['chips_after'] != "" else "")


if __name__ == "__main__":
    main()
