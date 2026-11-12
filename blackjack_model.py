"""
Blackjack outcome model.

1. SIMULATE millions of hands. Each one starts from a random player state
   (total, soft/hard, pair) vs. a dealer upcard, takes a random action
   (stand / hit / double / split), and records the result.
2. TRAIN two gradient-boosted models on that data:
     - a classifier -> P(win), P(push), P(lose) for (state, action)
     - a regressor  -> expected value (EV) in betting units for (state, action)
3. RANK the actions for every state and print/save the tables.
4. VALIDATE against an exact probability solver, so you can see how close
   the learned model gets to the true answer.

Rules assumed (change in RULES below): infinite deck (like a continuous
shuffler), dealer stands on soft 17, dealer peeks for blackjack, blackjack
pays 3:2, double on any first two cards, split once, split aces get one card,
no surrender.

"Hit" means: take a card now, then keep playing hit/stand optimally.

Usage:
    python blackjack_model.py                    # type in your hand + dealer card
    python blackjack_model.py --summary          # strategy chart + accuracy report
    python blackjack_model.py --retrain          # re-simulate and retrain
    python blackjack_model.py --hands 4000000    # more data = more accurate
    python blackjack_model.py --ask 16 --dealer 10
    python blackjack_model.py --ask 18 --soft --dealer 9
    python blackjack_model.py --ask 16 --pair --dealer 10   # 8,8
"""
import argparse
import functools
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

RULES = dict(dealer_hits_soft17=False, blackjack_pays=1.5)

CARDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]          # 1 = ace, 10 = any ten-value card
PROBS = [1 / 13] * 9 + [4 / 13]
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
ACTIONS = ["stand", "hit", "double", "split"]
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "blackjack_model.joblib")
CSV_PATH = os.path.join(HERE, "blackjack_predictions.csv")


def card_name(c):
    return "A" if c == 1 else str(c)


def add_card(total, soft, card):
    """Add a card to a hand given as (best total, is_soft)."""
    if card == 1:
        if total + 11 <= 21:
            return total + 11, True
        total += 1
    else:
        total += card
    if total > 21 and soft:
        total, soft = total - 10, False
    return total, soft


def dealer_must_hit(total, soft):
    return total < 17 or (total == 17 and soft and RULES["dealer_hits_soft17"])


# ----------------------------------------------------------------------------
# Exact solver (ground truth). Returns arrays [EV, P(win), P(push), P(lose)].
# ----------------------------------------------------------------------------
@functools.lru_cache(None)
def _dealer_final(total, soft):
    if total > 21:
        return {22: 1.0}
    if not dealer_must_hit(total, soft):
        return {total: 1.0}
    out = {}
    for c, p in zip(CARDS, PROBS):
        for k, v in _dealer_final(*add_card(total, soft, c)).items():
            out[k] = out.get(k, 0) + p * v
    return out


@functools.lru_cache(None)
def dealer_dist(up):
    """Dealer's final total distribution given upcard, conditioned on no dealer blackjack."""
    t, s = add_card(0, False, up)
    out, norm = {}, 0.0
    for c, p in zip(CARDS, PROBS):
        if (up == 1 and c == 10) or (up == 10 and c == 1):
            continue
        norm += p
        for k, v in _dealer_final(*add_card(t, s, c)).items():
            out[k] = out.get(k, 0) + p * v
    return {k: v / norm for k, v in out.items()}


@functools.lru_cache(None)
def exact_stand(t, up):
    if t > 21:
        return (-1.0, 0.0, 0.0, 1.0)
    w = p = l = 0.0
    for d, pr in dealer_dist(up).items():
        if d == 22 or d < t:
            w += pr
        elif d == t:
            p += pr
        else:
            l += pr
    return (w - l, w, p, l)


@functools.lru_cache(None)
def exact_best(t, soft, up):
    """Play hit/stand optimally from here."""
    s = exact_stand(t, up)
    if t >= 21:
        return s
    h = exact_hit(t, soft, up)
    return h if h[0] > s[0] else s


@functools.lru_cache(None)
def exact_hit(t, soft, up):
    acc = np.zeros(4)
    for c, p in zip(CARDS, PROBS):
        acc += p * np.array(exact_best(*add_card(t, soft, c), up))
    return tuple(acc)


def exact_double(t, soft, up):
    acc = np.zeros(4)
    for c, p in zip(CARDS, PROBS):
        acc += p * np.array(exact_stand(add_card(t, soft, c)[0], up))
    acc[0] *= 2
    return tuple(acc)


def exact_split_ev(card, up):
    t0, s0 = add_card(0, False, card)
    ev = 0.0
    for c, p in zip(CARDS, PROBS):
        t, s = add_card(t0, s0, c)
        ev += p * (exact_stand(t, up)[0] if card == 1 else exact_best(t, s, up)[0])
    return 2 * ev


def should_hit(t, soft, up):
    return t < 21 and exact_hit(t, soft, up)[0] > exact_stand(t, up)[0]


# ----------------------------------------------------------------------------
# Simulator (generates the training data)
# ----------------------------------------------------------------------------
def draw(rng):
    return DECK[rng.randrange(13)]


def sim_dealer(up, rng):
    t, s = add_card(0, False, up)
    while True:  # dealer peeked: re-draw hole card if it would make blackjack
        hole = draw(rng)
        if not ((up == 1 and hole == 10) or (up == 10 and hole == 1)):
            break
    t, s = add_card(t, s, hole)
    while dealer_must_hit(t, s):
        t, s = add_card(t, s, draw(rng))
    return t


def play_on(t, s, up, rng):
    while should_hit(t, s, up):
        t, s = add_card(t, s, draw(rng))
    return t


def settle(p, d):
    if p > 21:
        return -1
    if d > 21 or p > d:
        return 1
    return 0 if p == d else -1


def simulate(n, seed=0):
    # Sample every decision state equally often (rather than at natural deal
    # frequency) so rare hands like soft 13 or 6,6 still get plenty of data.
    rng = random.Random(seed)
    grid = state_grid()
    rows = []
    while len(rows) < n:
        t, s, pair = grid[rng.randrange(len(grid))]
        s = bool(s)
        up = rng.choice(CARDS)
        legal = 4 if pair else 3
        a = rng.randrange(legal)
        d = sim_dealer(up, rng)
        if a == 0:
            pay = settle(t, d)
        elif a == 1:
            pay = settle(play_on(*add_card(t, s, draw(rng)), up, rng), d)
        elif a == 2:
            pay = 2 * settle(add_card(t, s, draw(rng))[0], d)
        else:
            pay = 0
            for _ in range(2):
                ht, hs = add_card(*add_card(0, False, pair), draw(rng))
                if pair != 1:
                    ht = play_on(ht, hs, up, rng)
                pay += settle(ht, d)
        rows.append((t, int(s), up, pair, a, pay))
    return pd.DataFrame(rows, columns=["total", "soft", "dealer_up", "pair", "action", "payoff"])


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
FEATURES = ["total", "soft", "dealer_up", "pair", "action"]


def train(df):
    X = df[FEATURES].to_numpy()
    y_cls = np.sign(df["payoff"].to_numpy()).astype(int) + 1  # 0 lose, 1 push, 2 win
    cat = [False, True, True, True, True]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                         categorical_features=cat, random_state=0)
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                        categorical_features=cat, random_state=0)
    clf.fit(X, y_cls)
    reg.fit(X, df["payoff"].to_numpy())
    return clf, reg


def state_grid():
    """All first-decision states: hard 5-19, soft 13-20, and every pair."""
    states = [(t, 0, 0) for t in range(5, 20)] + [(t, 1, 0) for t in range(13, 21)]
    states += [(2 * c, 0, c) for c in range(2, 11)] + [(12, 1, 1)]
    return states


def state_label(t, soft, pair):
    if pair:
        return f"pair {card_name(pair)},{card_name(pair)}"
    return f"{'soft' if soft else 'hard'} {t}"


def predict_table(clf, reg):
    rows = []
    for t, soft, pair in state_grid():
        for up in CARDS:
            for a in range(4 if pair else 3):
                rows.append((t, soft, up, pair, a))
    X = np.array(rows)
    proba = clf.predict_proba(X)
    ev = reg.predict(X)
    df = pd.DataFrame(rows, columns=FEATURES)
    df["p_lose"], df["p_push"], df["p_win"] = proba[:, 0], proba[:, 1], proba[:, 2]
    df["model_ev"] = ev
    # exact values for comparison
    ex = []
    for t, soft, up, pair, a in rows:
        if a == 0:
            ex.append(exact_stand(t, up))
        elif a == 1:
            ex.append(exact_hit(t, bool(soft), up))
        elif a == 2:
            ex.append(exact_double(t, bool(soft), up))
        else:
            ex.append((exact_split_ev(pair, up), np.nan, np.nan, np.nan))
    ex = np.array(ex)
    df["exact_ev"], df["exact_p_win"], df["exact_p_push"], df["exact_p_lose"] = ex.T
    df["state"] = [state_label(t, s, p) for t, s, _, p, _ in rows]
    df["dealer"] = df["dealer_up"].map(card_name)
    df["action"] = df["action"].map(dict(enumerate(ACTIONS)))
    df["model_rank"] = df.groupby(["state", "dealer_up"])["model_ev"].rank(ascending=False).astype(int)
    df["exact_rank"] = df.groupby(["state", "dealer_up"])["exact_ev"].rank(ascending=False).astype(int)
    return df


def print_state(df, label, up=None):
    sub = df[df["state"] == label]
    if up is not None:
        sub = sub[sub["dealer_up"] == up]
    for u, g in sub.groupby("dealer_up"):
        g = g.sort_values("model_ev", ascending=False)
        print(f"\n{label} vs dealer {card_name(u)}")
        print(f"  {'rank':<5}{'action':<8}{'P(win)':>8}{'P(push)':>9}{'P(lose)':>9}{'EV':>8}   {'exact EV':>8}")
        for _, r in g.iterrows():
            print(f"  {r.model_rank:<5}{r.action:<8}{r.p_win:>8.1%}{r.p_push:>9.1%}{r.p_lose:>9.1%}"
                  f"{r.model_ev:>+8.3f}   {r.exact_ev:>+8.3f}")
        print(f"  -> best move: {g.iloc[0].action.upper()}")


def strategy_chart(df, col):
    best = df.loc[df.groupby(["state", "dealer_up"])[col].idxmax()]
    abbrev = dict(stand="S", hit="H", double="D", split="P")
    chart = best.pivot(index="state", columns="dealer_up", values="action").replace(abbrev)
    order = [state_label(*s) for s in state_grid()]
    chart = chart.loc[order]
    chart.columns = [card_name(c) for c in chart.columns]
    return chart[[str(c) for c in range(2, 11)] + ["A"]]


def parse_card(tok):
    tok = tok.strip().upper()
    if tok in ("A", "ACE", "1", "11"):
        return 1
    if tok in ("J", "Q", "K"):
        return 10
    v = int(tok)
    if not 2 <= v <= 10:
        raise ValueError
    return v


def parse_hand(text):
    """Turn what the user typed into a table label, or None if it can't be read.

    Accepts: a total ("16"), "soft 18", "pair 8", or the cards themselves
    ("10 6", "A 7", "8 8", "5,4,3").
    """
    words = text.lower().replace(",", " ").split()
    if not words:
        return None
    if words[0] == "soft" and len(words) == 2:
        t, soft, pair = int(words[1]), True, 0
    elif words[0] == "pair" and len(words) == 2:
        c = parse_card(words[1])
        return "pair A,A" if c == 1 else f"pair {c},{c}"
    elif words[0] == "hard" and len(words) == 2:
        t, soft, pair = int(words[1]), False, 0
    elif len(words) == 1:
        t, soft, pair = int(words[0]), False, 0
    else:
        cards = [parse_card(w) for w in words]
        t, soft = 0, False
        for c in cards:
            t, soft = add_card(t, soft, c)
        if len(cards) == 2 and cards[0] == cards[1]:
            return "pair A,A" if cards[0] == 1 else f"pair {cards[0]},{cards[0]}"
        pair = 0
    if t > 21:
        return "BUST"
    if t >= 20 or (soft and t == 21):
        return "STAND"
    if t == 21:
        return "STAND"
    if (not soft and t < 5) or (soft and t < 13):
        return None
    return state_label(t, soft, pair)


def interactive(df):
    print("\nBlackjack advisor. Type your hand, then the dealer's card.")
    print("  Hand examples: 16   soft 18   pair 8   10 6   A 7   8 8   5 4 3")
    print("  Dealer card:   2-10, J/Q/K, or A   (press Enter for all dealer cards)")
    print("  Type q to quit.\n")
    while True:
        try:
            hand = input("Your hand: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if hand.lower() in ("q", "quit", "exit"):
            return
        try:
            label = parse_hand(hand)
        except ValueError:
            label = None
        if label is None:
            print("  Couldn't read that hand. Try e.g. 16, soft 18, pair 8, or 10 6.\n")
            continue
        if label in ("BUST", "STAND"):
            print("  You're over 21: that hand is bust.\n" if label == "BUST"
                  else "  20 or 21: always STAND.\n")
            continue
        while True:
            try:
                d = input("Dealer's face-up card: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if d.lower() in ("q", "quit", "exit"):
                return
            try:
                up = parse_card(d) if d else None
                break
            except ValueError:
                if d.isdigit() and int(d) > 11:
                    print(f"  {d} is a total, not a card. Enter just the dealer's ONE face-up card")
                    print("  (2-10, J/Q/K, or A). The model works from that card alone, since the")
                    print("  other one is hidden.")
                else:
                    print("  Enter the dealer's face-up card: 2-10, J/Q/K, or A.")
        words = hand.lower().replace(",", " ").split()
        if len(words) == 1 and words[0].isdigit():
            n = int(words[0])
            if n % 2 == 0 and 4 <= n <= 20:
                print(f"  Tip: if this is a pair ({n // 2},{n // 2}), type '{n // 2} {n // 2}' instead to see SPLIT.")
        if hand.lower().strip() == "soft 12":
            print("  Tip: soft 12 is A,A. Type 'A A' to see SPLIT.")
        if len(words) >= 3:  # 3+ cards: doubling and splitting are no longer allowed
            sub = df[df["action"].isin(["stand", "hit"])].copy()
            sub["model_rank"] = sub.groupby(["state", "dealer_up"])["model_ev"].rank(ascending=False).astype(int)
            print_state(sub, label, up)
        else:
            print_state(df, label, up)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="print the strategy chart and accuracy report")
    ap.add_argument("--hands", type=int, default=2_000_000)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--ask", type=int, help="player total to look up")
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--pair", action="store_true", help="hand is a pair (e.g. --ask 16 --pair = 8,8)")
    ap.add_argument("--dealer", type=str, help="dealer upcard: 2-10 or A")
    args = ap.parse_args()

    if os.path.exists(MODEL_PATH) and not args.retrain:
        clf, reg = joblib.load(MODEL_PATH)
    else:
        t0 = time.time()
        print(f"Simulating {args.hands:,} hands...")
        data = simulate(args.hands)
        print(f"  done in {time.time() - t0:.0f}s. Training models...")
        t0 = time.time()
        clf, reg = train(data)
        print(f"  done in {time.time() - t0:.0f}s.")
        joblib.dump((clf, reg), MODEL_PATH)

    df = predict_table(clf, reg)
    df.to_csv(CSV_PATH, index=False)

    if args.ask:
        up = None
        if args.dealer:
            up = 1 if args.dealer.upper() == "A" else int(args.dealer)
        if args.pair:
            label = "pair A,A" if args.ask == 12 and args.soft else f"pair {card_name(args.ask // 2)},{card_name(args.ask // 2)}"
        else:
            label = state_label(args.ask, args.soft, 0)
        print_state(df, label, up)
        return

    if not (args.summary or args.retrain):
        interactive(df)
        return

    print_state(df, "hard 16", 10)
    print_state(df, "hard 12", 4)
    print_state(df, "hard 11", 6)

    mc, ec = strategy_chart(df, "model_ev"), strategy_chart(df, "exact_ev")
    print("\nLearned strategy chart (S=stand H=hit D=double P=split):")
    print(mc.to_string())
    agree = (mc == ec).to_numpy().mean()
    print(f"\nModel agrees with the exact optimal move in {agree:.1%} of {mc.size} situations.")
    diff = (mc != ec)
    if diff.any().any():
        print("Disagreements (model vs exact) are almost always near-ties:")
        for st in diff.index:
            for up in diff.columns:
                if diff.loc[st, up]:
                    u = 1 if up == "A" else int(up)
                    g = df[(df.state == st) & (df.dealer_up == u)].set_index("action")
                    m, e = mc.loc[st, up], ec.loc[st, up]
                    inv = {v: k for k, v in dict(stand="S", hit="H", double="D", split="P").items()}
                    gap = g.loc[inv[e], "exact_ev"] - g.loc[inv[m], "exact_ev"]
                    print(f"  {st:<11} vs {up:<2}: model {m}, exact {e}  (true EV cost {gap:.3f} units)")
    err = (df["model_ev"] - df["exact_ev"]).abs().mean()
    pe = (df["p_win"] - df["exact_p_win"]).abs().mean()
    print(f"Mean |EV error| = {err:.3f} units, mean |P(win) error| = {pe:.1%}")
    print(f"\nFull table ({len(df)} rows) saved to {os.path.basename(CSV_PATH)}")


if __name__ == "__main__":
    main()
