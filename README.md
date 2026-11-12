# Blackjack Outcome Model

A machine-learning model that predicts the chance of **winning, pushing or losing** for every blackjack move (stand, hit, double, split) in every situation, then ranks the moves by expected value.

It was trained on simulated hands, checked against exact probability math, tested in two simulated games, and then used for 100 hands in a live online game, Lagos Life's Eko Casino.

```
Your hand: 8 8
Dealer's face-up card: 7

pair 8,8 vs dealer 7
  rank action    P(win)  P(push)  P(lose)      EV   exact EV
  1    split      38.6%    28.4%    33.0%  +0.115     +0.164
  2    hit        25.8%     5.8%    68.4%  -0.423     -0.415
  3    stand      24.9%     0.0%    75.1%  -0.484     -0.475
  4    double     26.9%     5.3%    67.9%  -0.829     -0.830
  -> best move: SPLIT
```

EV (expected value) is the average amount won or lost per 1 unit bet. Ranking by EV, not just win chance, matters: doubling can win as often as hitting but loses twice as much when it doesn't.

## How it works

1. **Simulate**: play 3 million hands. Each starts from a random situation (player total, soft/hard, pair) against a random dealer upcard, takes a random action, and records the payout. Every situation is sampled equally often, so rare hands like 6,6 get enough data.
2. **Train**: two gradient-boosted tree models (scikit-learn `HistGradientBoosting`):
   - a classifier gives P(win), P(push) and P(lose)
   - a regressor gives EV
   Inputs: player total, soft or not, pair rank, dealer upcard, action.
3. **Rank**: score all 330 situations × every legal action and recommend the highest-EV move.
4. **Validate**: compare against an exact dynamic-programming solver for the same rules.

Rules: endless deck, dealer stands on soft 17 and peeks for blackjack, blackjack pays 3:2, double on any two cards, split once, split aces get one card, no surrender.

## Results

### 1. Against exact math (is the model correct?)
| Metric | Result |
|---|---|
| Best move matches the exact optimal move | **96%** of 330 situations |
| Mean error in P(win) | **0.6%** |
| Cost of the disagreements | at most 0.06 units per hand; all are near-ties |

### 2. Simulated play, 100,000 rounds (does it help?)
Compared with a "copy the dealer" baseline (hit below 17, never double or split), dealt identical cards:

| Table | ML model | Copy the dealer |
|---|---|---|
| Own table (full rules) | **−1.05%** | −5.30% |
| [Gymnasium Blackjack-v1](https://gymnasium.farama.org/environments/toy_text/blackjack/) (independent, hit/stand only) | **−2.49%** | −5.28% |

The model loses about 5× less than the baseline with full rules. It still loses, as expected: blackjack keeps a house edge even with perfect play (about −0.5% for these rules, within the simulation's ±0.7% margin).

### 3. Live game: Lagos Life Eko Casino (does it hold up in the real world?)
Lagos Life is a browser life simulator with a play-money casino. Its posted rules almost match the model's, but the VIP table adds a **7% tax on winnings**.

A read-only recorder in the browser captured every finished hand. Results over 100 tracked hands at a ₦100k stake:

| Metric | Result |
|---|---|
| Decisions matching the model | **129 / 129** |
| Won / pushed / lost | 40 / 9 / 51 |
| Net | −₦1,322,000 (−12%), of which ₦322,000 was tax |
| Expected with perfect play | about −₦385,000; the rest is normal short-run luck (about ±₦1.1m over 100 hands) |

**How their game is built:** 541 cards seen match a fair deck (10-value cards 30.5% vs 30.8% expected; chi-square 7.8 on 9 degrees of freedom). The rules work as posted. The house edge comes from the 7% tax, which moves the expected loss from about 0.5% to about 3.5% per hand, not from rigged cards.

### Which test is most realistic?
The 100,000-round simulation is the best predictor of real-world results: full rules, enough hands for luck to average out, and close to the published house edge. The live test is the most authentic but too short to judge profit. Gymnasium is independent but leaves out doubles and splits.

## Files

| File | Purpose |
|---|---|
| `blackjack_model.py` | Simulator, exact solver, model training and the interactive advisor |
| `play_rounds.py` | Plays N rounds on Gymnasium and the full-rules table vs the baseline |
| `check_decisions.py` | Grades every logged live hand against the model |
| `import_recorded.py` | Turns the browser recorder's captures into log rows |
| `blackjack_predictions.csv` | Model and exact probabilities for every situation and action |
| `lagos_life_hands.csv` | The 100 live hands (plus baseline row) |
| `lagos_life_hands_session1.csv` | An earlier 14-hand session |
| `blackjack_model.joblib` | Trained models, so the advisor starts instantly |

## Usage

```bash
pip install -r requirements.txt

python blackjack_model.py                  # interactive advisor: type your hand and the dealer's card
python blackjack_model.py --summary        # strategy chart + accuracy vs exact math
python blackjack_model.py --retrain        # re-simulate and retrain
python play_rounds.py --rounds 100000      # simulated play vs the baseline
python check_decisions.py                  # grade the logged live hands
```

Advisor input examples: `16`, `soft 18`, `pair 8`, `10 6`, `A 7`, `8 8`, `5 4 3`. Type the actual cards to see split options.

## Limitations and next steps

- Endless deck: fine for basic strategy, but card counting would need a finite shoe.
- One rule set. Many casinos hit soft 17, pay 6:5 on blackjack, or offer surrender and double after split.
- In this setup the exact solver is the ground truth, so the ML model can't beat it. The ML approach pays off where exact math is impractical: a finite shoe that depends on cards already dealt, or learning from real game logs.
- Next: add rule switches (H17, 6:5, DAS, surrender, tax) and a finite shoe to predict any specific table's expected return.

*All live testing used play money only.*
