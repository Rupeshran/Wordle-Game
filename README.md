# 🎯 AI Wordle Game — Entropy Edition

An interactive terminal Wordle game with a built-in AI assistant powered by **information theory and entropy**. After every guess, the solver tells you which words are the best next moves — ranked by how many bits of information they are expected to reveal.

---

## 👥 Team Members

1. Harshit Rajesh Benke
2. Piyush Kumar Thakur
3. Abhishek Kumar
4. Rupesh Singh Rana
5. Kattamuri Sri Gayatri
6. Katta Keerthi Priya
7. Moka Sowmya
8. Bangari Sri Joshna

---

## ⚙️ How It Works

The AI assistant uses **Shannon entropy** (H = −Σ p·log₂p) to score every possible guess before you type anything.

For each candidate word it asks: *"If I guessed this word, how many different colour patterns could I see, and how evenly would those patterns split the remaining answer pool?"* The word that produces the most even split — and therefore the highest expected information gain in **bits** — is ranked first.

After you submit a guess, the game:
1. Shows how many **bits** your guess actually gained (e.g. `+3.46 bits`).
2. Filters the answer pool down to only words still consistent with the colour pattern.
3. Re-scores every word in the vocabulary against the new, smaller pool.

This is the same approach explained by [3Blue1Brown on YouTube](https://www.youtube.com/watch?v=v68zYyaEmEA).

### Pattern encoding

Each of the five tiles is Green (2), Yellow (1), or Gray (0). The game encodes the five-tile result as a single base-3 integer (range 0–242), which makes pattern lookup and filtering very fast.

### Precomputed matrix

On startup the game builds a `vocabulary × answers` pattern matrix — every possible (guess, answer) pair is computed once and stored as a bytearray. All entropy calculations during play are simple lookups into this table, so suggestions appear instantly.

---

## 🚀 How to Run

### 1. Requirements

- **Python 3.10 or newer** — download from [python.org](https://www.python.org/downloads/)
- No external packages are required. The game uses only the Python standard library (`math`, `random`, `collections`, `time`).

### 2. Clone or download the project
```bash
git clone 
cd Wordle-Game-main
```

Or unzip the downloaded archive and open a terminal inside the `Wordle-Game-main` folder.

### 3. Check the data files are present
Wordle-Game-main/
├── wordle.py
└── data/
├── allowed_words.txt   ← answer pool  (~2 300 words)
└── possible_words.txt  ← full vocabulary for guessing (~12 000 words)

Both files are included in the repository. If you move them, update the paths at the top of `wordle.py`:
```python
ANSWERS_PATH = "data/allowed_words.txt"
GUESSES_PATH = "data/possible_words.txt"
```

### 4. Start the game
```bash
python wordle.py
```

**First launch only** — the game precomputes ~27 MB of pattern data (~12 000 × 2 300 pairs). This takes roughly 30–90 seconds depending on your machine. Subsequent rounds in the same session are instant because the matrix is kept in memory.

---

## 🎮 Playing the Game
──────────────────────────────────────────────────
W O R D L E   +   E N T R O P Y
──────────────────────────────────────────────────
Answer pool : 2,309 words   Starting entropy: 11.17 bits
Vocabulary  : 12,972 words
Top 5 suggestions  (expected bits gained):

SOARE  ████████████  6.371 bits    guess
RAISE  ███████████░  6.236 bits  ✓ answer
ROATE  ███████████░  6.210 bits    guess
...

Guess 1/6: raise

- Type any valid 5-letter word and press Enter.
- The board updates with colour tiles: 🟩 Green = right letter, right spot · 🟨 Yellow = right letter, wrong spot · ⬛ Gray = not in word.
- The AI suggestions update after every guess.
- `✓ answer` next to a suggestion means it is still a valid solution, not just a good guess.
- At the end of each game you are asked `Play again? (y/n)`.

---

## 🔧 Configuration

Open `wordle.py` and edit the constants at the top of the file:

| Constant | Default | What it controls |
|---|---|---|
| `ANSWERS_PATH` | `"data/allowed_words.txt"` | File the secret word is drawn from |
| `GUESSES_PATH` | `"data/possible_words.txt"` | Full vocabulary scored for entropy |
| `TOP_N` | `5` | Number of suggestions shown each turn |
| `WORD_LEN` | `5` | Word length (change for variants) |
| `MAX_GUESSES` | `6` | Maximum allowed guesses |

---

## 📂 Project Structure
```
Wordle-Game-main/
├── wordle.py
│   └── (game loop, entropy engine, rendering)
├── data/
│   ├── allowed_words.txt    ← answer pool (~2,300 words the game picks from)
│   └── possible_words.txt   ← full guess vocabulary (~12,000 words scored for entropy)
├── README.md
├── LICENSE
└── .gitignore
```


---

## 📈 References

- 3Blue1Brown, [*Solving Wordle using information theory*](https://www.youtube.com/watch?v=v68zYyaEmEA), YouTube, February 2022.
- [`3b1b/videos`](https://github.com/3b1b/videos/tree/master/_2022/wordle) — supplementary Python code from the video.

---

## 📊 Status

✅ Ready to play — no installation beyond Python 3.10+ required.
