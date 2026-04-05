# 🎯 AI Wordle Game


## 📌 Description
AI-based Wordle game using Bayes Theorem and information theory for intelligent word prediction. This project applies solvers to Wordle and Dungleon using the approach popularized by 3Blue1Brown.

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

## ⚙️ Features
- Solves Wordle puzzles using information theory and Bayes Theorem.
- Supports both Wordle and Dungleon games.
- Includes pre-computed optimal guesses and simulations.
- Exposes various command-line arguments to customize simulations (e.g., hard mode, optimizing for uniform distribution).

---

## 🚀 How to Run

### Requirements
- Install the latest version of [Python 3.X][python-download-url] (at least version 3.10).
- Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage
To print an exhaustive list of command-line arguments, run:

```bash
python simulations.py --help
```

Choose the game with `--game-name`:

```bash
python simulations.py --game-name wordle
```

```bash
python simulations.py --game-name dungleon
```

Alternatively, you can run the solver through a Jupyter notebook [`wordle_solver.ipynb`][colab-notebook]
[![Open In Colab][colab-badge]][colab-notebook]

---

## 📂 Project Structure
- `simulations.py`: Main script to run simulations and play the game.
- `src/`: Contains core source code logic (entropy, patterns, solvers, prior probabilities).
- `data/`: Contains word lists for Wordle and Dungleon.

---

## 📈 Results & References

Results from the models are shown [on the Wiki][wiki-results].

**References:**
- 3Blue1Brown, [*Solving Wordle using information theory*][youtube-video], posted on Youtube on February 6, 2022.
- [`3b1b/videos`][youtube-supplementary-code]: Supplementary code (in Python) accompanying the aforementioned video.
- [`woctezuma/dungleon-bot`][dungleon-bot]: Application of different solvers to [Dungleon][dungleon-rules].
- [`woctezuma/Wordle-Bot`][wordle-bot-python-fork]: Mentioning some initial results.

---

## 📊 Status
🚧 Project in progress

<!-- Definitions -->
[codacy]: <https://www.codacy.com/gh/woctezuma/3b1b-wordle-solver/dashboard>
[codacy-image]: <https://app.codacy.com/project/badge/Grade/ff156cc6b4604ba1a7527448480a118a>
[python-download-url]: <https://www.python.org/downloads/>
[colab-notebook]: <https://colab.research.google.com/github/woctezuma/3b1b-wordle-solver/blob/colab/wordle_solver.ipynb>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
[wiki-results]: <https://github.com/woctezuma/3b1b-wordle-solver/wiki>
[youtube-video]: <https://www.youtube.com/watch?v=v68zYyaEmEA>
[youtube-supplementary-code]: <https://github.com/3b1b/videos/tree/master/_2022/wordle>
[dungleon-bot]: <https://github.com/woctezuma/dungleon-bot>
[dungleon-rules]: <https://github.com/woctezuma/dungleon/wiki/Rules>
[wordle-bot-python-fork]: <https://github.com/woctezuma/Wordle-Bot>
