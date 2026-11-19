# AI Chess Move Predictor

A text-based chess game where you play against an AI that uses **Minimax** with **Alpha-Beta pruning** (adversarial search) to look several moves ahead.

## How it works

| Part | What it does |
|------|--------------|
| **Evaluator** (`evaluate`) | Scores a position: material (pawn=100, knight=320, bishop=330, rook=500, queen=900) + piece-square bonuses for pawns, knights and bishops. Checkmate = huge score, stalemate = 0. |
| **Minimax** (`search`) | White maximizes the score, Black minimizes it. The AI searches `depth` half-moves ahead and assumes the opponent also plays their best. |
| **Alpha-Beta pruning** | Skips branches that cannot change the final decision, giving the same answer with far fewer positions searched. |
| **Move ordering** | Captures and promotions are searched first (MVV-LVA), which makes pruning more effective. |

`python-chess` is used only for the rules of chess (legal moves, check, castling, en passant). The evaluation and search are written from scratch.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python chess_ai.py                 # play as White, AI looks 3 half-moves ahead
python chess_ai.py --depth 4       # stronger but slower
python chess_ai.py --color black   # play as Black
python chess_ai.py --benchmark     # compare plain Minimax vs Alpha-Beta
```

Enter moves like `e4`, `Nf3`, `O-O`, `exd5`, `e8=Q` or coordinates like `e2e4`.
Type `moves` to list legal moves, `quit` to exit.

## Possible improvements

- Iterative deepening and a transposition table
- Quiescence search (avoid evaluating mid-capture positions)
- Better evaluation: king safety, mobility, pawn structure
- Opening book
