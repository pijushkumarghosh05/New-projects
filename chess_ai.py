"""
AI Chess Move Predictor
=======================
A text-based chess game where a human plays against an AI.

The AI uses:
  * a board state evaluator (material + piece-square bonuses)
  * the Minimax algorithm (adversarial search)
  * Alpha-Beta pruning to search several moves ahead faster

The `python-chess` library is used ONLY for the rules of chess
(legal move generation, check, checkmate, castling, en passant...).
The evaluation and search logic are written from scratch below.

Run:
    python chess_ai.py                 # play as White, AI depth 3
    python chess_ai.py --depth 4       # stronger (slower) AI
    python chess_ai.py --color black   # play as Black
    python chess_ai.py --benchmark     # compare Minimax vs Alpha-Beta
"""

import argparse
import random
import time

import chess

# ---------------------------------------------------------------------------
# 1. BOARD STATE EVALUATOR
# ---------------------------------------------------------------------------
# Scores are from WHITE's point of view:
#   positive  -> good for White
#   negative  -> good for Black

INF = float("inf")
MATE_SCORE = 100_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # the king is never captured, so no material value
}

# Piece-square tables: small bonuses for placing pieces on good squares.
# Written from White's view, rank 1 first (index 0 = a1, index 63 = h8).
PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

POSITION_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
}


def evaluate(board, depth_left=0):
    """Return a score for the position from White's point of view."""
    # Terminal positions
    if board.is_checkmate():
        # The side to move is checkmated. Adding depth_left makes the AI
        # prefer faster mates (and delay being mated).
        if board.turn == chess.WHITE:
            return -(MATE_SCORE + depth_left)
        return MATE_SCORE + depth_left
    if board.is_game_over():  # stalemate, insufficient material, etc.
        return 0

    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]

        table = POSITION_TABLES.get(piece.piece_type)
        if table is not None:
            # Mirror the square for Black so the same table works for both.
            index = square if piece.color == chess.WHITE else chess.square_mirror(square)
            value += table[index]

        score += value if piece.color == chess.WHITE else -value
    return score


# ---------------------------------------------------------------------------
# 2. MINIMAX + ALPHA-BETA PRUNING
# ---------------------------------------------------------------------------

def move_order_key(board, move):
    """Heuristic: look at promising moves first (captures, promotions).

    Alpha-Beta prunes far more when good moves are searched early.
    Uses MVV-LVA: Most Valuable Victim, Least Valuable Attacker.
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square) or chess.PAWN  # en passant
        attacker = board.piece_type_at(move.from_square)
        score += 10 * PIECE_VALUES[victim] - PIECE_VALUES[attacker]
    if move.promotion:
        score += PIECE_VALUES[move.promotion]
    return score


def search(board, depth, alpha, beta, use_pruning, stats):
    """Minimax search, optionally with Alpha-Beta pruning.

    White is the MAXIMIZING player, Black is the MINIMIZING player.
    alpha = best score White can already guarantee
    beta  = best score Black can already guarantee
    If alpha >= beta, the rest of this branch cannot affect the final
    decision, so we stop exploring it (this is the "pruning").
    """
    stats["nodes"] += 1

    if depth == 0 or board.is_game_over():
        return evaluate(board, depth)

    moves = sorted(board.legal_moves, key=lambda m: move_order_key(board, m), reverse=True)

    if board.turn == chess.WHITE:  # maximizing
        best = -INF
        for move in moves:
            board.push(move)
            best = max(best, search(board, depth - 1, alpha, beta, use_pruning, stats))
            board.pop()
            alpha = max(alpha, best)
            if use_pruning and alpha >= beta:
                break  # beta cut-off
        return best
    else:  # minimizing
        best = INF
        for move in moves:
            board.push(move)
            best = min(best, search(board, depth - 1, alpha, beta, use_pruning, stats))
            board.pop()
            beta = min(beta, best)
            if use_pruning and alpha >= beta:
                break  # alpha cut-off
        return best


def find_best_move(board, depth, use_pruning=True):
    """Pick the best move for the side to move.

    Returns (best_move, score, stats) where stats["nodes"] is the number
    of positions examined.
    """
    stats = {"nodes": 0}
    maximizing = board.turn == chess.WHITE
    best_move = None
    best_score = -INF if maximizing else INF
    alpha, beta = -INF, INF

    moves = list(board.legal_moves)
    random.shuffle(moves)  # variety between equally good moves
    moves.sort(key=lambda m: move_order_key(board, m), reverse=True)  # stable sort

    for move in moves:
        board.push(move)
        score = search(board, depth - 1, alpha, beta, use_pruning, stats)
        board.pop()

        if maximizing and score > best_score:
            best_score, best_move = score, move
            alpha = max(alpha, score)
        elif not maximizing and score < best_score:
            best_score, best_move = score, move
            beta = min(beta, score)

    return best_move, best_score, stats


# ---------------------------------------------------------------------------
# 3. TEXT-BASED GAME
# ---------------------------------------------------------------------------

def print_board(board, perspective=chess.WHITE):
    """Print the board in plain text, from the human player's side."""
    ranks = range(7, -1, -1) if perspective == chess.WHITE else range(8)
    files = range(8) if perspective == chess.WHITE else range(7, -1, -1)

    print()
    for r in ranks:
        row = []
        for f in files:
            piece = board.piece_at(chess.square(f, r))
            row.append(piece.symbol() if piece else ".")
        print(f" {r + 1}  " + " ".join(row))
    print("    " + " ".join("abcdefgh"[f] for f in files))
    print()


def get_human_move(board):
    """Ask the human for a move. Returns a Move, or None if they quit."""
    while True:
        text = input("Your move (e.g. e4, Nf3, e2e4) > ").strip()

        if text.lower() in ("quit", "exit", "q"):
            return None
        if text.lower() in ("help", "?"):
            print("Enter a move in standard notation (e4, Nf3, O-O, exd5, e8=Q)")
            print("or coordinates (e2e4). Commands: moves, quit")
            continue
        if text.lower() == "moves":
            print(", ".join(sorted(board.san(m) for m in board.legal_moves)))
            continue

        # Try standard algebraic notation first (case-sensitive: Nf3, O-O)
        try:
            return board.parse_san(text)
        except ValueError:
            pass

        # Then try coordinate notation (e2e4, e7e8q)
        try:
            move = chess.Move.from_uci(text.lower())
            if move in board.legal_moves:
                return move
        except ValueError:
            pass

        print("Invalid or illegal move. Type 'moves' to see all legal moves.")


def announce_result(board):
    print_board(board)
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Checkmate! {winner} wins.")
    elif board.is_stalemate():
        print("Stalemate - draw.")
    else:
        print(f"Game over - draw ({board.result()}).")


def play_game(depth, human_color):
    board = chess.Board()
    print("=" * 40)
    print(" AI CHESS  (Minimax + Alpha-Beta)")
    print("=" * 40)
    print(f"You are {'White' if human_color == chess.WHITE else 'Black'}. AI depth: {depth}")
    print("Type 'help' for input formats, 'quit' to exit.")

    while not board.is_game_over():
        print_board(board, human_color)
        if board.is_check():
            print("Check!")

        if board.turn == human_color:
            move = get_human_move(board)
            if move is None:
                print("You resigned. Goodbye!")
                return
            board.push(move)
        else:
            print("AI is thinking...")
            start = time.time()
            move, score, stats = find_best_move(board, depth)
            elapsed = time.time() - start
            san = board.san(move)
            board.push(move)
            print(f"AI plays: {san}")
            print(f"  (eval {score / 100:+.2f} pawns for White, "
                  f"{stats['nodes']:,} positions searched in {elapsed:.2f}s)")

    announce_result(board)


# ---------------------------------------------------------------------------
# 4. BENCHMARK: show how much Alpha-Beta saves
# ---------------------------------------------------------------------------

def benchmark(depth):
    board = chess.Board()
    print(f"Benchmark from the starting position, depth {depth}\n")
    results = {}
    for label, pruning in (("Minimax (no pruning)", False), ("Minimax + Alpha-Beta", True)):
        start = time.time()
        move, score, stats = find_best_move(board, depth, use_pruning=pruning)
        elapsed = time.time() - start
        results[label] = stats["nodes"]
        print(f"{label:<22} nodes: {stats['nodes']:>9,}   time: {elapsed:6.2f}s   "
              f"best score: {score}")
    saved = 1 - results["Minimax + Alpha-Beta"] / results["Minimax (no pruning)"]
    print(f"\nAlpha-Beta examined {saved:.0%} fewer positions with the same best score.")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Play chess against a Minimax AI.")
    parser.add_argument("--depth", type=int, default=3,
                        help="how many half-moves (plies) the AI looks ahead (default 3)")
    parser.add_argument("--color", choices=["white", "black", "random"], default="white",
                        help="the colour you play (default white)")
    parser.add_argument("--benchmark", action="store_true",
                        help="compare plain Minimax with Alpha-Beta and exit")
    args = parser.parse_args()

    if args.benchmark:
        benchmark(args.depth)
        return

    if args.color == "random":
        human_color = random.choice([chess.WHITE, chess.BLACK])
    else:
        human_color = chess.WHITE if args.color == "white" else chess.BLACK

    play_game(args.depth, human_color)


if __name__ == "__main__":
    main()
