"""
PS04 – Hex Game AI: Monte Carlo Tree Search (MCTS) with UCT
============================================================
BITS Pilani WILP – MTech AI/ML – S2 2025-2026
AIMLCZG557 / AECLZG557 – Assignment 2

Problem Statement:
    Develop an intelligent Hex-playing agent that selects the best legal
    move within a specified time limit using MCTS + UCT.

Board Rules:
    - N×N grid (7 ≤ N ≤ 11)
    - Player A (AI, symbol 1): connects top row (row 0) → bottom row (row N-1)
    - Player B (Human, symbol 2): connects left col (col 0) → right col (col N-1)
    - 6-adjacency on the square grid:
          Allowed  : top(r-1,c), bottom(r+1,c), left(r,c-1), right(r,c+1),
                     top-right(r-1,c+1), bottom-left(r+1,c-1)
          Forbidden: top-left(r-1,c-1), bottom-right(r+1,c+1)

Algorithm:
    Monte Carlo Tree Search (MCTS) with Upper Confidence Bound for Trees (UCT)
    Four phases per iteration: Selection → Expansion → Simulation → Backpropagation

Usage:
    python PS04_HexGame_MCTS.py [inputPS04.txt] [outputPS04.txt]
"""

import math
import time
import random
import sys
from collections import deque


# =================================================================
# ===== HexBoard ==================================================
# =================================================================

class HexBoard:
    """
    Represents an N×N Hex game board.

    Cell values  : 0 = empty, 1 = Player A (AI), 2 = Player B (Human)
    Coordinates  : (row, col), (0,0) is top-left

    6-adjacency (hexagonal topology on a square grid):
        Allowed  : (r-1,c), (r+1,c), (r,c-1), (r,c+1),
                   (r-1,c+1) [top-right], (r+1,c-1) [bottom-left]
        Forbidden: (r-1,c-1) [top-left], (r+1,c+1) [bottom-right]
    """

    _ADJ = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]

    def __init__(self, n: int):
        if not (7 <= n <= 11):
            print(f"[WARNING] Board size N={n} is outside expected range 7-11.")
        self.n = n
        self.board = [[0] * n for _ in range(n)]

    # ── Copy ──────────────────────────────────────────────────────────

    def copy(self) -> 'HexBoard':
        """Return a shallow-row deep copy without calling __init__."""
        obj = object.__new__(HexBoard)
        obj.n = self.n
        obj.board = [row[:] for row in self.board]
        return obj

    # ── Adjacency & Move helpers ──────────────────────────────────────

    def get_neighbours(self, r: int, c: int) -> list:
        """Return all in-bounds 6-neighbours of cell (r, c)."""
        n = self.n
        return [
            (r + dr, c + dc)
            for dr, dc in self._ADJ
            if 0 <= r + dr < n and 0 <= c + dc < n
        ]

    def get_empty_cells(self) -> list:
        """Return all empty (r, c) positions in row-major order."""
        return [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if self.board[r][c] == 0
        ]

    def is_valid_move(self, r: int, c: int) -> bool:
        """True iff (r, c) is in-bounds and unoccupied."""
        return 0 <= r < self.n and 0 <= c < self.n and self.board[r][c] == 0

    def apply_move(self, r: int, c: int, player: int):
        """
        Place player's piece at (r, c).
        Raises ValueError for out-of-bounds or occupied cells.
        """
        if not (0 <= r < self.n and 0 <= c < self.n):
            raise ValueError(
                f"Move ({r},{c}) out of bounds for {self.n}x{self.n} board."
            )
        if self.board[r][c] != 0:
            raise ValueError(
                f"Cell ({r},{c}) is already occupied by player {self.board[r][c]}."
            )
        self.board[r][c] = player

    # ── Win detection ─────────────────────────────────────────────────

    def check_winner(self) -> int:
        """
        BFS connectivity check.
        Returns 1 (Player A wins), 2 (Player B wins), or 0 (no winner yet).
        """
        src_a = [(0, c) for c in range(self.n) if self.board[0][c] == 1]
        if src_a and self._bfs(1, src_a, goal_r=self.n - 1):
            return 1
        src_b = [(r, 0) for r in range(self.n) if self.board[r][0] == 2]
        if src_b and self._bfs(2, src_b, goal_c=self.n - 1):
            return 2
        return 0

    def _bfs(self, player: int, starts: list,
             goal_r: int = None, goal_c: int = None) -> bool:
        """BFS; returns True when any goal row/col is reached."""
        visited = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            if (goal_r is not None and r == goal_r) or \
               (goal_c is not None and c == goal_c):
                return True
            for nr, nc in self.get_neighbours(r, c):
                if (nr, nc) not in visited and self.board[nr][nc] == player:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def get_winning_path(self, player: int) -> list:
        """
        BFS with parent-pointer tracking.
        Returns ordered [(r,c), ...] from start edge to goal edge,
        or [] if player has not yet won.
        """
        if player == 1:
            starts = [(0, c) for c in range(self.n) if self.board[0][c] == 1]
            is_goal = lambda r, _c: r == self.n - 1
        else:
            starts = [(r, 0) for r in range(self.n) if self.board[r][0] == 2]
            is_goal = lambda _r, c: c == self.n - 1

        parent = {s: None for s in starts}
        queue = deque(starts)
        goal_cell = None

        while queue and goal_cell is None:
            r, c = queue.popleft()
            if is_goal(r, c):
                goal_cell = (r, c)
                break
            for nr, nc in self.get_neighbours(r, c):
                if (nr, nc) not in parent and self.board[nr][nc] == player:
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

        if goal_cell is None:
            return []

        path, cur = [], goal_cell
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        return list(reversed(path))

    # ── Display ───────────────────────────────────────────────────────

    def __str__(self) -> str:
        """
        Render board with hex-style diagonal row offset.
        Symbols: '.' = empty, '1' = Player A, '2' = Player B
        """
        sym = {0: '.', 1: '1', 2: '2'}
        col_hdr = '    ' + '  '.join(f'{c}' for c in range(self.n))
        rows = []
        for r in range(self.n):
            indent = ' ' * r
            cells = ' '.join(sym[self.board[r][c]] for c in range(self.n))
            rows.append(f"{r:2}  {indent}{cells}")
        return '\n'.join([col_hdr] + rows)


# =================================================================
# ===== MCTSNode ==================================================
# =================================================================

class MCTSNode:
    """
    A node in the Monte Carlo Search Tree.

    Two-player UCT convention:
        self.player  : the player who is NEXT to move at this state
        self.wins    : simulations won by the player who moved TO this node
                       (= 3 - self.player)
        self.visits  : total rollouts recorded at this node

    This lets UCT at the parent always maximize wins/visits regardless
    of which player the parent represents.
    """

    __slots__ = ('board', 'player', 'parent', 'children', 'untried_moves',
                 'move', 'depth', 'visits', 'wins')

    def __init__(self, board: HexBoard, player: int,
                 parent: 'MCTSNode' = None,
                 move: tuple = None, depth: int = 0):
        self.board = board
        self.player = player          # next to move at this state
        self.parent = parent
        self.children: list = []
        self.move = move              # move that led here (None at root)
        self.depth = depth
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = board.get_empty_cells()
        random.shuffle(self.untried_moves)  # randomise expansion order

    # ── UCT ───────────────────────────────────────────────────────────

    def uct_value(self, c: float = math.sqrt(2)) -> float:
        """
        UCT formula (Kocsis & Szepesv??ri, 2006):
            Q(v') / N(v') + c * sqrt(ln N(v) / N(v'))
        Returns +inf for unvisited nodes to guarantee exploration.
        """
        if self.visits == 0:
            return float('inf')
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child_uct(self) -> 'MCTSNode':
        """Child with the highest UCT score (selection policy)."""
        return max(self.children, key=lambda ch: ch.uct_value())

    def most_visited_child(self) -> 'MCTSNode':
        """Child with the most visits (robust final move selection)."""
        return max(self.children, key=lambda ch: ch.visits)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    @property
    def is_terminal(self) -> bool:
        return self.board.check_winner() != 0


# =================================================================
# ===== MCTS Algorithm ============================================
# =================================================================

def _rollout(board: HexBoard, starting_player: int) -> int:
    """
    Simulation phase: play random moves until a winner is found.
    Returns the winning player (1 or 2).
    In Hex, a winner always exists (no draws), so this always returns 1 or 2.
    """
    sim = board.copy()
    cur = starting_player
    while True:
        w = sim.check_winner()
        if w:
            return w
        empty = sim.get_empty_cells()
        if not empty:
            return 0   # safety guard; unreachable in a legal Hex position
        r, c = random.choice(empty)
        sim.apply_move(r, c, cur)
        cur = 3 - cur


def _backpropagate(node: MCTSNode, result: int):
    """
    Backpropagation phase: walk from node to root updating statistics.
    node.wins tracks wins for the mover who reached this node (3 - node.player).
    """
    cur = node
    while cur is not None:
        cur.visits += 1
        if result == (3 - cur.player):   # mover to this node won
            cur.wins += 1.0
        cur = cur.parent


def run_mcts(board: HexBoard, current_player: int,
             time_limit_ms: int) -> tuple:
    """
    Run time-limited MCTS with UCT.

    Args:
        board           : current game board (not modified)
        current_player  : 1 (AI) or 2 (Human)
        time_limit_ms   : computation budget in milliseconds

    Returns:
        best_move       : (r, c) of the chosen move
        nodes_expanded  : total tree nodes created during search
        max_depth       : deepest tree level reached
        heuristic_score : estimated win probability % for current_player (0-100)
    """
    root = MCTSNode(board.copy(), current_player)
    deadline = time.time() + time_limit_ms / 1000.0
    nodes_expanded = 0
    max_depth = 0

    while time.time() < deadline:

        # ---- Phase 1: SELECTION -------------------------------------
        node = root
        while node.is_fully_expanded and node.children and not node.is_terminal:
            node = node.best_child_uct()

        # ---- Phase 2: EXPANSION -------------------------------------
        if node.untried_moves and not node.is_terminal:
            move = node.untried_moves.pop()
            child_board = node.board.copy()
            child_board.apply_move(move[0], move[1], node.player)
            child = MCTSNode(
                child_board,
                player=3 - node.player,
                parent=node,
                move=move,
                depth=node.depth + 1
            )
            node.children.append(child)
            node = child
            nodes_expanded += 1
            if node.depth > max_depth:
                max_depth = node.depth

        # ---- Phase 3: SIMULATION (ROLLOUT) --------------------------
        result = _rollout(node.board, node.player)

        # ---- Phase 4: BACKPROPAGATION --------------------------------
        _backpropagate(node, result)

    # Final move selection: most-visited child (robust against variance)
    if not root.children:
        empty = board.get_empty_cells()
        return (random.choice(empty) if empty else None), 0, 0, 0.0

    best = root.most_visited_child()
    # best.wins = wins for mover to 'best' = wins for current_player
    # (because best.player = 3 - current_player, so 3 - best.player = current_player)
    score = (best.wins / best.visits * 100.0) if best.visits else 0.0
    return best.move, nodes_expanded, max_depth, score


# =================================================================
# ===== Output Formatting =========================================
# =================================================================

def _banner(turn: int) -> str:
    """Fixed-width turn banner."""
    label = f"Turn {turn}"
    return f"{'='*20}{label}{'='*(25 - len(str(turn)))}"


def format_ai_turn(turn: int, move: tuple, depth: int, nodes: int,
                   score: float, elapsed_ms: float, status: str,
                   board: HexBoard, winner: int = 0,
                   path: list = None) -> str:
    """Format one complete AI turn block for outputPS04.txt."""
    score_str = '+INF' if winner == 1 else f"{score:.2f}"
    lines = [
        _banner(turn),
        "Player : A (AI)",
        f"Move Selected : ({move[0]},{move[1]})",
        "Search Algorithm : MCTS with UCT",
        f"Search Depth : {depth}",
        f"Nodes Expanded : {nodes:,}",
        f"Heuristic Score : {score_str}",
        f"Execution Time : {int(elapsed_ms)} ms",
        f"Game Status : {status}",
    ]
    if winner == 1 and path:
        path_str = ' -> '.join(f"({r},{c})" for r, c in path)
        lines += [
            "Terminal State : YES",
            "Winner : Player A",
            f"Winning Path {path_str}",
        ]
    lines += ["Current Board", str(board), ""]
    return '\n'.join(lines)


def format_human_turn(turn: int, move: tuple, elapsed_ms: float,
                      status: str, board: HexBoard,
                      winner: int = 0, path: list = None) -> str:
    """Format one complete Human turn block for outputPS04.txt."""
    lines = [
        _banner(turn),
        "Player : B (Human)",
        f"Move Entered : ({move[0]},{move[1]})",
        "Move Validation : VALID",
        f"Execution Time : {int(elapsed_ms)} ms",
        f"Game Status : {status}",
    ]
    if winner == 2 and path:
        path_str = ' -> '.join(f"({r},{c})" for r, c in path)
        lines += [
            "Terminal State : YES",
            "Winner : Player B",
            f"Winning Path {path_str}",
        ]
    lines += ["Current Board", str(board), ""]
    return '\n'.join(lines)


def format_summary(winner: int, total_turns: int, ai_turns: int,
                   human_turns: int, ai_depths: list,
                   ai_nodes: list, ai_times: list) -> str:
    """Format the GAME OVER summary block."""
    avg  = lambda lst: sum(lst) / len(lst) if lst else 0
    peak = lambda lst: max(lst) if lst else 0
    winner_name = "Player A" if winner == 1 else "Player B"
    result = "PLAYER_A_WINS" if winner == 1 else "PLAYER_B_WINS"
    sep = '=' * 60
    lines = [
        f"{sep} GAME OVER {sep}",
        f"Winner : {winner_name}",
        f"Total Turns : {total_turns}",
        f"Total AI Moves : {ai_turns}",
        f"Total Human Moves : {human_turns}",
        f"Average Search Depth : {avg(ai_depths):.1f}",
        f"Average Nodes Expanded : {int(avg(ai_nodes)):,}",
        f"Average AI Move Time : {int(avg(ai_times))} ms",
        f"Maximum Search Depth : {peak(ai_depths)}",
        f"Maximum Nodes Expanded : {int(peak(ai_nodes)):,}",
        f"Game Result : {result}",
    ]
    return '\n'.join(lines)


# =================================================================
# ===== Input Parser ==============================================
# =================================================================

def parse_input_file(filepath: str) -> tuple:
    """
    Parse inputPS04.txt.

    Format:
        Line 1    : N              (board size, integer 7-11)
        Line 2    : time_limit_ms  (milliseconds per AI move)
        Lines 3.. : N rows of N space-separated integers (0, 1, or 2)

    Returns (n: int, time_limit_ms: int, board: HexBoard).
    """
    try:
        with open(filepath, 'r') as fh:
            raw = [ln.strip() for ln in fh if ln.strip()]
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: '{filepath}'")
        sys.exit(1)

    if len(raw) < 2:
        print("[ERROR] Input file must have at least 2 lines (N and time_limit_ms).")
        sys.exit(1)

    try:
        n = int(raw[0])
        time_limit_ms = int(raw[1])
    except ValueError as exc:
        print(f"[ERROR] Could not parse N or time_limit_ms: {exc}")
        sys.exit(1)

    board = HexBoard(n)

    if len(raw) >= n + 2:
        for r in range(n):
            try:
                vals = list(map(int, raw[2 + r].split()))
            except ValueError as exc:
                print(f"[ERROR] Row {r}: {exc}")
                sys.exit(1)
            if len(vals) != n:
                print(f"[ERROR] Row {r} has {len(vals)} values; expected {n}.")
                sys.exit(1)
            for c, v in enumerate(vals):
                if v not in (0, 1, 2):
                    print(f"[ERROR] Invalid cell value {v} at ({r},{c}).")
                    sys.exit(1)
                board.board[r][c] = v

    return n, time_limit_ms, board


# =================================================================
# ===== Human Move Input ==========================================
# =================================================================

def get_human_move(board: HexBoard) -> tuple:
    """
    Interactively prompt Player B for a move.
    Validates bounds and occupancy; exits after 3 failed attempts.
    Returns (r, c).
    """
    for attempt in range(1, 4):
        try:
            raw = input("Player B, Enter your move (row,col): ").strip()
            parts = raw.replace(' ', '').split(',')
            if len(parts) != 2:
                raise ValueError("Expected two comma-separated integers.")
            r, c = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            print(f"  [INVALID] Bad format - use row,col (e.g. 3,4)  "
                  f"[attempt {attempt}/3]")
            continue

        if not (0 <= r < board.n and 0 <= c < board.n):
            print(f"  [INVALID] ({r},{c}) out of bounds "
                  f"(valid range 0-{board.n - 1})  [attempt {attempt}/3]")
            continue

        if board.board[r][c] != 0:
            occ = "AI (Player A)" if board.board[r][c] == 1 else "Human (Player B)"
            print(f"  [INVALID] Cell ({r},{c}) already occupied by {occ}.  "
                  f"[attempt {attempt}/3]")
            continue

        return r, c

    print("[ERROR] Too many invalid inputs (3 attempts). Exiting.")
    sys.exit(1)


# =================================================================
# ===== Game Loop (Interactive) ===================================
# =================================================================

def run_game(input_file: str = "inputPS04.txt",
             output_file: str = "outputPS04.txt"):
    """
    Main interactive game loop.
    Player A (AI, symbol=1) always moves first.
    Player B (Human, symbol=2) enters moves interactively.
    All turns are logged to output_file in the required format.
    """
    n, time_limit_ms, board = parse_input_file(input_file)

    bar = '-' * 56
    print(f"\n{bar}")
    print(f"  HEX GAME  |  {n}x{n} board  |  {time_limit_ms} ms/move")
    print(f"  Player A (AI)   : connects top row  -> bottom row")
    print(f"  Player B (Human): connects left col -> right col")
    print(bar)
    print("\nInitial Board:")
    print(board)
    print()

    output_log = []
    turn = 1
    current_player = 1
    ai_turns, human_turns = 0, 0
    ai_depths, ai_nodes_list, ai_times = [], [], []

    while True:
        winner = board.check_winner()
        if winner:
            break

        if current_player == 1:
            # ---- AI TURN -------------------------------------------
            print(f"  Turn {turn} | Player A (AI) thinking ...", flush=True)
            t0 = time.time()
            move, nodes, depth, score = run_mcts(board, 1, time_limit_ms)
            elapsed_ms = (time.time() - t0) * 1000.0

            if move is None:
                print("[INFO] AI has no valid moves.")
                break

            board.apply_move(move[0], move[1], 1)
            ai_turns += 1
            ai_depths.append(depth)
            ai_nodes_list.append(nodes)
            ai_times.append(elapsed_ms)

            winner = board.check_winner()
            status = "TERMINAL" if winner else "CONTINUE"
            path = board.get_winning_path(1) if winner == 1 else []

            block = format_ai_turn(turn, move, depth, nodes, score,
                                   elapsed_ms, status, board, winner, path)
            output_log.append(block)

            print(f"    -> ({move[0]},{move[1]})  "
                  f"depth={depth}  nodes={nodes:,}  "
                  f"score={score:.1f}%  time={int(elapsed_ms)}ms")
            print(board)
            print()

        else:
            # ---- HUMAN TURN ----------------------------------------
            print(f"  Turn {turn} | Player B (Human)")
            print(board)
            t0 = time.time()
            r, c = get_human_move(board)
            elapsed_ms = (time.time() - t0) * 1000.0

            board.apply_move(r, c, 2)
            human_turns += 1

            winner = board.check_winner()
            status = "TERMINAL" if winner else "CONTINUE"
            path = board.get_winning_path(2) if winner == 2 else []

            block = format_human_turn(turn, (r, c), elapsed_ms,
                                      status, board, winner, path)
            output_log.append(block)

            print(f"    Entered: ({r},{c})")
            print(board)
            print()

        if winner:
            break

        current_player = 3 - current_player
        turn += 1

    # ---- Game Over -------------------------------------------------
    winner = board.check_winner()
    summary = format_summary(winner, turn, ai_turns, human_turns,
                             ai_depths, ai_nodes_list, ai_times)
    output_log.append(summary)

    winner_name = "Player A (AI)" if winner == 1 else "Player B (Human)"
    print(f"\n{'='*56}")
    print(f"  GAME OVER | Winner: {winner_name} | Turns: {turn}")
    print(f"{'='*56}")
    print(summary)

    with open(output_file, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(output_log))
    print(f"\n[OK] Output written to '{output_file}'")


# =================================================================
# ===== Automated Test Runner (for notebook / unit tests) =========
# =================================================================

def run_game_test(n: int, time_limit_ms: int,
                  initial_board: list = None,
                  human_moves: list = None,
                  output_file: str = "outputPS04_test.txt") -> dict:
    """
    Automated game runner with predefined human moves.
    Suitable for Jupyter notebook testing (no interactive input).

    Args:
        n              : board size (7-11)
        time_limit_ms  : AI time budget per move (ms)
        initial_board  : optional N x N list-of-lists (values 0/1/2);
                         None means start from an empty board
        human_moves    : list of (r, c) for Player B; if exhausted or
                         invalid, falls back to the first available cell
        output_file    : path for output log

    Returns:
        dict with keys: winner, total_turns, ai_turns, human_turns,
                        avg_depth, avg_nodes, avg_time_ms
    """
    board = HexBoard(n)
    if initial_board:
        for r in range(n):
            for c in range(n):
                board.board[r][c] = initial_board[r][c]

    human_iter = iter(human_moves or [])
    output_log = []
    turn = 1
    current_player = 1
    ai_turns, human_turns = 0, 0
    ai_depths, ai_nodes_list, ai_times = [], [], []

    print(f"\n{'='*56}")
    print(f"  TEST  |  {n}x{n}  |  {time_limit_ms} ms/move")
    print(f"{'='*56}")
    print(board)

    while True:
        winner = board.check_winner()
        if winner:
            break

        if current_player == 1:
            t0 = time.time()
            move, nodes, depth, score = run_mcts(board, 1, time_limit_ms)
            elapsed_ms = (time.time() - t0) * 1000.0

            if move is None:
                break

            board.apply_move(move[0], move[1], 1)
            ai_turns += 1
            ai_depths.append(depth)
            ai_nodes_list.append(nodes)
            ai_times.append(elapsed_ms)

            winner = board.check_winner()
            status = "TERMINAL" if winner else "CONTINUE"
            path = board.get_winning_path(1) if winner == 1 else []

            output_log.append(
                format_ai_turn(turn, move, depth, nodes, score,
                               elapsed_ms, status, board, winner, path)
            )
            print(f"  T{turn:02d} AI   -> ({move[0]},{move[1]})  "
                  f"nodes={nodes:,}  score={score:.1f}%  {int(elapsed_ms)}ms")

        else:
            hm = next(human_iter, None)
            if hm is None or not board.is_valid_move(hm[0], hm[1]):
                empty = board.get_empty_cells()
                hm = empty[0] if empty else None
            if hm is None:
                break

            t0 = time.time()
            board.apply_move(hm[0], hm[1], 2)
            elapsed_ms = (time.time() - t0) * 1000.0
            human_turns += 1

            winner = board.check_winner()
            status = "TERMINAL" if winner else "CONTINUE"
            path = board.get_winning_path(2) if winner == 2 else []

            output_log.append(
                format_human_turn(turn, hm, elapsed_ms,
                                  status, board, winner, path)
            )
            print(f"  T{turn:02d} HUM  -> ({hm[0]},{hm[1]})")

        if winner:
            break
        current_player = 3 - current_player
        turn += 1

    winner = board.check_winner()
    summary = format_summary(winner, turn, ai_turns, human_turns,
                             ai_depths, ai_nodes_list, ai_times)
    output_log.append(summary)

    print(f"\n{board}")
    print(f"\n{summary}")

    with open(output_file, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(output_log))
    print(f"\n[OK] Output written to '{output_file}'")

    avg  = lambda lst: sum(lst) / len(lst) if lst else 0
    return {
        'winner': winner,
        'total_turns': turn,
        'ai_turns': ai_turns,
        'human_turns': human_turns,
        'avg_depth': avg(ai_depths),
        'avg_nodes': avg(ai_nodes_list),
        'avg_time_ms': avg(ai_times),
    }


# =================================================================
# ===== Entry Point ===============================================
# =================================================================

if __name__ == "__main__":
    in_path  = sys.argv[1] if len(sys.argv) >= 2 else "inputPS04.txt"
    out_path = sys.argv[2] if len(sys.argv) >= 3 else "outputPS04.txt"
    run_game(in_path, out_path)
