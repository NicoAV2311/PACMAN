# pacman_py.py
# Prototipo Pac-Man en Python con Pygame
# Incluye: controles, audio, score, lives, level loader (PNG or generator),
# save/load high scores (JSON), speed adjustment, seedable RNG,
# y 5 modelos de IA de fantasmas.

import pygame, sys, os, json, math, random, time, traceback
from collections import deque, namedtuple
from pathlib import Path

# -----------------------
# Config & Paths
# -----------------------
WIDTH, HEIGHT = 896, 720  # ventana
TILE = 24                 # tamaño de celda
MAP_COLS = 28
MAP_ROWS = 31
FPS = 60

def resource_path(rel_path: str) -> Path:
    """Return an absolute path to a resource, working for dev and for PyInstaller.

    Usage: resource_path('pacman_data') or resource_path('assets/player.png')
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # use getattr to avoid static analyzers complaining about the attribute
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            base_path = Path(meipass)
        else:
            raise Exception("not frozen")
    except Exception:
        base_path = Path(__file__).parent
    return base_path / rel_path


DATA_DIR = resource_path("pacman_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
HIGHSCORE_FILE = DATA_DIR / "highscores.json"

# -----------------------
# Utility: reproducible LCG generator
# -----------------------
class LCG:
    def __init__(self, seed=12345):
        self.m = 2**32
        self.a = 1664525
        self.c = 1013904223
        self.state = seed & 0xFFFFFFFF
    def randint(self, a, b):
        self.state = (self.a*self.state + self.c) % self.m
        return a + (self.state % (b - a + 1))
    def rand(self):
        return self.randint(0, 2**31-1) / (2**31-1)


def generate_sample_map(cols=28, rows=31):
    # simple rectangular maze-ish map for demo
    grid = [[0 for _ in range(cols)] for __ in range(rows)]
    # border walls
    for x in range(cols):
        grid[0][x] = 1
        grid[rows-1][x] = 1
    for y in range(rows):
        grid[y][0] = 1
        grid[y][cols-1] = 1
    # some inner walls (pattern)
    for x in range(2, cols-2, 2):
        for y in range(2, rows-2, 4):
            grid[y][x] = 1
    # place power pellets in corners
    grid[1][1] = 2
    grid[1][cols-2] = 2
    grid[rows-2][1] = 2
    grid[rows-2][cols-2] = 2
    # spawn player and ghosts
    grid[rows//2 + 4][cols//2] = 3
    grid[rows//2][cols//2 - 2] = 4
    grid[rows//2][cols//2 + 2] = 4
    grid[rows//2 - 2][cols//2] = 4
    # add more structured corridors to better match classic Pac-Man feel
    midr = rows//2
    # horizontal bands near top and bottom
    for c in range(2, cols-2):
        grid[3][c] = 1
        grid[rows-4][c] = 1
    # central box
    box_r0 = midr - 3
    box_r1 = midr + 3
    box_c0 = cols//2 - 7
    box_c1 = cols//2 + 7
    for r in range(box_r0, box_r1+1):
        for c in range(box_c0, box_c1+1):
            if r in (box_r0, box_r1) or c in (box_c0, box_c1):
                grid[r][c] = 1
    # open multiple gates in the box so ghosts can leave
    # central top gate
    gate_r, gate_c = box_r0, cols//2
    grid[gate_r][gate_c] = 0
    # two additional top-side gates (left/right of center)
    left_top_gate_c = max(box_c0+1, cols//2 - 3)
    right_top_gate_c = min(box_c1-1, cols//2 + 3)
    grid[box_r0][left_top_gate_c] = 0
    grid[box_r0][right_top_gate_c] = 0
    # bottom center gate
    grid[box_r1][cols//2] = 0
    # create small islands/wall clusters to emulate classic maze
    for r in range(6, rows-6, 4):
        for c in range(3, cols-3, 6):
            grid[r][c] = 1
    # create tunnel openings at center row (left/right)
    grid[midr][0] = 0
    grid[midr][cols-1] = 0

    return grid


def mirror_horizontal(grid):
    rows = len(grid); cols = len(grid[0])
    new = [[grid[r][cols-1-c] for c in range(cols)] for r in range(rows)]
    return new


def rotate_180(grid):
    rows = len(grid); cols = len(grid[0])
    new = [[grid[rows-1-r][cols-1-c] for c in range(cols)] for r in range(rows)]
    return new


def jitter_inner_walls(grid, rng, density=0.05):
    # add/remove some inner walls (preserve border walls)
    rows = len(grid); cols = len(grid[0])
    new = [row[:] for row in grid]
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if (r, c) in ((1,1),(1,cols-2),(rows-2,1),(rows-2,cols-2)):
                continue
            if new[r][c] == 1:
                # remove some walls
                if rng.rand() < density/2:
                    new[r][c] = 0
            elif new[r][c] == 0:
                # add some walls
                if rng.rand() < density/2:
                    new[r][c] = 1
    return new


def shift_power_pellets(grid, rng):
    # move power pellets slightly
    rows = len(grid); cols = len(grid[0])
    new = [row[:] for row in grid]
    # collect power pellet positions
    ppos = []
    for r in range(rows):
        for c in range(cols):
            if new[r][c] == 2:
                new[r][c] = 0
                ppos.append((r,c))
    for (r,c) in ppos:
        dr = rng.randint(-2,2)
        dc = rng.randint(-2,2)
        nr, nc = max(1, min(rows-2, r+dr)), max(1, min(cols-2, c+dc))
        if new[nr][nc] == 0:
            new[nr][nc] = 2
    return new


def generate_level_map(cols=28, rows=31, level=1, seed=0):
    """Generate a level map derived from the base sample map.
    Variants:
      - level 1: base map
      - level 2: horizontal mirror
      - level 3: rotated 180
      - level 4..n: seeded jitter + shifted pellets
    """
    # base map for generation
    base = generate_sample_map(cols, rows)
    rng = LCG(seed or int(time.time()) & 0xFFFFFFFF)
    # For level 1 try to derive a 'classic-like' variant safely and validate it
    if level == 1:
        def make_classic_variant():
            g = generate_sample_map(cols, rows)
            midr = rows//2
            box_r0 = midr - 3
            box_r1 = midr + 3
            box_c0 = cols//2 - 7
            box_c1 = cols//2 + 7
            for r in range(box_r0, box_r1+1):
                for c in range(box_c0, box_c1+1):
                    if r in (box_r0, box_r1) or c in (box_c0, box_c1):
                        g[r][c] = 1
            # open gates carefully
            if 0 <= box_r0 < rows and 0 <= cols//2 < cols:
                g[box_r0][cols//2] = 0
            left_top_gate_c = max(box_c0+1, cols//2 - 3)
            right_top_gate_c = min(box_c1-1, cols//2 + 3)
            if 0 <= box_r0 < rows:
                if 0 <= left_top_gate_c < cols: g[box_r0][left_top_gate_c] = 0
                if 0 <= right_top_gate_c < cols: g[box_r0][right_top_gate_c] = 0
            if 0 <= box_r1 < rows and 0 <= cols//2 < cols:
                g[box_r1][cols//2] = 0
            # ensure tunnel openings
            if 0 <= midr < rows:
                g[midr][0] = 0
                g[midr][cols-1] = 0
            # set corner power pellets
            if rows > 2 and cols > 2:
                g[1][1] = 2
                g[1][cols-2] = 2
                g[rows-2][1] = 2
                g[rows-2][cols-2] = 2
            # ensure spawns exist
            pr, pc = rows//2 + 4, cols//2
            if not any(3 in row for row in g):
                g[pr][pc] = 3
            grs = [(rows//2, cols//2 - 2), (rows//2, cols//2 + 2), (rows//2 - 2, cols//2)]
            for gr, gc in grs:
                if 0 <= gr < rows and 0 <= gc < cols and g[gr][gc] == 0:
                    g[gr][gc] = 4
            return g

        candidate = make_classic_variant()
        if is_map_playable(candidate):
            base = candidate
        else:
            print("[map] classic-like variant failed validation; falling back to sample map")
    if level == 1:
        grid = base
    elif level == 2:
        grid = mirror_horizontal(base)
    elif level == 3:
        grid = rotate_180(base)
    else:
        # start from base and apply jitter proportional to level
        density = min(0.25, 0.02 + (level-4) * 0.01)
        grid = jitter_inner_walls(base, rng, density=density)
        grid = shift_power_pellets(grid, rng)

    # ensure player and ghost spawns exist in predictable locations
    # clear any spawn markers and set our spawns
    # Player spawn: lower center-ish
    pr, pc = rows//2 + 4, cols//2
    grs = [(rows//2, cols//2 - 2), (rows//2, cols//2 + 2), (rows//2 - 2, cols//2)]
    # make sure spawn tiles are empty
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in (3,4):
                grid[r][c] = 0
    # place player spawn
    grid[pr][pc] = 3
    # place up to 4 ghost spawns (if space occupied try nearby)
    for i, (gr, gc) in enumerate(grs):
        if grid[gr][gc] == 0:
            grid[gr][gc] = 4
        else:
            # search nearby empty cell
            placed = False
            for dr in range(-2,3):
                for dc in range(-2,3):
                    nr, nc = gr+dr, gc+dc
                    if 1 <= nr < rows-1 and 1 <= nc < cols-1 and grid[nr][nc] == 0:
                        grid[nr][nc] = 4
                        placed = True
                        break
                if placed: break

    return grid




# -----------------------
# SaveManager: highscores JSON
# -----------------------
def load_highscores():
    if not HIGHSCORE_FILE.exists():
        return []
    try:
        with open(HIGHSCORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # ensure sorting by score descending
            try:
                data = sorted(data, key=lambda x: x.get("score", 0), reverse=True)
            except Exception:
                pass
            return data
    except Exception:
        return []

def save_highscores(list_scores):
    # sort and keep top 10
    try:
        out = sorted(list_scores, key=lambda x: x.get("score", 0), reverse=True)[:10]
    except Exception:
        out = list_scores[:10]
    with open(HIGHSCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

# -----------------------
# Pathfinding helpers (BFS for grid)
# -----------------------
Point = namedtuple("Point", ["r","c"])

def neighbors_of(pt, grid):
    rows = len(grid); cols = len(grid[0])
    # allow wrap-around through tunnels at center row(s)
    tunnel_rows = {rows//2}
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = pt.r+dr, pt.c+dc
        # normal neighbor
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] != 1:
                yield Point(nr,nc)
        else:
            # handle horizontal wrap (tunnel)
            if dc == -1 and nc < 0 and pt.r in tunnel_rows:
                # wrap to right edge
                if grid[pt.r][cols-1] != 1:
                    yield Point(pt.r, cols-1)
            elif dc == 1 and nc >= cols and pt.r in tunnel_rows:
                # wrap to left edge
                if grid[pt.r][0] != 1:
                    yield Point(pt.r, 0)

def bfs_shortest(src, dst, grid):
    # returns next step (Point) from src towards dst, or None
    if src == dst: return src
    q = deque([src])
    prev = {src: None}
    while q:
        u = q.popleft()
        if u == dst:
            # backtrack
            cur = u
            while prev[cur] != src:
                cur = prev[cur]
            return cur
        for v in neighbors_of(u, grid):
            if v not in prev:
                prev[v] = u
                q.append(v)
    return None


def reachable_from(start, grid):
    """Return set of Points reachable from start using neighbors_of."""
    rows = len(grid); cols = len(grid[0])
    if not (0 <= start.r < rows and 0 <= start.c < cols):
        return set()
    q = deque([Point(start.r, start.c)])
    seen = {Point(start.r, start.c)}
    while q:
        u = q.popleft()
        for v in neighbors_of(u, grid):
            if v not in seen:
                seen.add(v); q.append(v)
    return seen


def is_map_playable(grid):
    """Validate map: player spawn must exist and all pellets must be reachable from player."""
    rows = len(grid); cols = len(grid[0])
    # find player spawn
    player_pos = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                player_pos = Point(r, c); break
        if player_pos: break
    if not player_pos:
        return False
    reachable = reachable_from(player_pos, grid)
    # check all pellet tiles (0 and 2) are reachable
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in (0, 2):
                if Point(r, c) not in reachable:
                    return False
    return True

# -----------------------
# Ghost behaviors
# -----------------------
class GhostBehavior:
    RANDOM = "random"
    BIASED = "biased"
    FSM = "fsm"
    MARKOV = "markov"
    SMOOTH = "smooth"
    BLINKY = "blinky"
    PINKY = "pinky"
    INKY = "inky"
    CLYDE = "clyde"

# -----------------------
# Game entities
# -----------------------
class Player:
    def __init__(self, r: int, c: int):
        self.r = r; self.c = c
        self.dir: tuple[int,int] = (0,0)
        self.next_dir: tuple[int,int] = (0,0)
        # tiles per second (movement will be applied in discrete tile steps
        # based on accumulators so movement is framerate-independent)
        # match ghosts' base speed to avoid Pac-Man being faster than ghosts
        self.speed = 4.0
        self._move_acc = 0.0
        self.alive = True

    def update(self, dt, grid):
        # accumulate movement (tiles). When accumulator >= 1.0 we move one tile.
        # This makes movement proportional to dt and not tied to FPS.
        self._move_acc += self.speed * dt
        # number of whole-tile moves to attempt this update
        moves = int(self._move_acc)
        if moves <= 0:
            return
        for _ in range(moves):
            # attempt to change direction if possible
            if self.next_dir != (0,0):
                nr, nc = self.r + self.next_dir[0], self.c + self.next_dir[1]
                rows = len(grid); cols = len(grid[0])
                # handle horizontal tunnel wrap
                if nc < 0:
                    # wrap to right edge if this row has a tunnel opening
                    if self.r == rows//2 and grid[self.r][cols-1] != 1:
                        nr, nc = self.r, cols-1
                elif nc >= cols:
                    if self.r == rows//2 and grid[self.r][0] != 1:
                        nr, nc = self.r, 0
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
                    self.dir = self.next_dir
            # move along dir if possible
            nr, nc = self.r + self.dir[0], self.c + self.dir[1]
            rows = len(grid); cols = len(grid[0])
            # handle wrap on actual move
            if nc < 0:
                if self.r == rows//2 and grid[self.r][cols-1] != 1:
                    nr, nc = self.r, cols-1
            elif nc >= cols:
                if self.r == rows//2 and grid[self.r][0] != 1:
                    nr, nc = self.r, 0
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
                self.r, self.c = nr, nc
            else:
                # hit wall: stop moving
                self.dir = (0,0)
                break
        # remove the moves we applied from the accumulator
        self._move_acc -= moves

class Ghost:
    def __init__(self, r: int, c: int, behavior=GhostBehavior.RANDOM, seed=None, scatter_target=None, color=(255,0,0)):
        self.r = r; self.c = c
        self.behavior = behavior
        self.state = "scatter"  # scatter / chase / frightened / eaten
        self.state_timer = 20.0
        # tiles per second
        self.speed = 4.0
        self._move_acc = 0.0
        self.scatter_target = scatter_target or Point(1,1)
        self.rng = LCG(seed or random.randint(1,2**31-1))
        # For markov
        self.last_move = None
        # for smooth noise
        self.phase = self.rng.rand()*10.0
        self.color = color

    def update(self, dt, grid, player, ghosts=None):
        # state timer handling (simple schedule)
        self.state_timer -= dt
        if self.state_timer <= 0:
            if self.state == "scatter":
                self.state = "chase"; self.state_timer = 7.0
            elif self.state == "chase":
                self.state = "scatter"; self.state_timer = 20.0
            elif self.state == "frightened":
                self.state = "chase"; self.state_timer = 10.0
        # accumulate movement (tiles). Only attempt to pick and move when
        # we have at least one whole tile worth of movement.
        self._move_acc += self.speed * dt
        moves = int(self._move_acc)
        if moves <= 0:
            return
        for _ in range(moves):
            # choose next tile based on behavior and state
            current = Point(self.r, self.c)
            options = list(neighbors_of(current, grid))
            if not options:
                break
            chosen = options[0]
            if self.state == "frightened":
                # choose random among options
                chosen = options[self.rng.randint(0, len(options)-1)]
            else:
                if self.behavior == GhostBehavior.RANDOM:
                    chosen = options[self.rng.randint(0, len(options)-1)]
                elif self.behavior == GhostBehavior.BIASED:
                    # bias towards player: pick option minimizing distance to player
                    best = None; bestd = None
                    for o in options:
                        d = (o.r - player.r)**2 + (o.c - player.c)**2
                        # add small jitter
                        d -= self.rng.rand()*2.0
                        if best is None or d < bestd:
                            best, bestd = o, d
                    chosen = best
                elif self.behavior == GhostBehavior.FSM:
                    # chase uses BFS one-step; scatter heads to scatter_target
                    if self.state == "chase":
                        target = Point(player.r, player.c)
                    else:
                        target = self.scatter_target
                    next_step = bfs_shortest(current, target, grid)
                    if next_step: chosen = next_step
                elif self.behavior == GhostBehavior.MARKOV:
                    # simple: avoid reversing if possible; use last_move to bias
                    weights = []
                    for o in options:
                        w = 1.0
                        if self.last_move is not None:
                            # discourage direct reverse
                            if (o.r == self.r - self.last_move.r) and (o.c == self.c - self.last_move.c):
                                w = 0.2
                        weights.append(w)
                    s = sum(weights)
                    rnum = self.rng.rand()*s
                    acc = 0.0
                    for i,w in enumerate(weights):
                        acc += w
                        if rnum <= acc:
                            chosen = options[i]; break
                elif self.behavior == GhostBehavior.SMOOTH:
                    # smooth using sine to index options
                    idx = int(abs(math.sin(time.time() + self.phase)) * 1000) % len(options)
                    chosen = options[idx]
                elif self.behavior == GhostBehavior.BLINKY:
                    # direct chase: target player current tile
                    target = Point(player.r, player.c)
                    ns = bfs_shortest(current, target, grid)
                    if ns: chosen = ns
                elif self.behavior == GhostBehavior.PINKY:
                    # ambusher: target 4 tiles ahead of player dir
                    try:
                        dr, dc = player.dir
                        target = Point(player.r + 4*dr, player.c + 4*dc)
                    except Exception:
                        target = Point(player.r, player.c)
                    ns = bfs_shortest(current, target, grid)
                    if ns: chosen = ns
                elif self.behavior == GhostBehavior.INKY:
                    # vector chase based on Blinky position
                    blinky = None
                    if ghosts:
                        for gg in ghosts:
                            if gg is not self and gg.behavior == GhostBehavior.BLINKY:
                                blinky = gg; break
                    try:
                        dr, dc = player.dir
                        base = Point(player.r + 2*dr, player.c + 2*dc)
                    except Exception:
                        base = Point(player.r, player.c)
                    if blinky:
                        vr = base.r - blinky.r
                        vc = base.c - blinky.c
                        target = Point(base.r + vr, base.c + vc)
                    else:
                        target = base
                    ns = bfs_shortest(current, target, grid)
                    if ns: chosen = ns
                elif self.behavior == GhostBehavior.CLYDE:
                    # if far from player chase, else scatter to corner
                    d = (current.r - player.r)**2 + (current.c - player.c)**2
                    if d > 64:
                        target = Point(player.r, player.c)
                        ns = bfs_shortest(current, target, grid)
                        if ns: chosen = ns
                    else:
                        # scatter - head to bottom-left corner
                        target = self.scatter_target or Point(len(grid)-2, 1)
                        ns = bfs_shortest(current, target, grid)
                        if ns: chosen = ns
            # move to chosen (safety: ensure chosen is valid)
            if chosen is None:
                break
            self.last_move = Point(chosen.r - self.r, chosen.c - self.c)
            self.r, self.c = chosen.r, chosen.c
        # subtract applied moves
        self._move_acc -= moves

# -----------------------
# Main Game class
# -----------------------
class PacmanGame:
    def __init__(self, seed=42, windowed=True):
        pygame.init()
        # start in fullscreen using desktop resolution so the game fills the screen
        info = pygame.display.Info()
        global WIDTH, HEIGHT
        # default to desktop size, but when in windowed mode compute a window
        # size that guarantees the map + UI fits without going fullscreen.
        desktop_w, desktop_h = info.current_w, info.current_h
        # logical map pixel size
        map_w = MAP_COLS * TILE
        map_h = MAP_ROWS * TILE
        # default offsets used in draw()
        offset_y = 40
        # preferred window size to fit the entire playfield plus UI margins
        preferred_w = max(896, map_w + 200)
        preferred_h = max(720, map_h + offset_y + 60)
        # cap to available desktop size with small margins (taskbar etc.)
        win_w = min(preferred_w, desktop_w - 40)
        win_h = min(preferred_h, desktop_h - 60)
        WIDTH, HEIGHT = int(win_w), int(win_h)
        # allow windowed mode for easier testing/dev; otherwise fullscreen
        if windowed:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            self.screen = pygame.display.set_mode((desktop_w, desktop_h), pygame.FULLSCREEN)
        pygame.display.set_caption("PacPy - Prototype")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.bigfont = pygame.font.SysFont("Arial", 48)
        self.seed = seed
        self.lcg = LCG(seed)
        # sensible defaults before loading settings
        self.game_speed = 1.0
        self.sound_enabled = True
        # ensure asset dirs exist so user-provided sprites persist
        try:
            (DATA_DIR / 'assets' / 'ghost_images').mkdir(parents=True, exist_ok=True)
            (DATA_DIR / 'assets' / 'player_images').mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # load persisted settings (seed, game_speed)
        s = self.load_settings()
        if s:
            try:
                self.seed = int(s.get('seed', self.seed))
            except Exception:
                pass
            try:
                self.game_speed = float(s.get('game_speed', self.game_speed))
            except Exception:
                pass
            try:
                # restore sound preference if present
                self.sound_enabled = bool(s.get('sound_enabled', self.sound_enabled))
            except Exception:
                pass
            # re-seed LCG if seed changed
            self.lcg = LCG(self.seed)
        # ensure level exists before generating level-specific map
        self.level = 1
        self.grid = generate_level_map(MAP_COLS, MAP_ROWS, level=self.level, seed=self.seed)
        # count pellets and spawn entities from map
        self.pellet_count = 0
        self._spawn_entities_from_grid()
        # game state
        self.score = 0
        self.lives = 3
    # preserve game_speed loaded from settings; don't reset here
        self.paused = False
        # player animation state
        self.player_anim_time = 0.0
        self.player_anim_index = 0
        # seconds per frame for player animation
        self.player_anim_speed = 0.12
        # overlay text (e.g., READY!) shown until overlay_until timestamp
        self.overlay_text = None
        self.overlay_until = 0.0
        # audio
        pygame.mixer.init()
        self.sfx_pellet = None
        self.sfx_power = None
        self.sfx_death = None
        self.load_sounds()
        # reserve dedicated channels to reduce accidental overlap
        try:
            # ensure at least 4 channels
            pygame.mixer.set_num_channels(max(8, pygame.mixer.get_num_channels()))
        except Exception:
            pass
        try:
            # allocate a specific channel index for pellet chomp sound
            self._pellet_channel = pygame.mixer.Channel(6)
        except Exception:
            self._pellet_channel = None
        # UI toggles
        self.show_menu = True
        try:
            self.stop_all_sounds()
        except Exception:
            pass
        self.menu_selection = 0
        # highscores
        self.highscores = load_highscores()
        # diagnostic startup message
        try:
            print(f"[pacman] initialized (seed={self.seed}, windowed={windowed})")
        except Exception:
            pass

    def load_settings(self):
        p = DATA_DIR / "settings.json"
        if not p.exists():
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _count_pellets(self):
        """Count pellets (regular pellets represented by 0) on current grid."""
        self.pellet_count = sum(1 for row in self.grid for v in row if v == 0)

    def _spawn_entities_from_grid(self):
        """Populate self.player and self.ghosts from markers (3 for player, 4 for ghosts).
        Clears the markers from grid and ensures sensible fallbacks.
        """
        self.player = None
        self.ghosts = []
        rows = len(self.grid); cols = len(self.grid[0])
        roles = [GhostBehavior.BLINKY, GhostBehavior.PINKY, GhostBehavior.INKY, GhostBehavior.CLYDE]
        colors = [(255,0,0),(255,184,255),(0,255,255),(255,165,0)]
        for r in range(rows):
            for c in range(cols):
                v = self.grid[r][c]
                if v == 3:
                    self.player = Player(r, c)
                    self.grid[r][c] = 0
                elif v == 4:
                    behavior = roles[len(self.ghosts) % len(roles)]
                    color = colors[len(self.ghosts) % len(colors)]
                    g = Ghost(r, c, behavior=behavior, seed=self.lcg.randint(1, 9999999), scatter_target=Point(1,1), color=color)
                    self.ghosts.append(g)
                    self.grid[r][c] = 0
        if not self.player:
            # fallback player spawn
            self.player = Player(rows//2+4, cols//2)
        # update pellet count
        self._count_pellets()

    def save_settings(self):
        p = DATA_DIR / "settings.json"
        d = {"seed": self.seed, "game_speed": self.game_speed, "sound_enabled": getattr(self, 'sound_enabled', True)}
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_sounds(self):
        # Robust audio loading with diagnostics
        try:
            # ensure mixer is initialized; try fallback params if not
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
            except Exception:
                # try common fallback parameters
                try:
                    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                except Exception as e:
                    print("[audio] mixer.init failed:", e)

            repo_candidate = Path("external_repos") / "PythonPacman"
            repo_base = repo_candidate if repo_candidate.exists() else None

            # build a list of search roots (repo assets, repo root, data dir and common subfolders)
            search_roots = []
            if repo_base:
                search_roots.append(repo_base / "assets")
                search_roots.append(repo_base)
            search_roots.append(DATA_DIR / "assets")
            search_roots.append(DATA_DIR / "sounds")
            search_roots.append(DATA_DIR)
            # keep only existing dirs
            search_roots = [p for p in search_roots if p is not None and p.exists()]

            def find_file(name):
                # exact name in roots
                for root in search_roots:
                    p = root / name
                    if p.exists():
                        return p
                # search recursively (rglob)
                for root in search_roots:
                    try:
                        for f in root.rglob(name):
                            return f
                    except Exception:
                        pass
                # fallback: search by filename match ignoring case
                lname = name.lower()
                for root in search_roots:
                    try:
                        for f in root.rglob('*'):
                            if f.name.lower() == lname:
                                return f
                    except Exception:
                        pass
                return None

            # music candidates - generic background music (do NOT include intro/intermission SFX)
            music_candidates = ["music.ogg","music.mp3","bgm.ogg","bgm.mp3"]
            music_path = None
            for m in music_candidates:
                p = find_file(m)
                if p:
                    music_path = p; break
            if music_path:
                try:
                    pygame.mixer.music.load(str(music_path))
                    # load music but do not auto-play it (we want silence on menus)
                    pygame.mixer.music.set_volume(0.6 if self.sound_enabled else 0.0)
                    print(f"[audio] music loaded: {music_path}")
                except Exception as e:
                    print(f"[audio] failed to load music {music_path}: {e}")

            # sfx mapping - include the pacman_data/sounds filenames as highest priority
            sfx_map = {
                # pellet sound (chomp)
                'sfx_pellet': [
                    "pacman_chomp.wav", "pacman_chomp.ogg",
                    "pellet.wav","pellet.ogg","pellet.mp3"
                ],
                # power/pickup sounds
                'sfx_power': [
                    "pacman_eatfruit.wav", "pacman_extrapac.wav",
                    "power.wav","power.ogg","power.mp3"
                ],
                # eat ghost
                'sfx_eatghost': [
                    "pacman_eatghost.wav", "eatghost.wav","eatghost.ogg","eatghost.mp3"
                ],
                # death
                'sfx_death': [
                    "pacman_death.wav", "death.wav","death.ogg","death.mp3"
                ],
            }
            for attr, names in sfx_map.items():
                loaded = None
                for nm in names:
                    p = find_file(nm)
                    if p:
                        try:
                            loaded = pygame.mixer.Sound(str(p))
                            setattr(self, attr, loaded)
                            print(f"[audio] loaded {attr} from {p}")
                            break
                        except Exception as e:
                            print(f"[audio] failed to load {p}: {e}")
                if loaded is None:
                    setattr(self, attr, None)

            # explicit start/intermission sounds (prefer them as Sound objects so we can play once)
            try:
                b = find_file('pacman_beginning.wav') or find_file('pacman_beginning.ogg')
                if b:
                    try:
                        self.sfx_begin = pygame.mixer.Sound(str(b))
                        self.sfx_begin.set_volume(0.9)
                        print(f"[audio] loaded sfx_begin from {b}")
                    except Exception:
                        self.sfx_begin = None
                else:
                    self.sfx_begin = None
            except Exception:
                self.sfx_begin = None

            try:
                inter = find_file('pacman_intermission.wav') or find_file('pacman_intermission.ogg')
                if inter:
                    try:
                        self.sfx_intermission = pygame.mixer.Sound(str(inter))
                        self.sfx_intermission.set_volume(0.8)
                        print(f"[audio] loaded sfx_intermission from {inter}")
                    except Exception:
                        self.sfx_intermission = None
                else:
                    self.sfx_intermission = None
            except Exception:
                self.sfx_intermission = None

            # set volumes if present (consolidated)
            for name, vol in (('sfx_pellet', 0.6), ('sfx_power', 0.7), ('sfx_eatghost', 0.8), ('sfx_death', 0.9)):
                try:
                    s = getattr(self, name, None)
                    if s is not None:
                        try:
                            s.set_volume(vol)
                        except Exception:
                            pass
                except Exception:
                    pass

            # diagnostic summary
            loaded = [k for k in ('sfx_pellet','sfx_power','sfx_eatghost','sfx_death') if getattr(self, k, None) is not None]
            if loaded:
                print(f"[audio] SFX loaded: {', '.join(loaded)}")
            else:
                print("[audio] no SFX loaded (checked search roots)")

            # load sprite assets (ghosts and player frames) if present
            try:
                self.ghost_sprites = {}
                self.player_sprites = []
                # prefer assets located in DATA_DIR/assets so user-provided images
                # are persistent and not lost by code updates
                asset_paths = []
                if repo_base:
                    asset_paths.append(repo_base / "assets")
                asset_paths.append(DATA_DIR / "assets")
                asset_paths.append(DATA_DIR)
                for ap in asset_paths:
                    # ghost images
                    gdir = ap / "ghost_images"
                    if gdir.exists():
                        for f in gdir.iterdir():
                            name = f.stem.lower()
                            try:
                                img = pygame.image.load(str(f)).convert_alpha()
                                self.ghost_sprites[name] = img
                            except Exception:
                                pass
                        # player images
                        pdir = ap / "player_images"
                        if pdir.exists():
                            for f in sorted(pdir.iterdir()):
                                try:
                                    img = pygame.image.load(str(f)).convert_alpha()
                                    self.player_sprites.append(img)
                                except Exception:
                                    pass
                    # additionally, accept Adobe Illustrator (.ai) embedded thumbnail
                    # in the user's images folder (common from vector packs); attempt
                    # to extract an embedded JPEG thumbnail and add it as a sprite.
                    # previously we attempted to extract thumbnails from AI files here,
                    # but that feature has been reverted to allow manual per-asset imports.
            except Exception:
                self.ghost_sprites = {}
                self.player_sprites = []

        except Exception as e:
            print("[audio] unexpected error in load_sounds:", e)
            self.sfx_pellet = None
            self.sfx_power = None
            self.sfx_eatghost = None
            self.sfx_death = None

    def stop_all_sounds(self):
        """Stop music and any loaded sound effects."""
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        for k in ('sfx_pellet','sfx_power','sfx_eatghost','sfx_death'):
            s = getattr(self, k, None)
            try:
                if s is not None:
                    # stop Sound if it supports stop()
                    try:
                        s.stop()
                    except Exception:
                        pass
            except Exception:
                pass

    def play_test_sfx(self):
        """Play a short sample (pellet) to test SFX output."""
        try:
            s = getattr(self, 'sfx_pellet', None)
            if s is not None and self.sound_enabled:
                try:
                    s.play()
                except Exception:
                    pass
        except Exception:
            pass

    def play_sfx(self, sound):
        """Play a pygame.mixer.Sound if sound is enabled and sound object exists."""
        if not getattr(self, 'sound_enabled', True):
            return
        # ensure mixer is initialized
        try:
            if not pygame.mixer.get_init():
                try:
                    pygame.mixer.init()
                except Exception:
                    return
        except Exception:
            # if mixer.get_init is not available for some reason, bail safely
            return
        if sound:
            try:
                sound.play()
            except Exception:
                pass

    def play_pellet_sfx(self):
        s = getattr(self, 'sfx_pellet', None)
        if s is not None:
            try:
                s.play()
            except Exception:
                pass

    def reset_level(self, reset_score: bool = False):
        # generate map for current level
        self.grid = generate_level_map(MAP_COLS, MAP_ROWS, level=self.level, seed=(self.seed + self.level))
        # spawn entities and count pellets from the generated grid
        self._spawn_entities_from_grid()
        if reset_score:
            self.score = 0
            self.lives = 3
        # create a static surface with walls to speed up drawing
        try:
            self.static_map_surf = pygame.Surface((MAP_COLS*TILE, MAP_ROWS*TILE), flags=pygame.SRCALPHA)
            self.static_map_surf.fill((0,0,0,0))
            for r in range(len(self.grid)):
                for c in range(len(self.grid[0])):
                    if self.grid[r][c] == 1:
                        x = c*TILE; y = r*TILE
                        pygame.draw.rect(self.static_map_surf, (0,0,128), (x,y,TILE,TILE))
        except Exception:
            self.static_map_surf = None


    def update_game_logic(self, dt):
        if self.paused or self.show_menu:
            return
        # player movement is driven by keypresses set in event loop (next_dir)
        if self.player is not None:
            # apply global game_speed scaling to player movement as well so
            # pacman matches ghost pacing when game_speed != 1.0
            self.player.update(dt * self.game_speed, self.grid)
            # consume pellet on player's tile
            v = self.grid[self.player.r][self.player.c]
        else:
            return
        if v == 0:
            self.grid[self.player.r][self.player.c] = -1  # consumed
            self.score += 10
            self.pellet_count -= 1
            # play pellet sfx if available, using a reserved channel to avoid overlap
            try:
                s = getattr(self, 'sfx_pellet', None)
                if s is not None and self.sound_enabled:
                    ch = getattr(self, '_pellet_channel', None)
                    if ch is not None:
                        try:
                            # play only if channel not busy to avoid rapid overlap
                            if not ch.get_busy():
                                ch.play(s)
                        except Exception:
                            try: s.play()
                            except Exception: pass
                    else:
                        try: s.play()
                        except Exception: pass
            except Exception:
                pass
        elif v == 2:
            self.grid[self.player.r][self.player.c] = -1
            self.score += 50
            # frighten ghosts
            for g in self.ghosts:
                g.state = "frightened"; g.state_timer = 10.0
            # play power pellet sfx
            self.play_sfx(getattr(self, 'sfx_power', None))
        # update ghosts
        for g in self.ghosts:
            # guard player existence
            if self.player is not None:
                g.update(dt * self.game_speed, self.grid, self.player, self.ghosts)
            else:
                # create a temp player-like point at center
                temp = Player(len(self.grid)//2+4, len(self.grid[0])//2)
                g.update(dt * self.game_speed, self.grid, temp, self.ghosts)
            # collision with player?
            if self.player is not None and g.r == self.player.r and g.c == self.player.c:
                if g.state == "frightened":
                    # player eats ghost
                    self.score += 200
                    # send ghost to "eaten" state -> teleport home
                    g.r, g.c = len(self.grid)//2, len(self.grid[0])//2
                    g.state = "scatter"
                    # play eat-ghost sfx
                    self.play_sfx(getattr(self, 'sfx_eatghost', None))
                else:
                    # player dies
                    self.lives -= 1
                    # death sfx: play and wait for its duration to avoid overlap
                    sd = getattr(self, 'sfx_death', None)
                    try:
                        if sd is not None and self.sound_enabled:
                            # try to obtain length
                            length = None
                            try:
                                length = float(sd.get_length())
                            except Exception:
                                length = None
                            chd = pygame.mixer.find_channel(True)
                            if chd is not None:
                                try:
                                    chd.play(sd, loops=0)
                                except Exception:
                                    pass
                                # wait for reported length if available, otherwise channel busy
                                if length is not None:
                                    t0 = time.time()
                                    while time.time() - t0 < length:
                                        for ev2 in pygame.event.get():
                                            if ev2.type == pygame.QUIT:
                                                pygame.quit(); sys.exit()
                                        self.draw(); self.clock.tick(FPS)
                                else:
                                    while chd.get_busy():
                                        for ev2 in pygame.event.get():
                                            if ev2.type == pygame.QUIT:
                                                pygame.quit(); sys.exit()
                                        self.draw(); self.clock.tick(FPS)
                                # small gap 0.5s to avoid overlap
                                t0 = time.time()
                                while time.time() - t0 < 0.5:
                                    for ev2 in pygame.event.get():
                                        if ev2.type == pygame.QUIT:
                                            pygame.quit(); sys.exit()
                                    self.draw(); self.clock.tick(FPS)
                    except Exception:
                        pass

                    if self.lives <= 0:
                        self.game_over()
                    else:
                        # play intro and reset positions while it plays
                        sb = getattr(self, 'sfx_begin', None)
                        try:
                            # reset positions immediately
                            rows = len(self.grid); cols = len(self.grid[0])
                            pr, pc = rows//2 + 4, cols//2
                            if self.player is not None:
                                self.player.r, self.player.c = pr, pc
                                self.player.dir = (0,0); self.player.next_dir = (0,0)
                                try: self.player._move_acc = 0.0
                                except Exception: pass
                            grs = [(rows//2, cols//2 - 2), (rows//2, cols//2 + 2), (rows//2 - 2, cols//2), (rows//2 + 2, cols//2)]
                            for i, g in enumerate(self.ghosts):
                                if i < len(grs):
                                    g.r, g.c = grs[i]
                                else:
                                    placed = False
                                    for dr in range(-2,3):
                                        for dc in range(-2,3):
                                            nr, nc = rows//2 + dr, cols//2 + dc
                                            if 0 <= nr < rows and 0 <= nc < cols and self.grid[nr][nc] == 0:
                                                g.r, g.c = nr, nc
                                                placed = True; break
                                        if placed: break
                                g.state = 'scatter'; g.state_timer = 7.0; g.last_move = None
                                try: g._move_acc = 0.0
                                except Exception: pass
                            # now play intro if available and wait its duration
                            if sb is not None and self.sound_enabled:
                                try:
                                    blen = None
                                    try: blen = float(sb.get_length())
                                    except Exception: blen = None
                                    chb = pygame.mixer.find_channel(True)
                                    if chb is not None:
                                        try: chb.play(sb, loops=0)
                                        except Exception: pass
                                        if blen is not None:
                                            t0 = time.time()
                                            while time.time() - t0 < blen:
                                                for ev2 in pygame.event.get():
                                                    if ev2.type == pygame.QUIT:
                                                        pygame.quit(); sys.exit()
                                                self.draw(); self.clock.tick(FPS)
                                        else:
                                            while chb.get_busy():
                                                for ev2 in pygame.event.get():
                                                    if ev2.type == pygame.QUIT:
                                                        pygame.quit(); sys.exit()
                                                self.draw(); self.clock.tick(FPS)
                                except Exception:
                                    pass
                        except Exception:
                            pass
        # win condition: all pellets consumed
        if self.pellet_count <= 0:
            self.level += 1
            self.reset_level()

    def game_over(self):
        # Prompt for 4-letter initials (blocking)
        initials = self.prompt_initials_blocking()
        if not initials:
            initials = "----"
        # save score to highscores with initials
        entry = {"initials": initials, "score": self.score}
        self.highscores.append(entry)
        self.highscores = sorted(self.highscores, key=lambda x: x.get("score", 0), reverse=True)
        save_highscores(self.highscores)
        self.show_menu = True
        try:
            self.stop_all_sounds()
        except Exception:
            pass

    def prompt_initials_blocking(self):
        # modal to collect up to 4 uppercase letters
        initials = [" "," "," "," "]
        idx = 0
        info_font = pygame.font.SysFont("Arial", 28)
        prompt = self.font.render("Enter 4-letter initials:", True, (255,255,255))
        # reset clock so time spent in this blocking modal doesn't leak into game dt
        try:
            self.clock.tick(FPS)
        except Exception:
            pass
        while True:
            # draw background
            self.screen.fill((0,0,0))
            self.screen.blit(prompt, (WIDTH//2 - prompt.get_width()//2, HEIGHT//2 - 80))
            # render initials
            s = "".join(ch if ch != " " else "_" for ch in initials)
            surf = self.bigfont.render(s, True, (255,255,0))
            self.screen.blit(surf, (WIDTH//2 - surf.get_width()//2, HEIGHT//2 - 20))
            info = info_font.render("Use letters, Backspace, Enter to confirm", True, (200,200,200))
            self.screen.blit(info, (WIDTH//2 - info.get_width()//2, HEIGHT//2 + 80))
            pygame.display.flip()
            ev = pygame.event.wait()
            if ev.type == pygame.QUIT:
                try: self.clock.tick(FPS)
                except Exception: pass
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    # finish when all chars entered or allow early confirm
                    try: self.clock.tick(FPS)
                    except Exception: pass
                    return "".join(ch if ch != " " else "_" for ch in initials)
                elif ev.key == pygame.K_BACKSPACE:
                    if idx > 0:
                        idx -= 1
                        initials[idx] = " "
                else:
                    # accept letters only
                    if ev.unicode and ev.unicode.isalpha() and idx < 4:
                        initials[idx] = ev.unicode.upper()[0]
                        idx += 1

    def draw(self):
        self.screen.fill((0,0,0))
        # draw map
        offset_x = (WIDTH - MAP_COLS*TILE)//2
        offset_y = 40
        for r in range(len(self.grid)):
            for c in range(len(self.grid[0])):
                x = offset_x + c*TILE
                y = offset_y + r*TILE
                val = self.grid[r][c]
                if val == 1:
                    pygame.draw.rect(self.screen, (0,0,128), (x,y,TILE,TILE))  # wall
                elif val == 0:
                    # pellet
                    cx, cy = x + TILE//2, y + TILE//2
                    pygame.draw.circle(self.screen, (255,255,255), (cx,cy), 3)
                elif val == 2:
                    cx, cy = x + TILE//2, y + TILE//2
                    pygame.draw.circle(self.screen, (255,255,0), (cx,cy), 6)
        # draw player (animated if frames available)
        if self.player is not None:
            px = offset_x + self.player.c*TILE + TILE//2
            py = offset_y + self.player.r*TILE + TILE//2
            frames_available = getattr(self, 'player_frames', None)
            if frames_available:
                # choose direction key
                dir_key = 'right'
                dr, dc = self.player.dir if getattr(self.player, 'dir', None) is not None else (0,1)
                if (dr, dc) == (0,-1): dir_key = 'left'
                elif (dr, dc) == (-1,0): dir_key = 'up'
                elif (dr, dc) == (1,0): dir_key = 'down'
                # prefer keyed frames, else default
                flist = frames_available.get(dir_key) or frames_available.get('default') or frames_available.get('right')
                if flist:
                    # advance animation timer
                    try:
                        self.player_anim_time += self.clock.get_time()/1000.0
                    except Exception:
                        self.player_anim_time += self.player_anim_speed
                    if self.player_anim_time >= self.player_anim_speed:
                        self.player_anim_time -= self.player_anim_speed
                        self.player_anim_index = (self.player_anim_index + 1) % len(flist)
                    img = flist[self.player_anim_index]
                    try:
                        im = pygame.transform.scale(img, (TILE-4, TILE-4))
                        self.screen.blit(im, (px - (TILE-4)//2, py - (TILE-4)//2))
                    except Exception:
                        try:
                            self.screen.blit(img, (px - img.get_width()//2, py - img.get_height()//2))
                        except Exception:
                            pygame.draw.circle(self.screen, (255,255,0), (px,py), TILE//2 - 2)
                else:
                    pygame.draw.circle(self.screen, (255,255,0), (px,py), TILE//2 - 2)
            else:
                pygame.draw.circle(self.screen, (255,255,0), (px,py), TILE//2 - 2)
        # draw ghosts
        for g in self.ghosts:
            gx = offset_x + g.c*TILE + TILE//2
            gy = offset_y + g.r*TILE + TILE//2
            # ghost sprite by color name if available
            img = None
            if getattr(self, 'ghost_sprites', None):
                # try keys: red, blue, pink, orange, dead, powerup
                key = 'red'
                if g.state == 'frightened': key = 'powerup'
                if g.color == (255,165,0): key = 'orange'
                if g.color == (0,255,255): key = 'blue'
                if g.color == (255,184,255): key = 'pink'
                if g.color == (255,0,0): key = 'red'
                img = self.ghost_sprites.get(key) or self.ghost_sprites.get('red')
            if img is not None:
                im = pygame.transform.scale(img, (TILE-4, TILE-4))
                self.screen.blit(im, (gx - (TILE-4)//2, gy - (TILE-4)//2))
            else:
                col = g.color if g.state != "frightened" else (100,100,255)
                pygame.draw.circle(self.screen, col, (gx,gy), TILE//2 - 2)
        # UI
        score_surf = self.font.render(f"SCORE: {self.score}", True, (255,255,255))
        lives_surf = self.font.render(f"LIVES: {self.lives}", True, (255,255,255))
        level_surf = self.font.render(f"LEVEL: {self.level}", True, (255,255,255))
        self.screen.blit(score_surf, (10, 6))
        self.screen.blit(lives_surf, (WIDTH - 140, 6))
        self.screen.blit(level_surf, (WIDTH//2 - 40, 6))
        # draw menu if active
        if self.show_menu:
            rect = pygame.Rect(WIDTH//2 - 240, HEIGHT//2 - 160, 480, 320)
            pygame.draw.rect(self.screen, (20,20,60), rect)
            pygame.draw.rect(self.screen, (200,200,255), rect, 4)
            title = self.bigfont.render("PACPY", True, (255,255,0))
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 140))
            items = ["Start Game", f"Seed: {self.seed}", f"Game Speed: {self.game_speed:.1f}x", "High Scores", "Quit"]
            for i, it in enumerate(items):
                c = (255,255,255) if i != self.menu_selection else (255,255,0)
                surf = self.font.render(it, True, c)
                self.screen.blit(surf, (WIDTH//2 - surf.get_width()//2, HEIGHT//2 - 60 + i*36))
        pygame.display.flip()
        # draw overlay text on top if active
        try:
            if self.overlay_text and time.time() < getattr(self, 'overlay_until', 0):
                txt = self.bigfont.render(self.overlay_text, True, (255,255,0))
                self.screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - txt.get_height()//2))
                pygame.display.update()
        except Exception:
            pass

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS)/1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if self.show_menu:
                        if ev.key == pygame.K_UP:
                            self.menu_selection = max(0, self.menu_selection - 1)
                        elif ev.key == pygame.K_DOWN:
                            self.menu_selection = min(4, self.menu_selection + 1)
                        elif ev.key == pygame.K_RETURN:
                            if self.menu_selection == 0:
                                # start a fresh game: ensure menu is silent, play intro sound once, then begin
                                try:
                                    pygame.mixer.music.stop()
                                except Exception:
                                    pass
                                self.show_menu = False
                                self.level = 1
                                self.reset_level(reset_score=True)
                                # play pacman beginning sound if available (use a dedicated Channel
                                # and wait until it's finished; this prevents accidental looping)
                                try:
                                    if getattr(self, 'sfx_begin', None) and self.sound_enabled:
                                        try:
                                            # stop any music to ensure intro SFX is distinct
                                            try:
                                                pygame.mixer.music.stop()
                                            except Exception:
                                                pass
                                            ch = pygame.mixer.find_channel(True)
                                            sb = getattr(self, 'sfx_begin', None)
                                            if sb is not None:
                                                ch.play(sb, loops=0)
                                            # wait while the channel is busy, keeping UI responsive
                                            while ch.get_busy():
                                                for ev2 in pygame.event.get():
                                                    if ev2.type == pygame.QUIT:
                                                        pygame.quit(); sys.exit()
                                                self.draw()
                                                self.clock.tick(FPS)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            elif self.menu_selection == 1:
                                # change seed
                                self.seed = (self.seed + 137) % 1000000
                                self.lcg = LCG(self.seed)
                                try: self.save_settings()
                                except Exception: pass
                            elif self.menu_selection == 2:
                                self.game_speed = round(self.game_speed + 0.5, 1)
                                if self.game_speed > 3.0: self.game_speed = 0.5
                                try: self.save_settings()
                                except Exception: pass
                            elif self.menu_selection == 3:
                                # show highscores immediately (simple)
                                self.display_highscores_blocking()
                            elif self.menu_selection == 4:
                                running = False
                    else:
                        # game controls (if player exists)
                        if self.player is not None:
                            if ev.key in (pygame.K_LEFT, pygame.K_a):
                                self.player.next_dir = (0,-1)
                            elif ev.key in (pygame.K_RIGHT, pygame.K_d):
                                self.player.next_dir = (0,1)
                            elif ev.key in (pygame.K_UP, pygame.K_w):
                                self.player.next_dir = (-1,0)
                            elif ev.key in (pygame.K_DOWN, pygame.K_s):
                                self.player.next_dir = (1,0)
                        # global pause key
                        if ev.key in (pygame.K_ESCAPE, pygame.K_p):
                            self.pause_menu_blocking()
                elif ev.type == pygame.KEYUP:
                    pass
            # update game logic
            self.update_game_logic(dt)
            # draw
            self.draw()
        pygame.quit()
        try:
            self.save_settings()
        except Exception:
            pass
        sys.exit()

    def display_highscores_blocking(self):
        # render highscores temporarily until keypress
        bg = pygame.Surface((WIDTH, HEIGHT))
        bg.fill((8,8,20))
        title = self.bigfont.render("HIGH SCORES", True, (255,255,0))
        # reset clock so time spent viewing highscores won't produce a later large dt
        try:
            self.clock.tick(FPS)
        except Exception:
            pass
        while True:
            self.screen.blit(bg, (0,0))
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 60))
            for i, e in enumerate(self.highscores[:10]):
                label = e.get('initials') or e.get('name') or '----'
                s = f"{i+1}. {label} - {e.get('score',0)}"
                surf = self.font.render(s, True, (255,255,255))
                self.screen.blit(surf, (WIDTH//2 - surf.get_width()//2, 160 + i*28))
            info = self.font.render("Press any key to go back", True, (200,200,200))
            self.screen.blit(info, (WIDTH//2 - info.get_width()//2, HEIGHT - 80))
            pygame.display.flip()
            ev = pygame.event.wait()
            if ev.type in (pygame.KEYDOWN, pygame.QUIT):
                try:
                    self.clock.tick(FPS)
                except Exception:
                    pass
                break

    def pause_menu_blocking(self):
        """Blocking pause menu with Resume, Exit to Main Menu, Options"""
        # reset clock so time spent in pause doesn't affect the main loop dt
        try:
            self.clock.tick(FPS)
        except Exception:
            pass
        self.paused = True
        sel = 0
        items = ["Resume Play", "Exit to Main Menu", "Options"]
        info_font = pygame.font.SysFont("Arial", 20)
        # ensure current frame is rendered and capture the map area so we can avoid covering it
        self.draw()
        map_w = MAP_COLS * TILE
        map_h = MAP_ROWS * TILE
        map_x = (WIDTH - map_w) // 2
        map_y = 40
        map_rect = pygame.Rect(map_x, map_y, map_w, map_h)
        try:
            map_snapshot = self.screen.subsurface(map_rect).copy()
        except Exception:
            map_snapshot = None

        panel_w, panel_h = 360, 220
        # choose a panel position that does not intersect the map rect
        candidates = [
            pygame.Rect(WIDTH - panel_w - 20, 20, panel_w, panel_h),  # top-right
            pygame.Rect(20, 20, panel_w, panel_h),  # top-left
            pygame.Rect(WIDTH - panel_w - 20, HEIGHT - panel_h - 20, panel_w, panel_h),  # bottom-right
            pygame.Rect(20, HEIGHT - panel_h - 20, panel_w, panel_h),  # bottom-left
            pygame.Rect(map_x + map_w + 10, map_y, panel_w, panel_h),  # right of map
            pygame.Rect(map_x, map_y + map_h + 10, panel_w, panel_h),  # below map
        ]
        panel_rect = None
        for cand in candidates:
            if 0 <= cand.left and cand.right <= WIDTH and 0 <= cand.top and cand.bottom <= HEIGHT and not cand.colliderect(map_rect):
                panel_rect = cand; break
        if panel_rect is None:
            # fallback to center but shrink so it doesn't fully cover
            panel_rect = pygame.Rect(WIDTH//2 - panel_w//2, HEIGHT//2 - panel_h//2, panel_w, panel_h)

        while True:
            # draw a dimmed background but restore the map area so playfield stays visible
            overlay = pygame.Surface((WIDTH, HEIGHT), flags=pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            if map_snapshot:
                self.screen.blit(map_snapshot, (map_rect.x, map_rect.y))

            # draw panel background
            pygame.draw.rect(self.screen, (30,30,60), panel_rect)
            pygame.draw.rect(self.screen, (200,200,255), panel_rect, 3)
            title = self.bigfont.render("PAUSED", True, (255,255,0))
            self.screen.blit(title, (panel_rect.centerx - title.get_width()//2, panel_rect.top + 12))
            for i, it in enumerate(items):
                c = (255,255,0) if i == sel else (255,255,255)
                surf = self.font.render(it, True, c)
                self.screen.blit(surf, (panel_rect.centerx - surf.get_width()//2, panel_rect.top + 80 + i*36))
            help_s = info_font.render("Use Up/Down, Enter. Esc or P to resume.", True, (200,200,200))
            self.screen.blit(help_s, (panel_rect.centerx - help_s.get_width()//2, panel_rect.bottom - 36))
            pygame.display.flip()

            ev = pygame.event.wait()
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_p):
                    self.paused = False
                    break
                elif ev.key == pygame.K_UP:
                    sel = max(0, sel-1)
                elif ev.key == pygame.K_DOWN:
                    sel = min(len(items)-1, sel+1)
                elif ev.key == pygame.K_RETURN:
                    if sel == 0:
                        # Resume
                        self.paused = False
                        break
                    elif sel == 1:
                        # Exit to main menu
                        self.show_menu = True
                        self.menu_selection = 0
                        self.paused = False
                        break
                    elif sel == 2:
                        # Options
                        self.options_menu_blocking()
            # continue loop
        # reset clock after leaving pause menu so the main loop timestep won't jump
        try:
            self.clock.tick(FPS)
        except Exception:
            pass

    def options_menu_blocking(self):
        """Blocking options menu to change game parameters"""
        # reset clock so time spent in options doesn't affect the main loop dt
        try:
            self.clock.tick(FPS)
        except Exception:
            pass
        sel = 0
        options = ["Game Speed", "Sound: ", "Back"]
        info_font = pygame.font.SysFont("Arial", 18)
        # ensure attribute
        if not hasattr(self, 'sound_enabled'):
            self.sound_enabled = True
        while True:
            overlay = pygame.Surface((WIDTH, HEIGHT), flags=pygame.SRCALPHA)
            overlay.fill((0,0,0,200))
            self.screen.blit(overlay, (0,0))
            title = self.bigfont.render("OPTIONS", True, (255,255,0))
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 160))
            for i, it in enumerate(options):
                if it.startswith("Sound"):
                    label = f"Sound: {'On' if self.sound_enabled else 'Off'}"
                elif it.startswith("Game Speed"):
                    label = f"Game Speed: {self.game_speed:.1f}x"
                else:
                    label = it
                c = (255,255,0) if i == sel else (255,255,255)
                surf = self.font.render(label, True, c)
                self.screen.blit(surf, (WIDTH//2 - surf.get_width()//2, HEIGHT//2 - 40 + i*36))
            help_s = info_font.render("Use Up/Down, Left/Right to change, Enter to select", True, (200,200,200))
            self.screen.blit(help_s, (WIDTH//2 - help_s.get_width()//2, HEIGHT - 80))
            pygame.display.flip()

            ev = pygame.event.wait()
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_UP:
                    sel = max(0, sel-1)
                elif ev.key == pygame.K_DOWN:
                    sel = min(len(options)-1, sel+1)
                elif ev.key == pygame.K_LEFT:
                    if sel == 0:
                        # decrease game speed
                        self.game_speed = max(0.5, round(self.game_speed - 0.1, 1))
                        try: self.save_settings()
                        except Exception: pass
                elif ev.key == pygame.K_RIGHT:
                    if sel == 0:
                        # increase game speed
                        self.game_speed = min(3.0, round(self.game_speed + 0.1, 1))
                        try: self.save_settings()
                        except Exception: pass
                elif ev.key == pygame.K_RETURN:
                    if sel == 1:
                        # toggle sound
                        self.sound_enabled = not self.sound_enabled
                        # apply action
                        if self.sound_enabled:
                            # try to init mixer and reload sounds
                            try:
                                if not pygame.mixer.get_init():
                                    pygame.mixer.init()
                            except Exception:
                                pass
                            try:
                                self.load_sounds()
                            except Exception:
                                pass
                            # play a quick test sfx if available
                            try: self.play_test_sfx()
                            except Exception:
                                pass
                        else:
                            # stop music and sfx
                            try: self.stop_all_sounds()
                            except Exception:
                                pass
                        try: self.save_settings()
                        except Exception: pass
                    elif sel == 2:
                        try:
                            self.clock.tick(FPS)
                        except Exception:
                            pass
                        break
                elif ev.key in (pygame.K_ESCAPE, pygame.K_p):
                    try:
                        self.clock.tick(FPS)
                    except Exception:
                        pass
                    break
        # reset clock after leaving options menu
        try:
            self.clock.tick(FPS)
        except Exception:
            pass

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    # if user wants to provide a seed via arg
    seed = 2025
    if len(sys.argv) > 1:
        try:
            for a in sys.argv[1:]:
                try:
                    seed = int(a)
                    break
                except Exception:
                    continue
        except Exception:
            seed = 2025

    # Wrap startup in try/except so when running a bundled exe we capture
    # unhandled exceptions and write them to a log in the data dir. This
    # helps debugging when the exe is run by double-click (no console visible).
    try:
        # always start in windowed mode
        game = PacmanGame(seed=seed, windowed=True)
        game.run()
    except Exception:
        # attempt to write a traceback to pacman_data/pacman_error.log
        try:
            logp = DATA_DIR / "pacman_error.log"
            with open(logp, "a", encoding="utf-8") as f:
                f.write("=== Exception on run: " + time.strftime('%Y-%m-%d %H:%M:%S') + " ===\n")
                traceback.print_exc(file=f)
                f.write("\n")
        except Exception:
            # if logging fails, ignore and fallback to printing to stderr
            try:
                traceback.print_exc()
            except Exception:
                pass
        # If running from a console, pause so the user can see the error
        try:
            if sys.stdin and sys.stdin.isatty():
                input("An error occurred while running PacPy. Press Enter to exit...")
        except Exception:
            pass
        sys.exit(1)
