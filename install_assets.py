"""install_assets.py
Utility to copy downloaded sprite images into pacman_data/assets so they persist
across code updates. Usage:
  python install_assets.py --ghosts path/to/ghost1.png,path/to/ghost2.png --player path/to/pacman1.png

By default files are copied without overwriting existing files. Use --force to replace.
"""
import shutil
from pathlib import Path
import argparse

DATA_DIR = Path(__file__).parent / 'pacman_data'
ASSETS_DIR = DATA_DIR / 'assets'
GHOST_DIR = ASSETS_DIR / 'ghost_images'
PLAYER_DIR = ASSETS_DIR / 'player_images'

def ensure_dirs():
    for d in (ASSETS_DIR, GHOST_DIR, PLAYER_DIR):
        d.mkdir(parents=True, exist_ok=True)

def copy_files(src_list, dest_dir, force=False):
    copied = []
    for s in src_list:
        p = Path(s)
        if not p.exists():
            print(f"skip (not found): {p}")
            continue
        dest = dest_dir / p.name
        if dest.exists() and not force:
            print(f"skip (exists): {dest}")
            continue
        shutil.copy2(str(p), str(dest))
        copied.append(dest)
        print(f"copied -> {dest}")
    return copied

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ghosts', help='Comma separated list of ghost image files')
    parser.add_argument('--player', help='Comma separated list of player image files')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()
    ensure_dirs()
    if args.ghosts:
        ghosts = [s.strip() for s in args.ghosts.split(',') if s.strip()]
        copy_files(ghosts, GHOST_DIR, force=args.force)
    if args.player:
        players = [s.strip() for s in args.player.split(',') if s.strip()]
        copy_files(players, PLAYER_DIR, force=args.force)
    print('done')

if __name__ == '__main__':
    main()
