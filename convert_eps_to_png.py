"""convert_eps_to_png.py
Helper to convert an EPS file (vector) to a PNG raster for use as game sprites.

This script will try to use Inkscape or ImageMagick (magick) if available on PATH.
It does NOT attempt to split a sprite sheet into individual frames (layout unknown).
Instead it exports the EPS to a high-resolution PNG that you can crop into directional
frames (or load as a sheet and provide slicing info).

Usage examples:
    python convert_eps_to_png.py images/pac-man-pixel-characters-2.eps pacman_sheet.png --width 512

If Inkscape or ImageMagick aren't installed, the script will print instructions.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import argparse

def find_tool():
    # prefer Inkscape, then ImageMagick 'magick'
    if shutil.which('inkscape'):
        return 'inkscape'
    if shutil.which('magick'):
        return 'magick'
    return None

def run_inkscape(src, out, width=None, height=None):
    cmd = ['inkscape', str(src), '--export-filename', str(out)]
    if width:
        cmd += ['--export-width', str(width)]
    if height:
        cmd += ['--export-height', str(height)]
    return subprocess.run(cmd, check=False)

def run_magick(src, out, width=None, height=None):
    cmd = ['magick', str(src)]
    if width or height:
        sz = ''
        if width: sz += str(width)
        sz += 'x'
        if height: sz += str(height)
        cmd += ['-resize', sz]
    cmd += [str(out)]
    return subprocess.run(cmd, check=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('src', help='source EPS file')
    p.add_argument('out', help='output PNG file')
    p.add_argument('--width', type=int, help='output width in pixels')
    p.add_argument('--height', type=int, help='output height in pixels')
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        print('Source file not found:', src)
        sys.exit(2)

    tool = find_tool()
    if not tool:
        print('No Inkscape or ImageMagick (magick) found on PATH.')
        print('Install one of them and re-run. Example commands:')
        print('  Inkscape: inkscape input.eps --export-filename=output.png --export-width=512')
        print('  ImageMagick: magick input.eps -resize 512x output.png')
        sys.exit(1)

    print(f'Using tool: {tool} to convert {src} -> {out}')
    if tool == 'inkscape':
        res = run_inkscape(src, out, width=args.width, height=args.height)
    else:
        res = run_magick(src, out, width=args.width, height=args.height)

    if res.returncode == 0:
        print('Conversion successful:', out)
    else:
        print('Conversion failed (return code', res.returncode, ').')
        print('Try running the printed example commands manually.')

if __name__ == '__main__':
    main()
