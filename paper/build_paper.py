"""Rebuild the peer-readable four-page article from the existing experimental logs.

Requires numpy, pandas, matplotlib and a XeLaTeX installation (latexmk, xelatex).
There are no synthetic data and this command does not retrain the agents.
"""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent


def main():
    if not shutil.which('latexmk') or not shutil.which('xelatex'):
        print('XeLaTeX/latexmk unavailable; report build skipped. Install TeX Live.',file=sys.stderr)
        return 2
    subprocess.run([sys.executable,str(HERE/'make_figures.py')],cwd=ROOT,check=True)
    subprocess.run(['latexmk','-xelatex','-interaction=nonstopmode','-halt-on-error','-jobname=main','main.tex'],cwd=HERE,check=True)
    pdf=HERE/'main.pdf'
    dest=ROOT/'rapport_4_pages.pdf'
    shutil.copy2(pdf,dest)
    print('Report written:',dest)
    return 0

if __name__=='__main__':
    sys.exit(main())
