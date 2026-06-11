# Build configuration for the RSIF manuscript.
# The 2014 rsproca_new class and modern biblatex (>=3.19) emit a few benign,
# recoverable patching warnings ("Patching \addtocontents failed") that set a
# non-zero pdflatex exit code without affecting the output. force_mode lets
# latexmk run the full pdflatex -> biber -> pdflatex -> pdflatex sequence to a
# correct PDF regardless. Build with:  latexmk -pdf main.tex
$pdf_mode      = 1;
$force_mode    = 1;
$pdflatex      = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use    = 2;        # run biber/bibtex as needed
$clean_ext     = 'bbl bcf run.xml synctex.gz';
