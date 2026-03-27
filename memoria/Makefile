# Nombre del archivo principal LaTeX (sin la extensión .tex)
MAIN = tfm

# Directorio que contiene los capítulos
CAPITULOS_DIR = capitulos

# Obtener todos los archivos .tex del directorio capitulos
CAPITULOS_TEX = $(wildcard $(CAPITULOS_DIR)/*.tex)

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex bibliografia.bib $(CAPITULOS_TEX)
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex


clean:
	$(RM) $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lot $(MAIN).lof $(CAPITULOS_DIR)/*.aux $(CAPITULOS_DIR)/*.log $(CAPITULOS_DIR)/*~ *~
