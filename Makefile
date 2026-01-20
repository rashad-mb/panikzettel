all:ai.pdfb elehre.pdf 

typst:
	find . -type f -iname "*.typ" -not -name "conf.typ" -not -iname "_*.typ" -exec ./scripts/compile-typst.sh {} \;

effi.pdf: effi.tex panikzettel.cls effi.last-change
	latexmk -output-directory=./build -pdflatex="pdflatex -interaction=nonstopmode" -pdf -use-make effi.tex

afi.pdf: afi.tex panikzettel.cls afi.last-change
	latexmk -output-directory=./build -pdflatex="pdflatex -interaction=nonstopmode" -pdf -use-make afi.tex

ai.pdf: ai.tex panikzettel.cls ai.last-change
	latexmk -output-directory=./build -pdflatex="pdflatex -interaction=nonstopmode" -pdf -use-make ai.tex

%.last-change: %.tex
	echo -n "Version " > $@
	git log --format=oneline -- $< | wc -l >> $@
	echo "---" >> $@
	git log --pretty=format:%ad --date=format:'%d.%m.%Y' -n 1 -- $< >> $@

clean:
	rm -f *.last-change
	latexmk -CA
	rm -rf ./build
