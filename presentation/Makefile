TEXINPUTS=./template//:

%.pdf: %.tex
	TEXINPUTS=$(TEXINPUTS) pdflatex $<
	TEXINPUTS=$(TEXINPUTS) pdflatex $<

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.pdf
