# Makes report
.PHONY : report
report : report.tex *.png
	pdflatex report.tex 

# Generates plots
.PHONY : plots
plots : diffEqs.py
	python diffEqs.py

# Deletes plots
.PHONY : clean
clean :
	rm -f *.png