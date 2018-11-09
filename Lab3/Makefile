# All the plots needed to make the report
ALL_PLOTS=PosExp.png VelExp.png ErrPosExp.png ErrVelExp.png NrgExp.png hVsErrExp.png PosImp.png VelImp.png ErrPosImp.png ErrVelImp.png NrgImp.png PSGeoExp.png PSGeoImp.png PSGeoExp.png PSGeoAna.png PSGeoSym01.png PSGeoSym3.png PSGeoSym7.png NrgSym.png

# Makes report
report.pdf : report.tex diffEqs.py plots log.txt
	pdflatex -shell-escape $< 

# Generates plots
.PHONY : plots
plots : $(ALL_PLOTS)

%.png : diffEqs.py
	python $< $*

# Deletes plots
.PHONY : clean
clean :
	rm -f *.png
	rm -f report.pdf
	rm -f report.log
	rm -f report.aux
	rm -f log.txt

# Generates git log
log.txt :
	git --no-pager log > $@