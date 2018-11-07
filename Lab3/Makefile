# Makes report
.PHONY : report
report : report.tex diffEqs.py PosExp.png VelExp.png ErrPosExp.png ErrVelExp.png NrgExp.png hVsErrExp.png PosImp.png VelImp.png ErrPosImp.png ErrVelImp.png NrgImp.png PSGeoExp.png PSGeoImp.png PSGeoExp.png PSGeoAna.png PSGeoSym01.png PSGeoSym3.png PSGeoSym7.png NrgSym.png
	pdflatex $< 

# Generates plots
.PHONY : plots
plots : PosExp.png VelExp.png ErrPosExp.png ErrVelExp.png NrgExp.png hVsErrExp.png PosImp.png VelImp.png ErrPosImp.png ErrVelImp.png NrgImp.png PSGeoExp.png PSGeoImp.png PSGeoExp.png PSGeoAna.png PSGeoSym01.png PSGeoSym3.png PSGeoSym7.png NrgSym.png

PosExp.png : diffEqs.py
	python $< 1
VelExp.png : diffEqs.py 
	python $< 2
ErrPosExp.png : diffEqs.py
	python $< 3
ErrVelExp.png : diffEqs.py
	python $< 4
NrgExp.png : diffEqs.py
	python $< 5
hVsErrExp.png : diffEqs.py
	python $< 6
PosImp.png : diffEqs.py
	python $< 7
VelImp.png : diffEqs.py
	python $< 8
ErrPosImp.png : diffEqs.py
	python $< 9
ErrVelImp.png : diffEqs.py
	python $< 10
NrgImp.png : diffEqs.py
	python $< 11
PSGeoExp.png : diffEqs.py
	python $< 12
PSGeoImp.png : diffEqs.py
	python $< 13
PSGeoAna.png : diffEqs.py
	python $< 14
PSGeoSym01.png : diffEqs.py
	python $< 15
PSGeoSym3.png : diffEqs.py
	python $< 16
PSGeoSym7.png : diffEqs.py
	python $< 17
NrgSym.png : diffEqs.py
	python $< 18

# Deletes plots
.PHONY : clean
clean :
	rm -f *.png
	rm -f report.pdf
	rm -f report.log
	rm -f report.aux