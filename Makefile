documentation:
	rm -rf docs
	mkdir docs
	cd docs && doxygen -g
	cd docs && echo "PROJECT_NAME = Quantum Dynamics of Dipolar Coupled Nuclear Spins" >> Doxyfile
	cd docs && echo "INPUT = ../QNV4py/ " >> Doxyfile; echo "INPUT += ../examples/ " >> Doxyfile
	cd docs && echo "EXTRACT_ALL = YES" >> Doxyfile; echo "RECURSIVE = YES" >> Doxyfile
	cd docs && echo "GENERATE_LATEX = NO" >> Doxyfile
	cd docs && echo "GENERATE_TREEVIEW = YES" >> Doxyfile
	cd docs && echo "HTML_EXTRA_STYLESHEET  = ../doxygen-awesome.css" >> Doxyfile
	cd docs && doxygen
	cd docs && cp -a ./html/ .
	cd docs && rm -r html 

show docu:
	open https://ctfle.github.io/nv_dynamics/

install:
	pip install .

update_github:
	git add *
	git commit QNV4py examples setup.py -m "update of QNV4py and examples"
	git push origin master:master

	git add doxygen-awesome.css
	git commit doxygen-awesome.css Makefile docs -m "updates on documentation"
	git push origin docu:docu

