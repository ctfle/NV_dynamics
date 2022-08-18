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
	cd docs/html && open index.html

install:
	pip install .

github_update_master:
	git add QNV4py
	git add setup.py
	git add examples
	git commit -m "update of QNV4py and examples"
	git checkout master
	git push origin master

github_update_docu:
	git add Makefile
	git add docs
	git add doxygen-awesome.css
	git commit -m "update of QNV4py and examples"
	git checkout docu
	git push origin docu
	git checkout master
