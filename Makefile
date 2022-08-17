documentation:
	rm -rf docs
	mkdir docs
	cd docs && doxygen -g
	cd docs && echo "PROJECT_NAME = Quantum Dynamics of Dipolar Coupled Nuclear Spins" >> Doxyfile
	cd docs && echo "INPUT = ../QNV4py/ " >> Doxyfile; echo "INPUT += ../examples/ " >> Doxyfile
	cd docs && echo "EXTRACT_ALL = YES" >> Doxyfile; echo "RECURSIVE = YES" >> Doxyfile
	cd docs && echo "GENERATE_TREEVIEW = YES" >> Doxyfile
	cd docs && echo "HTML_EXTRA_STYLESHEET  = ../doxygen-awesome.css" >> Doxyfile
	cd docs && doxygen

show docu:
	cd docs/html && open index.html

install:
	pip install .

