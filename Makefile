make:
	echo "Welcome to Project 'prosit_timsTOF_2023_wrapper'"

upload_test_pypi:
	rm -rf dist || True
	python setup.py sdist
	twine -r testpypi dist/* 

upload_pypi:
	rm -rf dist || True
	python setup.py sdist
	twine upload dist/* 

ve_prosit_timsTOF_2023_wrapper:
	python3 -m venv ve_prosit_timsTOF_2023_wrapper
