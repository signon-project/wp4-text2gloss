# Format source code automatically
style:
	black --line-length 119 --target-version py38 src/amr_to_sl_reprr
	isort src/mbart_amamr_to_sl_repr

# Control quality
quality:
	black --check --line-length 119 --target-version py38 src/amr_to_sl_reprr
	isort --check-only src/amr_to_sl_reprr
	flake8 src/amr_to_sl_repr --exclude __pycache__,__init__.py
