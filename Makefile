# Format source code automatically
style:
	black --line-length 119 --target-version py38 src/text2gloss
	isort src/text2gloss

# Control quality
quality:
	black --check --line-length 119 --target-version py38 src/text2gloss
	isort --check-only src/text2gloss
	flake8 src/text2gloss
