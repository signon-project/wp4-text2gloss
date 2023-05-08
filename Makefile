# Format source code automatically
style:
	black src/text2gloss
	isort src/text2gloss

# Control quality
quality:
	black --check src/text2gloss
	isort --check-only src/text2gloss
	flake8 src/text2gloss
