.PHONY: data run pca test lint fix clean all

all: data run pca

data:            ## regenerate the on-disk fixtures (~5 s)
	python -m proteomics_revertant.make_data --outdir data --hdf5

run:             ## full analysis, one results set per dataset (~110 s)
	python -m proteomics_revertant.run --data data --outdir results

pca:             ## genotype-coloured PCA figures and loading tables (~5 s)
	python -m proteomics_revertant.pca --data data --outdir results

test:            ## 34 self-contained checks (~4 min)
	python -m proteomics_revertant.tests

lint:
	ruff check .

fix:
	ruff check . --fix

clean:
	rm -rf results __pycache__ proteomics_revertant/__pycache__ .ruff_cache
