# Benchmark Results

This directory contains benchmark results generated when running benchmarks locally.

## File Format

Benchmark results are saved as JSON files with the naming pattern:
```
{hostname}_{benchmark_name}_{size}_{timestamp}.json
```

## CI/CD Integration

When a PR is opened or updated:
1. The CI workflow converts all JSON files in this directory to `gh-pages/benchmarks/data/latest.json`
2. The Jekyll site is built with the aggregated results
3. A preview is deployed to `https://mathiasbourgoin.github.io/Sarek/preview/pr-{number}/benchmarks/`

## Important Note

**Individual benchmark result files should NOT be committed to git.**

The `.gitignore` file excludes `*.json` and `*.csv` in this directory to prevent accumulation.
The CI system processes results at build time from local benchmark runs.

## Running Benchmarks

To generate results:
```bash
opam exec -- dune exec benchmarks/bench_vector_add.exe
opam exec -- dune exec benchmarks/bench_matrix_mul_naive.exe
# ... etc
```

Results will be saved here automatically.
