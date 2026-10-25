# Raw data

This directory contains the paper-supplied melting-point CSV files and OCHEM exports.

The two `OCHEM_free_melting_point_402369_*.csv.gz` files are gzip-compressed byte-preserving copies because the uncompressed CSV files exceed GitHub's 100 MB per-file limit. Decompress them to recover the original CSV bytes.

The `raw.csv` and `readable.csv` OCHEM files are two versions of the same export: `raw` preserves the website bytes; `readable` repairs CSV quoting/column widths while preserving source fields.