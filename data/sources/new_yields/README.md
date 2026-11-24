# New yield rows (DataOps)

One CSV per new book, written by `src/dataops/read_books.py --accept data/sources/new_yields`
in the monthly "Fetch source data" workflow. A book gets a file here only if every check passed
(all totals add up, every district name known). The file arrives in a pull request; a person
opens that book's Excel (in the workflow run's artifact), checks it and merges.

`src/build_yield_labels.py` adds these rows to the workbook's rows. The workbook always wins
for a district, crop and year it already has; between new books, Punjab CRS wins for Punjab and
MNFSR elsewhere, then the newer book.

Columns: the same as the "All Data" sheet of `data/raw/Pakistan_Master_Yield_1981-2025.xlsx`.
