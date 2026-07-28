# MapBiomas enrichment (post-inference)

Bakes land-cover / crop-type + area columns onto Trazo-detected field boundaries.
This is the step **between** raw field detection (this repo's inference output) and
downstream analysis (e.g. `wri/rural-land`). It produces the `mbmode*`, `mbcov_area_*`,
`mbvalid_area_*` columns that `rural-land` treats as already baked into the fields.

## Pipeline

```
raw detections (Stage E, GDB)
  -> gridcode == 1 (drop boundary polygons)
  -> 0.03 ha area filter (equal-area ESRI:102033)    [regen_layer.py]
  -> clean per-region/year gpkg
  -> MapBiomas zonal extract: area-weighted modal class per field
     + geography clip                                 [extract_mapbiomas_fieldboundaries_v3.Rmd]
  -> field boundaries with mbmode<YY> + coverage columns
```

## Files

- **`extract_mapbiomas_fieldboundaries_v3.Rmd`** — the extraction engine (R, `exactextractr`).
  Per field: area-weighted modal MapBiomas class (`mbmode<YY>`), sampled/valid coverage area.
  Fast-bbox overlap + geography clip + chunked, MapBiomas legend harmonized per country.
  Reads `gpkg_outputs_v2/<Region>_<Year>.gpkg`, writes `field boundaries/<Region>_<Year>.gpkg`.
- **`regen_layer.py`** — rebuilds a clean 0.03 ha, gridcode==1 gpkg from the source GDB.
  Uses `OGR_ORGANIZE_POLYGONS=ONLY_CCW` to avoid the O(n^2) multipart-polygon stall.
  Also the recovery path for any layer produced with a wrong area floor.

## Configure

Paths are constants at the top of each file (Windows absolute). Point them at your
GDB / gpkg source / field-boundary output before running.

## Notes

- Area floor is **0.03 ha**, not 1 ha. A batch was once produced at a 1 ha floor
  (dropping all sub-hectare fields, ~40% of features); `regen_layer.py` is the fix.
- Extraction filters `gridcode == 1` at read (drops boundary polygons) and clips to the
  admin geometry before zonal stats, so no discarded polygon is ever extracted.
