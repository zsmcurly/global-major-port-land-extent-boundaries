# Port land-extent mapping artifact

This repository is an anonymized review artifact for a 2017–2025 port land-extent mapping study. It is intended to expose the core method logic and the complete derived boundary dataset, not to provide a turnkey reproduction environment.

No author names, affiliations, email addresses, personal repository links, or local machine paths are included.

## Repository contents

```text
.
|-- data/
|   |-- port_land_extent_2017_2025_v1.0.gpkg
|   |-- port_annual_expansion_2017_2025.csv
|   `-- regional_statistics_2017_2025.csv
|-- scripts/
|   |-- modeltraining.py
|   `-- hstgo.py
|-- LICENSE
`-- DATA_LICENSE.md
```

## Core method code

`scripts/modeltraining.py` implements the port-aware multi-task segmentation model and its training procedure. The model consumes 12-channel Sentinel-1/2 feature arrays ordered as B2, B3, B4, B8, B11, B12, VV, VH, VV/VH, NDVI, MNDWI, and NDBI.

`scripts/hstgo.py` implements annual inference, spatial refinement, and hierarchical spatiotemporal graph-cut optimization (H-STGO). Its expected raster filename pattern is:

```text
CLUSTER_<cluster_id>_cluster_<cluster_uid>_<year>_feat12_u16.tif
```

Training patches, annual source raster stacks, the pretrained checkpoint, and restricted original AIS records are outside the scope of this review artifact. Sentinel-1/2 imagery is available from public archives. The scripts are supplied to document the implemented logic and parameters.

## Data products

- `port_land_extent_2017_2025_v1.0.gpkg`: complete annual port-boundary dataset with 900 MultiPolygon features (100 ports by nine years), EPSG:4326.
- `port_annual_expansion_2017_2025.csv`: port-year area and expansion statistics, 900 records.
- `regional_statistics_2017_2025.csv`: regional and global summary statistics, 10 records.

The complete GeoPackage is approximately 87.1 MiB. GitHub can store this file in a regular Git repository, although it will issue a large-file warning. The anonymous review service used for the submission may not render or serve files larger than its own per-file limit.

## Licenses

Python source code is released under the MIT License. The derived data products are released under CC BY 4.0; see `DATA_LICENSE.md`.
