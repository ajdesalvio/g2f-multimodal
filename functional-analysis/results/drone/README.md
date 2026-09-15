# Flight metadata and imagery access

- [Drone_Data_Description_V6.xlsx](Drone_Data_Description_V6.xlsx): unchanged final metadata, cameras, and exported raster GSD.
- [D2S_STAC_Links.xlsx](D2S_STAC_Links.xlsx): 16 Data to Science collection links and associated Figshare DOIs, covering the 19 study environments.

Browse the [clickable collection and DOI table](../../../README.md#geospatial-data) in the repository README. All study geospatial data are deposited in Data to Science.

The metadata workbook has 400 rows but **356 unique environment/DAP combinations**. These match the retained flight-date conversion table. MOH1 C5a/C5b field entries account for the repeated keys; combined Texas imagery collections account for fewer collections than environments.

GSD describes the exported orthomosaic pixel spacing. It is not a recovered processing report or a measure of georeferencing accuracy. Use the [raster manifest example](../../data/orthomosaic_file_manifest.example.csv) with [extract_orthomosaic_gsd.R](../../scripts/07_results_figures/drone/extract_orthomosaic_gsd.R) to inspect local GeoTIFFs.

Workbook contents were preserved byte-for-byte. On 14 September 2026, all 16 STAC URLs returned HTTP 200 and all 16 DOIs redirected to the expected Figshare records. Figshare returned HTTP 202, so record contents were not verified automatically. Individual assets and their license terms were not inspected by this automated check.
