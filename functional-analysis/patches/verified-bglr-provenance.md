# Verified final BGLR package

On 14 September 2026, the author identified `BGLR_Mod_4.4.2.tar.gz` as the
package library used for the final HPRC prediction jobs. Inspection of that
archive independently confirmed the package metadata and one-predictor fix.
The connection to the final jobs is the author's attestation, not a claim
that historical job logs were independently matched to the archive.

| Archived package | Version |
| --- | --- |
| BGLR | 1.1.5 |
| pacman | 0.5.1 |
| fastmatrix | 0.6-6 |
| fdapace | 0.6.0 |

BGLR's `DESCRIPTION` and installed-package metadata agree on **R 4.4.2**,
**x86_64-pc-linux-gnu**, and build time **2026-03-19 15:20:01 UTC**. Its
serialized `setLT.RKHS()` helper contains:

```r
LT$V = LT$V[, tmp, drop = FALSE]
```

All four retained CV/LOEO entrypoint guards passed against that archived
helper. Inspection did not install the package, load archived native
libraries, or run prediction models. The [distributed patch](BGLR-single-predictor-drop-false.patch)
also passes an application check against the available unmodified 1.1.5
source; this does not establish that every source file is otherwise identical.

## Archive identity and limits

Archive size: **7,651,626 bytes**. SHA-256:

```text
9400050FFB5B44926EEAA15F152DAE923F121307DFE7C60FA3FE78162B54FEB8
```

For additional verification, the archived `BGLR/R/BGLR.rdb` SHA-256 is:

```text
D8BC4D73B5AE30F0C6A3483BE8F856249ABDEAC8E94DF9A7BB4C83DD4010F4AF
```

This is an installed Linux package-library overlay, not a complete container
or R installation. It does not include every dependency, the original full
source tree, an upstream Git commit, or a saved final-job `sessionInfo()`.
The binary archive is kept outside GitHub; see the [data availability guide](../docs/DATA_AVAILABILITY.md)
for release locations. BGLR remains GPL-3 licensed; the repository's own
license does not replace third-party package licenses.
