# BGLR one-predictor fix

The prediction workflow can retain only PTR FPC1. In BGLR's internal
`setLT.RKHS()` helper, subsetting a one-column eigenvector matrix can simplify
it to a vector. The [patch](BGLR-single-predictor-drop-false.patch) preserves
the matrix dimension:

```diff
-    LT$V = LT$V[, tmp]
+    LT$V = LT$V[, tmp, drop = FALSE]
```

All four retained CV/LOEO prediction entrypoints check this exact assignment
in the installed `setLT.RKHS()` helper before fitting models. An unrelated
`drop = FALSE` elsewhere in the package does not satisfy the check.

The final HPRC library archive contains BGLR **1.1.5**, built with **R 4.4.2**,
with this exact patch. See
[package provenance](verified-bglr-provenance.md) for checksums and verification scope.

## Install and verify

1. Load the TAMU R 4.4.2 module stack and activate `BGLR_Mod_4.4.2`.
2. Obtain the BGLR 1.1.5 source corresponding to the archived package metadata.
   Record the downloaded source checksum. The archive does not identify an
   upstream Git commit or contain the complete original source tree.
3. From the extracted BGLR source directory, apply the patch:

   ```sh
   git apply --check /path/to/repository/functional-analysis/patches/BGLR-single-predictor-drop-false.patch
   git apply /path/to/repository/functional-analysis/patches/BGLR-single-predictor-drop-false.patch
   R CMD INSTALL .
   ```

   The patch has been checked against the available unmodified 1.1.5 source,
   and the patched assignment matches the archived final package. If the
   check fails on another release, inspect its RKHS helper before editing;
   do not force an unrelated patch.
4. Restart R, confirm the assignment, and retain a software record:

   ```r
   packageVersion("BGLR")
   find.package("BGLR")
   body(getFromNamespace("setLT.RKHS", "BGLR"))
   sessionInfo()
   ```

5. Run one prediction smoke task before submitting arrays.

## Reusing the archived library

`BGLR_Mod_4.4.2.tar.gz` is the final Linux package library;
it is archived separately from GitHub. It contains four installed packages,
not the full R/module stack or every dependency, and its native libraries are
not portable to Windows. Use compatible HPRC modules or rebuild from source.
Do not substitute the older unpatched `BGLR_HPRC` copy. BGLR retains its
separate GPL-3 license.
