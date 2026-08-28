# Data and CUAD

The deterministic examples generate small synthetic vendor SaaS agreements locally. Private contracts should not be committed.

Create a reproducible 20-contract CUAD subset from the official annotation archive:

~~~bash
.venv/bin/cuad-data ingest --download --output data/cuad-subset --limit 20 --seed 42
~~~

Or ingest a local CUAD v1 release containing `CUAD_v1.json`, `full_contract_txt`, and/or `full_contract_pdf`:

~~~bash
.venv/bin/cuad-data ingest --source /path/to/CUAD_v1 --output data/cuad-subset --limit 20
~~~

The generated manifest contains exact answer spans, source hashes, attribution, selection parameters, matched PDF paths, and materialized text paths. Generated CUAD data should remain outside source control unless deliberately versioned with its attribution and license.

See [the repository data-attribution notice](../DATA_ATTRIBUTION.md) before
downloading, publishing, or redistributing any CUAD-derived material.
