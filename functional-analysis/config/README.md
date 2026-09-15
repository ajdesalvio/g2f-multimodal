# Path configuration

Within `functional-analysis/config/`, copy `paths.example.yml` to `paths.yml` and edit the local values. `paths.yml` is ignored by Git so personal paths are never published. `G2F_PROJECT_DIR` refers to the `functional-analysis/` directory, not the outer Git repository.

The configuration order is:

1. environment variable;
2. value in `config/paths.yml`;
3. a project-relative default where appropriate.

Supported environment variables:

- `G2F_PROJECT_DIR`
- `G2F_CONFIG_FILE`
- `G2F_DATA_DIR`
- `G2F_RESULTS_DIR`
- `G2F_INCLUDE_ZE`

Example local configuration:

```yaml
project:
  data_dir: "D:/research/g2f-multimodal-data"
  results_dir: "D:/research/g2f-multimodal-results"
```

Example HPRC setup:

```bash
export G2F_PROJECT_DIR="/scratch/user/USER/g2f-multimodal/functional-analysis"
export G2F_DATA_DIR="/scratch/user/USER/g2f-multimodal-data"
export G2F_RESULTS_DIR="/scratch/user/USER/g2f-multimodal-results"
export G2F_INCLUDE_ZE=FALSE
```

HPRC launchers use environment variables for scheduler-visible paths and the
custom R environment; those site-specific settings are intentionally not read
from or stored in the committed YAML example.
