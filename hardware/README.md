# Hardware descriptor dataset

This directory keeps raw node captures separate from normalized, model-ready
hardware descriptors.

## Generated files

- `hardware_descriptors.csv`: one row per device. Columns prefixed with `gpu_`,
  `pcie_`, `cpu_`, and `system_` are candidate `H_src`/`H_target` inputs.
- `hardware_descriptors.json`: the same values in a structured form, together
  with source-file provenance and explicit missing-field lists.
- `collection_observations.csv`: transient values observed when the hardware
  snapshot was collected. These values are not static hardware descriptors and
  must not be joined into `H_src`/`H_target` by default.

Regenerate all three files with:

```text
python analysis/build_hardware_descriptors.py
```

## Field policy

- Numeric values are normalized to units stated in the column name.
- Missing values remain empty in CSV and `null` in JSON. They are never filled
  with zero.
- `env_` fields document the software environment. Keep them for reproducibility;
  use them as model inputs only as an explicit experimental choice.
- `gpu_configured_power_limit_w` records the effective power limit at collection
  time. This can affect runtime and differs from `gpu_default_power_limit_w`.
- PCIe negotiated generation and width describe the actual host/device link,
  while device and host maximum generations describe their capabilities.
- CUDA core count, Tensor Core count, memory bus width, and theoretical memory
  bandwidth are currently unavailable in the supplied raw files. Their schema
  columns are retained but deliberately left missing.
- GPU utilization, temperature, power draw, current clocks, memory use, and
  performance state are collection-time observations. They belong in
  `collection_observations.csv`, not the static hardware feature table.

## Raw-data handling

Files below `hardware/<device>/raw/` are source evidence and are not modified by
the normalization script.
