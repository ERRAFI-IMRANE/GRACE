# third_party/DSCT

`grace/models/facial/dsct.py` depends on the external DSCT (Decoupled
Subject/Context Transformer) codebase for its model definitions
(`models.build_model`) and dataset transforms
(`datasets.caer.make_face_transforms`). This code is not vendored into
this repository.

To use the facial stream, place your local DSCT repository here, such
that this directory contains its `models/` and `datasets/` packages
directly, e.g.:

```
third_party/DSCT/
├── models/
│   └── ...
├── datasets/
│   └── caer.py
└── ...
```

Then point `configs/facial_config.yaml` -> `paths.dsct_repo_path` at
this directory (already set to `third_party/DSCT` by default).

Install any additional dependencies required by your local DSCT
codebase separately - they are not included in this repository's
`requirements.txt`.
