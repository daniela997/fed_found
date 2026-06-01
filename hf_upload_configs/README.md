# HuggingFace upload configs

One YAML per dataset you want to push to HuggingFace via `upload_to_hf.py`.

## Quick start

1. Copy `example_ifcb.yaml` to `<your_dataset_name>.yaml`.
2. Fill in `data_dir`, `hf_dataset_name`, `hf_org_name`, and the metadata
   fields (license, citation, etc.).
3. Set `HF_TOKEN` in your shell:
   ```bash
   export HF_TOKEN=hf_xxx...
   ```
   (Or put it in the YAML, but don't commit that.)
4. Dry-run to check stats and the rendered card without pushing:
   ```bash
   python upload_to_hf.py hf_upload_configs/<your_dataset_name>.yaml --dry-run
   ```
5. Push for real:
   ```bash
   python upload_to_hf.py hf_upload_configs/<your_dataset_name>.yaml
   ```

## Data layout

The script expects ImageFolder format on disk:

```
<data_dir>/
  class_A/
    img001.jpg
    img002.jpg
    ...
  class_B/
    ...
```

That's it — no nested splits, no extra metadata files. The script produces a
single `train` split with `(image, label as ClassLabel[N])` schema. This
matches what `load_dataset("imagefolder")` produces, which is the same
convention used by `project-oceania/whoi-plankton`, `syke_ifcb_2022`,
`flowcamnet`, etc.

If your raw data is laid out differently (e.g. `<class>/images/*.jpg` with a
nested `images/` subdir, or a flat directory with a TSV of labels), reorganize
it on disk first — symlinks work fine. See `prepare_tiny_imagenet.py` for a
similar reorganization helper.

## What gets uploaded

- The image dataset itself, pushed as Parquet via `Dataset.push_to_hub`.
- A dataset card (`README.md`) populated with: pretty name, description,
  license, source URL, citation (APA + BibTeX), per-channel mean/std (computed
  on a 2000-image sample), and a label-distribution table.
- The HF dataset card frontmatter exposes `task_categories`, `task_ids`,
  `annotations_creators`, `language`, `paperswithcode_id`, `arxiv_id`, and a
  free-form `tags` list — same fields project-oceania datasets use.

## Adding taxonomic enrichment later

This script produces a "Convention A" dataset (image + label only) — the same
schema as the per-instrument project-oceania datasets. If you later want
Kingdom..Species, environmental metadata, or a `proposed_label` harmonized
across datasets, that's a follow-up enrichment step that joins against a
WoRMS lookup. See `CROSS_INSTRUMENT_EXPERIMENT_DESIGN.md` for context.
