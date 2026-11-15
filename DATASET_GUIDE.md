# Data needed to improve recognition

The project has separate models and data paths. New examples help only after
they are labeled, converted to the matching format, and used to retrain the
matching model.

## What to collect

For each target sign, collect repeated video clips from many different signers.
Include variation in signing speed, hand size, left/right handedness, camera
distance, background, and lighting. Keep the full hand and relevant movement in
frame. Avoid using adjacent frames or clips from one recording session across
training and evaluation splits.

Keep a metadata table with at least:

| Field | Meaning |
| --- | --- |
| `clip_id` | Unique clip name |
| `label` | Agreed sign label/gloss |
| `signer_id` | Stable anonymous signer identifier |
| `language_region` | Sign language and regional variety |
| `split` | Train, validation, or test; split by signer |
| `source` | Dataset/source and original clip identifier |
| `rights` | License or documented permission to use the clip |

Consent and usage rights should cover model training and any demonstrations
shown in the app. Do not include names or other personal identifiers in the
signer ID.

## Formats currently accepted

- **WLASL100:** a ZIP containing `nslt_100.json`,
  `wlasl_class_list.txt`, and videos under `videos/`. The loader uses the
  supplied train/validation/test subsets and only recognizes the 100 labels in
  that class list. A new ASL dataset or new labels need a loader/label mapping
  update; it is not automatically merged into WLASL.
- **ISL emergency:** the outer ZIP must contain a `cropped_data/` directory
  with `accident_cropped.zip`, `call_cropped.zip`, `doctor_cropped.zip`,
  `help_cropped.zip`, `hot_cropped.zip`, `lose_cropped.zip`,
  `pain_cropped.zip`, and `thief_cropped.zip`. Inner clip names must identify a
  signer in the form `..._NNN_clipid.avi` (MP4 is also accepted). The current
  vocabulary is fixed to those eight labels. New labels require updating the
  loader, profile vocabulary, and training configuration.
- **ASL alphabet:** uses a separate static-image training path. It currently
  does not model motion for J and Z; video clips and motion-aware changes are
  needed for those letters.

Run the matching training script after preparing data. Preserve signer IDs so
validation and test signers remain unseen during training. Never tune settings
against the test split. Report per-class recall as well as top-1 accuracy,
because overall accuracy can hide poorly represented signs.

## Current limits

The available word models recognize fixed vocabularies (100 WLASL labels or
eight ISL emergency signs); they do not recognize arbitrary signs or translate
sign-language grammar into sentences. A sentence composed from recognized
labels is a sequence of glosses. The saved held-out results are 63% for WLASL
(100 clips) and 91.25% for ISL (80 signer-independent clips); these small test
sets do not establish performance across the broader signing community.
