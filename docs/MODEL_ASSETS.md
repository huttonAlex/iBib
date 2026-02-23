# Model Assets

PointCam uses pinned model assets to keep deployments reproducible. The manifest
lives at `models/manifest.json` and includes SHA256 checksums and sizes.

## Verify Assets

```bash
python scripts/verify_model_assets.py
```

This reports missing files or checksum mismatches.

## Fetch Assets

```bash
python scripts/fetch_model_assets.py --base-url <MODEL_URL>
```

Set `POINTCAM_MODEL_BASE_URL` to avoid passing `--base-url` each time. The fetch
script downloads missing or mismatched files, verifies SHA256, then replaces the
local copies.

## PARSeq Tokenizer

PARSeq ONNX decoding requires a tokenizer JSON at:
`models/ocr_parseq.tokenizer.json`

Generate it when exporting PARSeq:

```bash
python scripts/export_ocr_model.py \
  --model parseq \
  --checkpoint runs/ocr_finetune/parseq_v1/best.pt \
  --tokenizer-out models/ocr_parseq.tokenizer.json
```

Include the tokenizer JSON in your model bundle so runtime inference does not
require `torch.hub` or network access.

If you only use CRNN, the tokenizer is optional. The verification script will
warn if it is missing.

## Updating the Manifest

If you update any model files, refresh `models/manifest.json` with new SHA256
hashes and sizes. The verification script expects the manifest to be accurate.
