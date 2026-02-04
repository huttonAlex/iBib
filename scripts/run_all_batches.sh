#!/bin/bash
# Process all unlabeled image batches

set -e

cd /mnt/c/Users/alex/Projects/pointCam
source .venv/bin/activate

echo "=========================================="
echo "Processing Batch 2 (749 images)"
echo "=========================================="

python scripts/process_unlabeled.py \
  --source "/mnt/c/Users/alex/Downloads/drive-download-20260203T030055Z-3-001" \
           "/mnt/c/Users/alex/Downloads/drive-download-20260203T030055Z-3-002" \
           "/mnt/c/Users/alex/Downloads/drive-download-20260203T030055Z-3-003" \
           "/mnt/c/Users/alex/Downloads/drive-download-20260203T030055Z-3-004" \
           "/mnt/c/Users/alex/Downloads/drive-download-20260203T030055Z-3-005" \
  --output data/unlabeled_batch2 \
  --conf 0.5

echo ""
echo "=========================================="
echo "All batches complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Batch 1: data/unlabeled_batch1/"
echo "  Batch 2: data/unlabeled_batch2/"
echo ""
echo "Next steps:"
echo "  1. Review CSV files in each batch directory"
echo "  2. Verify/correct OCR predictions"
echo "  3. Run: python scripts/process_unlabeled.py --generate-training data/unlabeled_batch1"
echo "  4. Run: python scripts/process_unlabeled.py --generate-training data/unlabeled_batch2"
