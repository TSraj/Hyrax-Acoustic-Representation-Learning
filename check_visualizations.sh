#!/bin/bash
# Quick check script to see visualization progress

echo "=== Visualization Progress Check ==="
echo ""

echo "Macaque - wav2vec2_base:"
echo "  Individual layers:"
ls outputs/figures/macaque/wav2vec2_base/individual_layers/*.png 2>/dev/null | wc -l | xargs echo "    Files:"
echo "  Comparison grids:"
ls outputs/figures/macaque/wav2vec2_base/comparison_grids/*.png 2>/dev/null | wc -l | xargs echo "    Files:"

echo ""
echo "Macaque - wav2vec2_xlsr:"
echo "  Individual layers:"
ls outputs/figures/macaque/wav2vec2_xlsr/individual_layers/*.png 2>/dev/null | wc -l | xargs echo "    Files:"
echo "  Comparison grids:"
ls outputs/figures/macaque/wav2vec2_xlsr/comparison_grids/*.png 2>/dev/null | wc -l | xargs echo "    Files:"

echo ""
echo "Zebra Finch - wav2vec2_base:"
echo "  Individual layers:"
ls outputs/figures/zebra_finch/wav2vec2_base/individual_layers/*.png 2>/dev/null | wc -l | xargs echo "    Files:"
echo "  Comparison grids:"
ls outputs/figures/zebra_finch/wav2vec2_base/comparison_grids/*.png 2>/dev/null | wc -l | xargs echo "    Files:"

echo ""
echo "Zebra Finch - wav2vec2_xlsr:"
echo "  Individual layers:"
ls outputs/figures/zebra_finch/wav2vec2_xlsr/individual_layers/*.png 2>/dev/null | wc -l | xargs echo "    Files:"
echo "  Comparison grids:"
ls outputs/figures/zebra_finch/wav2vec2_xlsr/comparison_grids/*.png 2>/dev/null | wc -l | xargs echo "    Files:"

echo ""
echo "=== Expected Counts ==="
echo "Individual layers per dataset/model:"
echo "  wav2vec2_base: 13 layers × 4 methods = 52 files"
echo "  wav2vec2_xlsr: 25 layers × 4 methods = 100 files"
echo "Comparison grids per dataset/model: 4 files (pca, lda, tsne, umap)"
