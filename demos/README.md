# Demo Papers

This directory contains demo review results for papers from different research domains. The selected papers are grouped into three directions so that the review outputs can be compared across text/language modeling, image/video understanding, and graph/knowledge graph research.

## Research Directions

### Text / Language Models

- `bert`: BERT
- `Prefix-Tuning`: Prefix-Tuning

### Image / Video

- `beit`: BEiT
- `fixmatch`: FixMatch
- `lrcn`: LRCN
- `uda`: UDA

### Graph / Knowledge Graph

- `graphormer`: Graphormer
- `compgcn`: CompGCN
- `sacn`: SACN

## File Guide

Each paper folder contains the source paper and the generated review artifacts.

- `paper.pdf`: the original paper used as the input for the review.
- `report.pdf`: the generated review report in PDF format.
- `teaser_figure.png`: the generated teaser figure summarizing the paper or review output.
- `execution/`: intermediate execution files or runtime artifacts produced during the review process, when available.

Not every demo folder contains every file type. The common files are `paper.pdf`, `report.pdf`, and `teaser_figure.png`; additional files are included when that demo has extra outputs or references.
