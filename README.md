# RegClassifier

A deep-learning classifier that predicts the regulatory role of any 2 kb genomic window from DNA sequence and chromatin marks. The notebook (`Regclassifer.ipynb`) implements pretraining on five GI / endoderm epigenomes, fine-tuning on adult gastric tissue (Roadmap E094), and cross-cell-line transfer to GM12878 (B-lymphoblastoid) and K562 (myeloid).

## Overview of the model

Each 2 kb window is assigned to one of five mutually exclusive regulatory classes:

| Code | Class | Biology                      | TSS overlap | H3K4me3      | H3K4me1      | H3K27ac      | H3K27me3     | DNase        |
| ---- | ----- | ---------------------------- | ----------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| AT   | 3     | Active TSS / active promoter | required    | required     | excluded     | required     | excluded     | required     |
| PT   | 2     | Poised / bivalent TSS        | required    | required     | excluded     | excluded     | required     | any          |
| AE   | 1     | Active enhancer              | excluded    | any          | required     | required     | any          | required     |
| PE   | 0     | Poised / primed enhancer     | excluded    | any          | required     | excluded     | required     | any          |
| BG   | 4     | Background                   | —           | any / absent | any / absent | any / absent | any / absent | any / absent |

The architecture combines two modalities:

- **DNA-sequence feature** — character-level embedding of hg38 (`A/C/G/T/N`) → four 1-D conv blocks (`Conv1d → BatchNorm → ReLU → MaxPool`) with kernel sizes 19 / 11 / 7 / 5 → global average pool to a 256-dim feature vector. Warm-started from a multi-task pretrain (`SharedSeqMultiTask`) across five Roadmap EIDs so the conv filters encode universal motif statistics (TF-binding sites, CpG islands, promoter / enhancer sequence patterns) before fine-tuning sees any stomach data.
- **Chromatin-signal feature** — 1-D CNN on five binned fold-change tracks (H3K4me3, H3K4me1, H3K27ac, H3K27me3, DNase), 20 bins × 100 bp across the 2 kb window. Random-initialised; learns cell-type-specific mark configurations.

The two towers' GAP features are concatenated and passed through a 2-layer MLP head to produce five class logits. Training uses AdamW + warmup-cosine LR, AMP on CUDA, gradient clipping, sqrt-scaled class weights, and best-by-validation-macro-F1 checkpoint selection.

## Scientific question

**Can deep learning models trained on DNA sequence and histone modifications classify regulatory elements in primary human tissue?**

The question here is whether a fused sequence + chromatin CNN, trained end-to-end, can recover the same five-class structure directly from raw signal in primary human stomach tissue — a tissue context where consolidated reference data is sparser and class balance is harsher than in well-studied immortalised cell lines like K562 or GM12878.

## Hypothesis

**Each of the five regulatory states has a distinguishable combinatorial signature.** The model can distinguish *active from poised* forms of both enhancers and promoters, not just regulatory regions from background.

Concretely, this predicts that:

- The model achieves substantially-above-random per-class F1 on PE / AE / AT / BG (the well-populated classes), with AUROC > 0.9 indicating that the learned features rank true positives reliably above negatives.

- The model separates **active vs. poised** within each region type — AT from PT (active vs. bivalent promoter) and AE from PE (H3K27ac-positive vs. H3K27ac-negative enhancer) — rather than collapsing each pair into a single "promoter" or "enhancer" call.

- Cross-assay validation against FANTOM5 (CAGE-seq enhancer and TSS atlas — never seen during training) shows precision well above the baseline overlap rate, confirming that class calls reflect real regulatory biology, not artefacts of the chromatin rule set used to derive labels.

## Data

| Source | Identifier(s) | Role | Format |
|---|---|---|---|
| **GENCODE v47** | basic primary-assembly GTF | Protein-coding TSS annotation (AT / PT seeding, distal-enhancer exclusion) | GTF → BED |
| **hg38 reference genome** | UCSC primary assembly | DNA-sequence input (char-level tokenisation) | FASTA via `pyfaidx` |
| **Roadmap Epigenomics — consolidated** | E092, E079, E075, E085, E066, E094, E110, E111 (5 marks each) | Pretrain labels + signal (5 EIDs) and stomach fine-tune target (E094) | narrowPeak `.bed.gz` + fold-change `.bigWig` |
| **ENCODE — GM12878** | 5 marks (H3K4me3, H3K4me1, H3K27ac, H3K27me3, DNase) | Cross-cell-line transfer baseline (B-lymphoblastoid) | narrowPeak + bigWig |
| **ENCODE — K562** | 5 marks for peaks; 4 marks for bigwigs (H3K4me1 bigWig omitted) | Second cross-cell-line transfer (myeloid lineage) | narrowPeak + bigWig |
| **FANTOM5 — hg38** | enhancer atlas + CAGE TSS peaks | Independent cross-assay validation (never used during training) | BED |

### Roadmap EID assignments

| Role | EIDs | Tissue / source |
|---|---|---|
| Pretraining (5 EIDs × sequence-only multi-task) | E092, E079, E075, E085, E066 | Fetal stomach, esophagus, colonic mucosa, rectum mucosa, liver |
| Stomach fine-tune target | **E094** | Adult gastric — only stomach reference with all 5 marks consolidated |
| Auxiliary stomach references (labels reconstructed) | E110, E111 | Stomach mucosa, stomach smooth muscle |

E110 and E111 lack some marks in the consolidated release. The unconsolidated-reconstruction cell recovers their H3K27ac and DNase labels from BI's per-replicate narrowPeaks (with E092 fetal-stomach DNase used as a proxy where adult-tissue DNase does not exist in Roadmap). These reconstructed columns of `y_multi` are documented but not consumed downstream after multi-CT analysis was dropped from the final pipeline.

### Region pool and tensors

- **Shared region pool** (`shared_regions`, n ≈ 30,000): GENCODE-TSS-centred windows + union-H3K4me1-peak-centre windows across all 8 EIDs + chromosome-length-weighted random background.
- **`y_multi`**: shape `(n_regions, 8)`, `int64` — 5-class label per region per EID, derived by applying the chromatin-rule set against each EID's narrowPeak interval trees.
- **`X_seq`**: shape `(n_regions, 2000)`, `uint16` — tokenised hg38 sequence aligned to `shared_regions`.
- **`X_chrom`**: shape `(n_regions, 5, 20)`, `float32` — mean fold-change signal in twenty 100-bp bins for the 5 stomach (E094) bigWigs.
- **Splits**: 70 / 15 / 15 stratified train / val / test. Chromosomal-split benchmark uses 10 leave-out-chromosome folds with no chromosome leakage.

## Files

- `Regclassifer.ipynb` — main notebook (setup, pretrain, stomach fine-tune, evaluation, FANTOM5 validation, GM12878 transfer, K562 zero-shot + fine-tune + UMAP).
- `README.md` — this file.
