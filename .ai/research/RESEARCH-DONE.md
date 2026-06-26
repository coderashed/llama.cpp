# Research Backlog — KVarN Paper

Source: https://arxiv.org/html/2606.03458v1
Title: KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks
Authors: Lorenz K. Muller, Philippe Bich, Chiara Boretti, Hyun-Min Chang, Jiawei Zhuang, Lukas Cavigelli (Huawei)
Reference impl: https://github.com/huawei-csl/KVarN

## Instructions

Each entry documents one part of the paper with implementation-relevant detail.
Pipeline: research.
Exit criterion: `kvarn/<NN>-<slug>.md` exists and describes the concept, math,
data flow, and concrete parameters needed for implementation, with references
to paper section numbers.

All items: Status done. See RESEARCH-DONE.md.

---

### 00 - Overview

- **Status**: done
- **Depends on**: nothing
- **Doc**: `kvarn/00-overview.md`

### 01 - Problem & Motivation

- **Status**: done
- **Depends on**: nothing
- **Doc**: `kvarn/01-problem-motivation.md`

### 02 - Error Decomposition (Magnitude vs Direction)

- **Status**: done
- **Depends on**: 01
- **Doc**: `kvarn/02-error-decomposition.md`

### 03 - Hadamard Transform Arrangement

- **Status**: done
- **Depends on**: 01
- **Doc**: `kvarn/03-hadamard-arrangement.md`

### 04 - Variance Normalization (Algorithm 1)

- **Status**: done
- **Depends on**: 03
- **Doc**: `kvarn/04-variance-normalization.md`

### 05 - KVarN Pipeline & Storage Format

- **Status**: done
- **Depends on**: 03, 04
- **Doc**: `kvarn/05-kvarn-pipeline.md`

### 06 - Three-Region Cache Layout

- **Status**: done
- **Depends on**: 05
- **Doc**: `kvarn/06-three-region-layout.md`

### 07 - Dequantization

- **Status**: done
- **Depends on**: 05
- **Doc**: `kvarn/07-dequantization.md`

### 08 - Pseudo-Decode Evaluation

- **Status**: done
- **Depends on**: 01
- **Doc**: `kvarn/08-pseudo-decode-eval.md`

### 09 - Experimental Setup

- **Status**: done
- **Depends on**: 05, 06
- **Doc**: `kvarn/09-experimental-setup.md`

### 10 - Results Summary

- **Status**: done
- **Depends on**: 09
- **Doc**: `kvarn/10-results-summary.md`

### 11 - Limitations

- **Status**: done
- **Depends on**: 00
- **Doc**: `kvarn/11-limitations.md`