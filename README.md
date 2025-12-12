

# LongEval 2025 Web Retrieval — Two-Stage BM25 and Neural Reranking

This repository contains the full implementation and experimental pipeline for a two-stage information retrieval system developed for the LongEval 2025 Web Retrieval challenge. The project investigates retrieval robustness under temporal drift by training ranking models on historical web snapshots and evaluating them on future snapshots, reflecting realistic deployment scenarios. The system combines a BM25 first-stage retriever with multiple supervised and neural reranking models, including gradient-boosted trees, classical learning-to-rank methods, and transformer-based CrossEncoders, evaluated across six cross-temporal train–test splits.

The codebase includes indexing scripts for heterogeneous document formats (TREC and JSON), query and relevance judgment handling, feature extraction, reranking pipelines, and evaluation routines built on PyTerrier, PyTorch, and related machine learning libraries. Particular attention is given to document and query identifier consistency across temporal splits, as well as to the engineering challenges introduced by independent indexing of training and test collections. The repository is intended to support reproducibility, analysis, and further experimentation with robust multi-stage retrieval systems under temporal distribution shift.

### LongEval 2025 dataset: https://researchdata.tuwien.ac.at/records/th5h0-g5f51
