"""Corpus acquisition subsystem for the SD1 data thread (WO-2).

This package acquires the unclassified background and clutter corpus into an S3-compatible object
store under a content-addressed layout, records every object in acquisition manifests and the
corpus ledger with its license and truth condition, decimates working derivatives through recorded
lineage, correlates AIS into tier-one vessel-presence truth, and generates the corpus audit report
that closes SD1. Credentials are read from the environment only and never touch the repository.
"""

from __future__ import annotations
