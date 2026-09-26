# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary
We normalise every name and address into comparable fields (transliteration of Indic scripts, country-specific
abbreviations, state/postcode/house-number extraction). We then generate candidates with country + state-partitioned
character/word TF-IDF nearest neighbours plus exact-key joins, fused by reciprocal rank. A two-stage gradient-boosted
classifier (XGBoost on GPU) scores 82 → 95 pair and "competition" features, and a decision layer tuned directly for macro
F0.5 picks, for each Source 1 entity, the candidate set with the highest expected F0.5 (possibly empty) under a
one-to-one constraint.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on the training data (2.2M S1, 10.3M S2+S3 records, 7.6M labelled links):
- **Match structure:** 5.6% of S1 entities are singletons; the rest have 3.46 matches on average (≈1.7 in S2, ≈1.8 in S3, max 11).
  **No S2/S3 record is linked to more than one S1 entity** (0 of 7.6M), which justifies a one-to-one constraint.
  About 26% of S2/S3 records match nothing (decoys). **No true link crosses countries** (0 of 7.6M), so all work is done per country.
- **Name noise:** digit-for-letter typos (`C0mpany`, `5ecure`), random accents, legal-form changes (Inc ↔ LP ↔ none),
  added descriptors ("Services", "Partners"), word order, junk prefixes (`--`, `>>`, `***`), aliases (dba / formerly / f/k/a / aka / t/a),
  websites as names, Indic scripts in 9–11% of names, and some links whose names are unrelated (only the address connects them).
- **Address noise:** St/Street/"Saint", reordered components, missing parts (3% empty or `NULL`), zero-padded or truncated
  house numbers, states as codes, full names or native script, landmarks ("Near SBI ATM"), Indian formats (`H.No`, `Plot No`, `5-513/4`).
- **Test shift:** the test set contains **France** (15% of test S1), which never appears in training, so no step may be US/India-only.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier + metric-aware set selection
**Core Innovation:** (1) normalisation whose rules are verified by unit tests and by separation statistics on labelled pairs,
including native-script state names **learned from the training labels**; (2) state-partitioned sparse top-k blocking with
reciprocal-rank fusion, which keeps 97.7% of true links in 40 candidates per entity; (3) a decision layer that directly maximises
expected macro F0.5 per entity (including the empty set, for singletons).

---

## 3. Candidate Generation (Blocking)
- **Normalised inputs:** core name (legal forms, honorifics and alias markers removed), street text, house numbers, state,
  postcode, phonetic keys.
- **Partitions:** country × state group (states agree in 99.3% of true pairs after normalisation; Telangana and Andhra Pradesh are
  merged because vendors still mix them). S2/S3 records with no state are compared against their whole country.
- **Views (sparse TF-IDF top-15 per source, `sparse_dot_topn`):** character 3–4-grams of the core name, of the street, and of a phonetic
  key of name+address, plus word 1–2-grams of name+address. Very frequent n-grams are pruned (`max_df`) for speed.
- **Exact keys:** (country, first house number, first street word) and (country, rarest name token); keys shared by >50 records are skipped.
- **Ranking and cap:** reciprocal-rank fusion across views; top **40** candidates per S1 entity.
- **Measured on 20k train S1 against all 10.3M S2/S3:** all views together find 98.3% of true links; the top-40 keeps **97.7%**
  (US 99.6%, India 96.5%). A 3–5-gram view was dropped: 58% of the runtime for 0.03% unique recall.
- **Candidate pairs generated (test):** 69,301,712 for 1,732,544 S1 entities (40 each): France 10.4M, India 32.4M, US 26.5M.
- **How true matches were protected:** every rule was measured on labelled pairs before use; state is a partition only with a
  no-state fallback, never a hard filter on disagreements that were shown to be vendor noise.

---

## 4. Matching Model

**Features (82 in stage 1):**
- Name: ratio, token-set/sort, partial ratio, Jaro-Winkler, Levenshtein, phonetic ratio, no-space ratio, IDF-weighted Jaccard,
  rarest shared/unshared token IDF, first-token match, legal-form agreement.
- Address: street ratio/token-set/partial, phonetic street ratio, IDF-weighted word Jaccard, house-number overlap/Jaccard/
  first-number agreement, postcode and state agreement, empty-field flags (empty text is treated as unknown, not as a perfect match).
- Blocking: per-view scores, number of views, RRF score, cosine similarity in every view.
- Competition: rank and gap of each candidate among the S1 entity's candidates and, in reverse, among all S1 entities competing
  for the same S2/S3 record.

**Stage 2 (13 extra features):** stage-1 out-of-fold probability, its rank and gap within the S1 group and across competing S1
entities, and "support" (similarity of a candidate to the other likely candidates of the same entity: duplicates cluster).

**Model type:** XGBoost (`hist`, CUDA), 5-fold GroupKFold by S1 entity, early stopping; stage 1 → stage 2.
**Threshold selection method:** decision rules searched on out-of-fold predictions with the exact macro F0.5 (including
singletons). Chosen rule: isotonic calibration → one-to-one (each S2/S3 record kept only for its best S1) → per-entity
Monte-Carlo expected-F0.5 maximisation over top-k prefixes, with k = 0 allowed.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold, 200k train S1 against all S2/S3): 0.9649**. US 0.972, India 0.954; singletons 0.944, non-singletons 0.966.
- Stage-1 OOF AUC 0.9995 / AP 0.9958; stage-2 AUC 0.9996 / AP 0.9962.
- **Common false negatives:** Indic-script names with very short addresses (e.g. `परफेक्ट फाइनेंस प्रा. लि.` + "UNIT NO. 9, MUMBAI"
  vs "Perfect Finance Pvt Ltd" + full address); unrelated brand names with partial addresses.
- **Test predictions:** 93.8% of test S1 entities receive at least one match (3.17 on average), in line with the training distribution (94.4% non-singletons, 3.46 matches); France, unseen in training, behaves the same (94.2%, 3.13).
- **Common false positives:** co-located businesses (same house number and street, different name) and near-namesakes in the same state; the reverse-competition and one-to-one steps target these.

---

## 6. Conclusion
Careful, measured normalisation plus recall-oriented, state-partitioned blocking and a metric-aware decision layer give 0.965 macro F0.5 out of fold with no external data. The main remaining losses are Indian records whose names are in native script and whose addresses are very short (a blocking recall ceiling of about 97.7%), and singletons; multilingual embeddings on the GPU for these records are the next planned step.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/er_pipeline.py` runs the whole pipeline end to end; `er_pipeline.ipynb` is the same code
with the outputs of the submitted run. See the `README.md` there for setup and run commands.

### B. Additional Results
Normalisation quality on labelled pairs (true vs random same-country pairs): name similarity 92.0 vs 30.8, address similarity
93.3 vs 34.8, house-number overlap 94.6% vs 3.4%, state agreement 99.3%.
