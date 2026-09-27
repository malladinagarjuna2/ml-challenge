# Iteration log — Business Entity Resolution

Every iteration: what changed, the measured result, and every mistake found (with its fix).
Scores are out-of-fold macro F0.5 on training data unless stated otherwise.

---

## Run 01: first full submission (2026-09-26 → 27)

**Setup:** random 200k train S1 vs all 10.3M S2/S3 · state-partitioned TF-IDF blocking (4 views + 2 exact keys, RRF, top 40) ·
82 + 13 features · 2-stage XGBoost (GPU, 5 folds) · expected-F0.5 + one-to-one decision.

**Result:** validation F0.5 **0.9649** (US 0.972, India 0.954) · precision 0.990 · recall 0.923 ·
blocking recall (top 40) 0.976 · perfect-model ceiling on the same shortlist 0.991 · test file: validator PASS.

**Loss breakdown:** model/decision 0.0264 · blocking 0.0088.

**Why recall is low (measured):** 12.9% of true pairs share no name word. The largest group is Indic-script names spelled
out in Latin (`intrneshnl` = international, `tek` = tech); then generator-made names (`Mira…`, `Zeph…`, `Brix…`) linked only by
address; websites as names; 4.3% of true pairs have an empty address. The best F0.5 cut-off is about F/1.25 ≈ 0.77, so every
uncertain true match is dropped.

### Mistakes in run 01
| # | Mistake | Effect | Fix |
|---|---|---|---|
| M1 | Preprocessing: abbreviations expanded before states were read (FL/CT/MT/NE lost), short words taken as states, city names as states, Saint/St inconsistent | wrong/missing states | fixed before run 01; 29 unit checks |
| M2 | Blocking with 2-grams and no partitions | 12–50 h estimate | country+state partitions, pruned n-grams, RRF |
| M3 | DEV mode used 10% of decoys; I first said 0.98 was inflated, then agreed it wasn't | confusing message | the real number (0.965) settled it; always validate with all decoys |
| M4 | Trained on a **random 200k** sample | owners of many candidates missing in training → over-cautious model (hypothesis) | run 02: train on all 2.2M |
| M5 | 35 min of CPU training before switching to GPU | wasted time | GPU from the start |
| M6 | Python loops on one core (features, stage-2 support, decision) | ~2 h of the 6 h run | run 02: vectorised / parallel |
| M7 | France not partitioned | 60 min for 15% of entities | run 02: France split by region |
| M8 | Models kept only in kernel memory | lost when the kernel locked | run 02: every stage saved to `runs/run02_full/` |
| M9 | `np.isin` on text-ID object arrays (200k × 8M comparisons) | kernel locked, could not interrupt | use hash lookups (`pd.Index.isin`), never `np.isin` on objects |

---

## Run 02: all three layers fixed, full training data (in progress)

Planned changes:
- **Layer 3:** train on all 2.2M train S1 (no sampling) → every candidate's real owner is present, as in test.
- **Layer 2a:** transliteration table learned from training pairs (`intrneshnl → international`), applied to all names.
- **Layer 2b:** generated-name features (words never seen in real S1 names; character-pattern score).
- **Layer 2c/d:** website / no-space containment; name-uniqueness counts (how many S1 share the name).
- **Layer 1:** decision rule re-tuned on the new out-of-fold predictions.
- **Mechanics:** stage outputs saved to disk; vectorised features; parallel loops; France by region.

### Mistakes found while building run 02
| # | Mistake | Effect | Fix |
|---|---|---|---|
| M10 | Regex `r"[ऀ-෿]"` passed to pandas 3 `.str.contains` (Arrow RE2 does not support `\u` escapes) | DEV cell 2 crashed | write the pattern with the real characters |
| M11 | Generated-name model learned from website names too (`earnosethroat.com` → glued words counted as "generated") | real words (`logistics` 0.71) scored like fake ones (`brixumbra` 0.72) | skip website names and tokens > 14 chars when learning |
| M12 | DEV India states (Goa, Himachal) have only 12 S1 records | India untested in DEV | use a large Indian state for future DEV checks; India measured in the full run |

### Run 02 DEV check (US states RI/VT/DE, 14.8k train S1; France PdL + small US/IN test sets) — 2026-09-27
Code verified end to end (stage outputs saved per chunk). Transliteration: Indic true pairs sharing a name word 28.9% → 99.4%
(in-sample). France: 97% of S2/S3 records get a region (3 partitions). Generated-name score separates fake (1.4–2.4) from real (≤0.22).
DEV validation (tiny training set, not comparable): F0.5 0.9727 · precision 0.9935 · recall 0.9432 · blocking recall 0.9907 · oracle 0.9957.
| M13 | Output writer indexed pandas-3 Arrow string arrays row by row (`ids_r[list]` per S1) | 20 min for 75k DEV rows → would be 7+ h for the full test set | convert ID columns to numpy object arrays once |
| M12 | Output writer indexed pandas-3 Arrow string arrays (`.values`) row by row | 13 min for 75k DEV rows → ~5 h for the full test | convert IDs to NumPy once (`to_numpy(dtype=object)`) |
| M13 | State detection scanned components from the end and accepted a 2-letter code at the end of a later component before a full state name at the start (`Delhi, …, Turkman Ga, Te, …` → "ga" = Goa) | S1 put in the wrong state partition → blocking can never reach its matches (DEV India F0.5 = 0; also in run 01) | two passes: full state name as a whole component first (anywhere), codes only as fallback |
| M14 | My first fix for M13 ("full state name first, anywhere") was too broad: city names like `Washington, DC` and mentions of neighbouring states were read as the state | measured: state agreement 0.9988 → 0.9974, unreachable S1 873 → 1,153 (worse) | reverted from the untouched run-01 cache; replaced by a narrow rule, kept only if it measures better |

**Result of the M13/M14 work:** narrow rule (override only when the detected state is not backed by any whole address component)
measured on all 7.6M true pairs: S1 cut off from all their matches **873 → 803** (0.042% → 0.039%), agreement 0.9988 → 0.9989 → kept.
Lesson: the DEV India score of 0 came from a tiny, unlucky sample (Goa/Himachal partitions are dominated by misassigned records);
the bug is real but small. Always size a bug on the full data before fixing it.

### Run 02 DEV check (small complete US states + Goa/Himachal, France Pays de la Loire)
- Transliteration table: 503 mappings; Indic-script true pairs sharing a name word **0.289 → 0.994** (in-sample, somewhat optimistic).
- Generated-name model (after M11): fake names score 1.4–2.4, real words ≤ 0.22.
- France: 97% of S2/S3 records get a region; 72,734 French S1 blocked in 110 s (run 01: 60 min for 259k).
- DEV validation (15k S1, weak because each fold trains on ~10k): F0.5 0.9727, precision 0.9935, recall 0.9432, blocking recall 0.9907.
- Pipeline resumes from saved stages in seconds; output writing 4 s (was 25+ min, M12).
