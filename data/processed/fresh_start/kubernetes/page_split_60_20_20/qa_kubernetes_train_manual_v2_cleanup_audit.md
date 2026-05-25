# Kubernetes Train Cleanup Audit Log

This log records the train-only cleanup passes that removed low-value docs-meta, contribution-workflow, page-chrome, and landing-page/grouping QA while keeping fully cleared pages in `page_ids_train.txt` as uncovered pages.

## Cleanup Summary

- cleanup target: `qa_kubernetes_train_manual_v2.jsonl`
- rows removed in initial meta/workflow cleanup: `156`
- rows removed in page-chrome cleanup: `8`
- rows removed in later sequential and targeted cleanup passes: `52`
- total rows removed across train cleanup passes: `216`
- pages touched by cleanup: `70`
- pages fully cleared of QA but kept in train split: `53`
- policy: manual `case-by-case`
- retained covered contribute page: `contribute_localization`

## Files In This Audit Bundle

- `qa_kubernetes_train_manual_v2_cleanup_audit.md`: summary of cleanup passes
- `qa_kubernetes_train_manual_v2_cleanup_removed_pages.csv`: page-level cleanup ledger
- `qa_kubernetes_train_manual_v2_cleanup_removed_rows.tsv`: archive of removed QA row IDs, original IDs, and questions

## Latest Targeted Pass

- removed `7` low-value `page?` questions that were primarily about page grouping, landing-page navigation, or page positioning rather than Kubernetes behavior/semantics
- latest removed IDs: `kubernetes-train-manualv2-00260, kubernetes-train-manualv2-02071, kubernetes-train-manualv2-02957, kubernetes-train-manualv2-02977, kubernetes-train-manualv2-02980, kubernetes-train-manualv2-02981, kubernetes-train-manualv2-02983`

## Notes

- No repository files were deleted during these passes; the deleted units were QA rows inside the train JSONL.
- The current train corpus already reflects all cleanup passes recorded in the archive.
- After train-side renumbering of transferred eval/test rows, `7` colliding archived IDs were remapped to fresh train-style archive IDs; each row keeps its pre-remap value in `original_id`.
- The affected low-value pages remain in the train split and now count as uncovered pages when they have no remaining train QA.
