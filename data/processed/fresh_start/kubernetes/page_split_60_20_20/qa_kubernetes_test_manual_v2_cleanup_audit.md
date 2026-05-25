# Kubernetes Test Cleanup Audit Log

This log records test-only cleanup passes that removed low-value docs-meta, contribution/reference/setup overview, generated-reference meta, and landing-page/grouping QA while keeping fully cleared pages in `page_ids_test.txt` as uncovered pages.

## Cleanup Summary

- cleanup target: `qa_kubernetes_test_manual_v2.jsonl`
- rows removed in current test cleanup pass: `34`
- total rows removed across test cleanup passes: `34`
- pages touched by cleanup: `14`
- pages fully cleared of QA but kept in test split: `12`
- policy: manual `case-by-case`
- rewritten surviving page-centric factual questions: `25`

## Files In This Audit Bundle

- `qa_kubernetes_test_manual_v2_cleanup_audit.md`: summary of cleanup passes
- `qa_kubernetes_test_manual_v2_cleanup_removed_pages.csv`: page-level cleanup ledger
- `qa_kubernetes_test_manual_v2_cleanup_removed_rows.tsv`: archive of removed QA row IDs, original IDs, and questions
- `qa_kubernetes_test_manual_v2_page_coverage.csv`: page-level test coverage after cleanup

## Removed Full Pages In This Pass

- `contribute_docs`
- `contribute_generate-ref-docs_kubernetes-api`
- `contribute_generate-ref-docs_kubernetes-components`
- `contribute_generate-ref-docs_quickstart`
- `contribute_style_write-new-topic`
- `reference_command-line-tools-reference`
- `reference_networking`
- `reference_scheduling`
- `reference_setup-tools_kubeadm_generated`
- `setup_production-environment_tools_kubeadm`
- `tasks_tools`
- `tutorials_kubernetes-basics_update`

## Removed Single Rows In This Pass

- `kubernetes-test-manualv2-0234`
- `kubernetes-test-manualv2-0683`

## Notes

- No repository files were deleted during this pass; the deleted units were QA rows inside the test JSONL.
- The current test corpus already reflects all cleanup passes recorded in the archive.
- Fully cleared low-value pages remain in the test split and now count as uncovered pages.
