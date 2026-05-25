# Kubernetes Eval Cleanup Audit Log

This log records eval-only cleanup passes that removed low-value docs-meta, contribution workflow/style-guide, generated-readme, and landing-page/grouping QA while keeping fully cleared pages in `page_ids_eval.txt` as uncovered pages.

## Cleanup Summary

- cleanup target: `qa_kubernetes_eval_manual_v2.jsonl`
- rows removed in current eval cleanup pass: `73`
- total rows removed across eval cleanup passes: `73`
- pages touched by cleanup: `24`
- pages fully cleared of QA but kept in eval split: `24`
- policy: manual `case-by-case`
- rewritten surviving page-centric factual questions: `7`

## Files In This Audit Bundle

- `qa_kubernetes_eval_manual_v2_cleanup_audit.md`: summary of cleanup passes
- `qa_kubernetes_eval_manual_v2_cleanup_removed_pages.csv`: page-level cleanup ledger
- `qa_kubernetes_eval_manual_v2_cleanup_removed_rows.tsv`: archive of removed QA row IDs, original IDs, and questions
- `qa_kubernetes_eval_manual_v2_page_coverage.csv`: page-level eval coverage after cleanup

## Removed Full Pages In This Pass

- `concepts_configuration`
- `concepts_storage`
- `contribute_advanced`
- `contribute_participate_issue-wrangler`
- `contribute_participate_pr-wranglers`
- `contribute_participate_roles-and-responsibilities`
- `contribute_style_content-guide`
- `contribute_style_style-guide`
- `contribute_suggesting-improvements`
- `home_supported-doc-versions`
- `reference`
- `reference_encodings`
- `reference_node_topics-on-dockershim-and-cri-compatible-runtimes`
- `reference_setup-tools`
- `reference_setup-tools_kubeadm_generated_readme`
- `tasks_administer-cluster_network-policy-provider`
- `tasks_configure-pod-container_assign-resources`
- `tasks_extend-kubernetes`
- `tasks_inject-data-application`
- `tasks_manage-daemon`
- `tasks_network`
- `tutorials_cluster-management`
- `tutorials_services`
- `tutorials_stateless-application`

## Notes

- No repository files were deleted during this pass; the deleted units were QA rows inside the eval JSONL.
- The current eval corpus already reflects all cleanup passes recorded in the archive.
- Fully cleared low-value pages remain in the eval split and now count as uncovered pages.
