# Kubernetes Manual QA Audit v2

This audit summarizes the manual-only Kubernetes train/eval/test QA files after manual_wave2_light_70_15_15, a final small manual eval/test third-pass swap, train cleanup passes, the first eval cleanup pass, and the first test cleanup pass. Existing manual QA rows were preserved at page level during split moves; later cleanup passes remove low-value docs-meta, page-chrome, landing-page/overview questions, and occasional tautological low-signal pairs.

## TRAIN

- rows: 3614
- covered pages: 578/632
- remaining pages: 54
- duplicate questions: 0
- page kinds: {'api_schema': 196, 'concept': 2071, 'concept_doc': 68, 'generated_cli': 307, 'guide': 45, 'reference': 635, 'task': 226, 'tutorial': 66}
- section mix: {'concepts': 1201, 'contribute': 16, 'reference': 1415, 'setup': 90, 'tasks': 743, 'tutorials': 149}
- top pages by QA count:
  - concepts_workloads_controllers_job: 42
  - concepts_workloads_pods_pod-lifecycle: 36
  - reference_using-api_server-side-apply: 36
  - concepts_storage_volumes: 35
  - concepts_workloads_controllers_statefulset: 32
  - concepts_services-networking_service: 30
  - concepts_workloads_controllers_deployment: 30
  - concepts_workloads_pods: 27
  - reference_kubernetes-api_workload-resources_pod-v1: 25
  - concepts_scheduling-eviction_dynamic-resource-allocation: 24
  - concepts_workloads_controllers_cron-jobs: 23
  - tasks_administer-cluster_kubeadm_kubeadm-certs: 23
  - tasks_extend-kubernetes_custom-resources_custom-resource-definitions: 23
  - concepts_containers_images: 22
  - concepts_extend-kubernetes: 22

## EVAL

- rows: 745
- covered pages: 131/155
- remaining pages: 24
- duplicate questions: 0
- page kinds: {'concept': 264, 'guide': 10, 'reference': 273, 'setup': 5, 'task': 169, 'tutorial': 24}
- section mix: {'concepts': 264, 'reference': 273, 'setup': 15, 'tasks': 169, 'tutorials': 24}
- top pages by QA count:
  - concepts_security_multi-tenancy: 19
  - concepts_architecture_controller: 14
  - concepts_configuration_configmap: 14
  - concepts_security_application-security-checklist: 13
  - concepts_workloads_controllers_daemonset: 12
  - tutorials_security_seccomp: 12
  - concepts_configuration_secret: 11
  - concepts_storage_ephemeral-storage: 11
  - concepts_workloads_pods_init-containers: 11
  - reference_access-authn-authz_rbac: 11
  - reference_labels-annotations-taints_audit-annotations: 11
  - tasks_administer-cluster_ip-masq-agent: 11
  - tasks_configure-pod-container_configure-service-account: 11
  - concepts_cluster-administration_node-shutdown: 10
  - concepts_cluster-administration_system-metrics: 10

## TEST

- rows: 785
- covered pages: 153/165
- remaining pages: 12
- duplicate questions: 0
- page kinds: {'concept': 243, 'reference': 245, 'setup': 23, 'task': 233, 'tutorial': 41}
- section mix: {'concepts': 243, 'reference': 245, 'setup': 23, 'tasks': 233, 'tutorials': 41}
- top pages by QA count:
  - tasks_debug_debug-application_debug-running-pod: 25
  - concepts_architecture_cloud-controller: 15
  - concepts_cluster-administration_swap-memory-management: 14
  - concepts_cluster-administration_proxies: 13
  - concepts_windows_intro: 13
  - setup_production-environment_container-runtimes: 13
  - concepts_architecture_nodes: 12
  - concepts_extend-kubernetes_api-extension_custom-resources: 12
  - tasks_administer-cluster_network-policy-provider_cilium-network-policy: 12
  - tutorials_stateless-application_guestbook: 12
  - concepts_cluster-administration_node-autoscaling: 11
  - concepts_scheduling-eviction_assign-pod-node: 11
  - tasks_configure-pod-container_configure-pod-configmap: 11
  - tasks_manage-kubernetes-objects_update-api-object-kubectl-patch: 11
  - concepts_scheduling-eviction_scheduler-perf-tuning: 10

## Manual move notes

- Existing manual QA rows were moved together with their source pages during split rebalancing; no row-level pruning was performed in those split-move passes.
- Final small `eval/test` swap pages: eval->test `concepts_services-networking_endpoint-slices`, `concepts_cluster-administration_compatibility-version`; test->eval `reference_config-api_apiserver-config.v1beta1`, `reference_kubernetes-api_authentication-resources_certificate-signing-request-v1`.

## Eval Cleanup Notes

- total removed rows archived in `qa_kubernetes_eval_manual_v2_cleanup_removed_rows.tsv`: `73`
- total pages touched in `qa_kubernetes_eval_manual_v2_cleanup_removed_pages.csv`: `24`
- cleanup policy: manual, case-by-case removal of docs-meta, contribution workflow/style-guide pages, generated/readme meta, and landing/overview/navigation questions that do not teach Kubernetes behavior or semantics.
- current eval/cleanup archive consistency check: no archived removed ID remains in `qa_kubernetes_eval_manual_v2.jsonl`.
- rewritten surviving page-centric factual questions: `7`

## Test Cleanup Notes

- total removed rows archived in `qa_kubernetes_test_manual_v2_cleanup_removed_rows.tsv`: `34`
- total pages touched in `qa_kubernetes_test_manual_v2_cleanup_removed_pages.csv`: `14`
- cleanup policy: manual, case-by-case removal of docs-meta, contribution/reference/setup overview pages, generated reference meta, and landing/overview/navigation questions that do not teach Kubernetes behavior or semantics.
- current test/cleanup archive consistency check: no archived removed ID remains in `qa_kubernetes_test_manual_v2.jsonl`.
- rewritten surviving page-centric factual questions: `25`

## Train Cleanup Notes

- total removed rows archived in `qa_kubernetes_train_manual_v2_cleanup_removed_rows.tsv`: `216`
- total pages touched in `qa_kubernetes_train_manual_v2_cleanup_removed_pages.csv`: `70`
- cleanup policy: manual, case-by-case removal of docs-meta, page-chrome, landing-page, navigation/grouping, and clearly low-signal tautological questions that do not teach Kubernetes behavior or semantics.
- current train/cleanup archive consistency check: no archived removed ID remains in `qa_kubernetes_train_manual_v2.jsonl`.
- retained `contribute_localization` because it was not part of the earlier explicit exclusion set.
