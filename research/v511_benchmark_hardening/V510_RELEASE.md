# V510 release manifest

Base: V509 cognitive bridge benchmark final

Purpose: benchmark hardening + reference normalization before teacher-trace/distillation work.

Changes:
- Fix QueryTarget object/dict mismatch in the bridge evidence path.
- Normalize yes/no color, size, and shape questions into typed property targets.
- Preserve the V509 36-case regression suite.
- Add deterministic V510 adversarial semantic-boundary suite.
- Add V510 architecture hardening tests.

Recommended command from the Graph-Topology repository root:

```powershell
python .\research\v510_benchmark_hardening\run_v510.py --memory ".\results\full_semantic_memory.sqlite" --freeze-knowledge --verbose --output ".\research\v510_benchmark_hardening\v510_benchmark_results.json"
```
