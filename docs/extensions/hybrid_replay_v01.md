# Hybrid saved-checkpoint replay

This route packages the completed E1 seed-11 hybrid pair and evaluates its saved selected checkpoints from an extracted evidence tree. It does not reproduce training. The original core study and archive remain separate.

Run the builder only after both 10,000-step fits and their original full evaluation have completed. It refuses incomplete runs, changed frozen source/configuration hashes, inconsistent checkpoint or label bindings, existing destinations, and evaluation payload hash mismatches. Actual execution status belongs in its saved build and replay receipts.

From the project root, use the existing locked environment:

```sh
rtk proxy uv run --frozen --offline python docs/evidence/phase_11/hybrid_full_v01/build_replay_snapshot.py \
  --manifest results/extensions/hybrid/p11_hybrid_full_v01.json \
  --evaluation-dir results/extensions/hybrid/p11_hybrid_evaluation_v01
```

The script exclusively creates these outputs:

- `results/extensions/hybrid/replay_v01/`, with the frozen Python source, referenced configurations, labels v02, full pair manifest, both completed full runs, original evaluation, environment files, this note, the builder, and the separate replay runner.
- `results/extensions/hybrid/hybrid_e1_seed11_v01.zip`.
- `results/extensions/hybrid/replay_extracted_v01/`, extracted from that archive.
- `results/extensions/hybrid/replay_v01_build_result.json`, with archive SHA256, CRC/member checks, extraction checks, file hashes, and executable replay commands.

It checks source hashes before and after copying, snapshot hashes before and during archiving, every ZIP member's CRC and SHA256, and the complete extracted file inventory. `snapshot_inventory.json` excludes its own bytes; the external build result binds its hash. Failed or partial outputs remain available for diagnosis. Choose new destinations for any later attempt.

After a successful build, open the extracted root and run the exact command in its `REPLAY_COMMANDS.json`. That command uses the existing Python interpreter directly, so it does not install packages or create another environment. It sets `PYTHONPATH` to the extracted `src`, suppresses the user site and bytecode writes, and invokes the unchanged `extension-hybrid-evaluate` CLI through `tools/replay_hybrid_saved_evaluation.py`.

The default new output is `results/extensions/hybrid/replay_evaluation_v01/` inside the extracted tree. The runner also saves `results/extensions/hybrid/replay_import_readback_v01.json`, containing its exit result and imported `hydro_pinn` module paths. Every imported project module must resolve inside the extracted `src`. Its exclusive `replay_execution_v01.launch.json` and `replay_execution_v01.json` receipts record the exact argv, PID, UTC start/end, exit, source hashes, import origins, and captured stdout/stderr file hashes. Git discovery stops before reaching the parent checkout; inherited Git metadata and filesystem access remain explicit limitations.

The evaluator verifies frozen source, configuration, label, protocol, checkpoint and partition bindings before generating the final reference. It uses saved selected model states and creates a new evaluation directory. Re-running requires a new output directory and import-report path. It never launches fitting.

After the runner succeeds, execute the separate comparison command in `REPLAY_COMMANDS.json`. The generated `tools/compare_hybrid_saved_evaluations.py` reads the bundled original evaluation and the new replay inside the extracted tree. It requires every saved NPZ array's keys, dtype, shape and values to match exactly. It also requires exact metrics, paired differences, shock diagnostics, conservation, post-fit validation and metric-definition sections, verifies their bindings, and confirms that the bundled original payload remains unchanged. The new `replay_comparison_v01.json` retains timing and metadata differences separately. The comparison performs no fitting or model evaluation.

The existing locked environment still supplies Python and third-party dependencies. Historical run metadata stays byte-identical and may contain original absolute paths. The extraction sits under the original repository, and that checkout remains OS-accessible; the runner limits Git discovery, but import-location readback does not prove process-wide filesystem read exclusion. Snapshot/ZIP checks establish copied-byte integrity. A successful separate evaluator and exact comparison establish saved-checkpoint replay on this host, without establishing a fresh training run, independent environment, cross-platform reproduction, field validity, human acceptance, or submission.
