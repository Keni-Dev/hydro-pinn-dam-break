# Viscosity full-run reproduction v01

The executable archive and one extracted positive-checkpoint replay completed successfully. The archive is `results/extensions/viscosity/viscosity_e1_seed11_v01.zip`, SHA256 `860f9f421a2cbba8b25ff91fbc957a5a9ff4e3059fdb299c1e844aa935f1e0cf`, 20,214,334 bytes. Its 274-file payload and immutable extracted copy passed byte/hash verification before replay.

Replay imported the verified copied source from a separate work tree, loaded selected positive step 4,289 from the completed 10,000-update checkpoint, and freshly queried 332,004 physical states plus 365 dual-operator points once. All 24 raw-array payloads matched their original keys, dtypes, shapes and values exactly. All eight canonical scientific report sections matched exactly; timing/path/manifest provenance and dual source-file SHA were excluded from that report comparison. The historical zero cache was reused unchanged, with no zero query, training, or new reference generation.

Package/copy/archive/extraction/replay work took 5.087408 s in total; the nested replay process took 2.926413 s. Existing locked runtime dependencies were reused. This is checkpoint-to-output and saved-diagnostic replay in that runtime, rather than a fresh dependency installation or an independent implementation of the equations.

The snapshot preserves the scientific findings as they stood before packaging; their packaging paragraph consequently says planned. This separate completion note and the external package terminal receipt record the observed outcome without changing snapshot bytes. The package receipt is `docs/evidence/phase_11/viscosity_v01/full_v01_package/terminal.json`, SHA256 `d3954d85e8e79e081c5f4573b6d6966901a410f2ccaaf61ac19d77f0bee8d817`.
