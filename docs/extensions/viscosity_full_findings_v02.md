# Viscosity full findings v02: current status

26 October 2026 · Asia/Manila

Packaging and the single extracted positive-checkpoint replay completed successfully. The pending wording in the immutable [full findings v01](viscosity_full_findings_v01.md) records its pre-packaging status. The observed completion is documented in [full reproduction v01](viscosity_full_reproduction_v01.md) and the [package terminal receipt](../evidence/phase_11/viscosity_v01/full_v01_package/terminal.json). Root independent acceptance is pending as of this note.

Replay freshly queried the positive checkpoint selected at step 4,289 once: 332,004 physical states plus 365 dual-operator points. All 24 raw-array payloads matched their original keys, dtypes, shapes and values exactly. All eight canonical scientific report sections matched exactly, excluding timing/path/manifest provenance and the dual source-file SHA from the report comparison. The historical zero cache was reused unchanged. Replay performed no training or zero requery. Separate package/copy/archive/extraction/replay work took **5.087408 s** in total, as recorded in the terminal receipt.

The qualified scientific outcome remains unchanged: this fixed E1 case and seed had lower pooled errors against the inviscid reference, alongside larger inviscid momentum residual, left discharge mismatch and bias-subtracted mass excursion. Local errors and undefined front widths remain. The inviscid target is not an exact viscous solution; this result establishes no general improvement, well-posed viscous boundary-value problem, entropy selection, energy-dissipation or unconditional-stability guarantee. No momentum budget is reported.

This additive note records current status only. The v01 findings, reproduction note, handoff, manifests, frozen payloads, source, guards and archive remain unchanged. No analysis, media generation, retuning or archive rebuild was performed for v02.
