# E1 hybrid full-pair findings v01

The released exploratory pair completed 10,000 Adam updates per variant on the M5 CPU, float32, four threads. Adding 256 analytical h/q labels slightly reduced held-out depth and discharge errors, while velocity error, initial-volume bias, boundary mismatch and corrected mass imbalance were worse. This single E1 seed-11 pair does not support an overall accuracy or conservation improvement claim.

The immutable protocol is `docs/extensions/hybrid_protocol_v01.json`, SHA256 `be55cd1c5dfc3e57481be5ede94add2ca256135c362990b849dcb98adaccd80b`. It keeps architecture, scaling, fixed physics samples, 1/10/10 weights, initialization, Adam settings and step budget matched. The fresh control uses no interior-label fitting payload; the hybrid adds only the weight-1 scaled h/q data loss. The control and hybrid selected checkpoints at steps 6512 and 7198 using the same separate physics/IC/BC validation criterion. Both optimizer states still record all 10,000 updates. No analytical errors selected checkpoints or changed settings.

Labels v02 are idealized analytical E1 values, not field observations. Their independent selection seed is 110211. The 256 training labels use `0 < t < 0.35 s`, and 128 separate data-validation labels use `0.35 ≤ t < 0.40 s`. Final evaluation uses 400 equal cell centres and 11 times from 0.40 through 0.50 s. Physics collocation covers the full PDE domain; this is a holdout from solution labels within one physical case. Data-validation errors were read after both fits and were never used for fitting, checkpoint selection or tuning.

| Pooled final-window result, 4,400 points | Physics-only control | Hybrid |
|---|---:|---:|
| h RMSE (m) | 0.060806 | 0.057883 |
| h relative L2 (%) | 8.9816 | 8.5499 |
| q RMSE (m²/s per unit width) | 0.154016 | 0.147574 |
| q relative L2 (%) | 35.6900 | 34.1974 |
| Reference-wet u RMSE (m/s) | 0.580068 | 0.601934 |
| Reference-wet u relative L2 (%) | 59.5204 | 61.7640 |

All 4,400 reference depths exceed the fixed 0.001 m wet threshold. No difficult points were removed from h/q errors. The 198 saved metric rows include each time, all three fields, the whole domain, the fixed ±0.1 m shock band and its complement. Both final grids were finite with positive depth; that sign constraint does not establish solution accuracy.

The shock band contains 88 of the pooled final points. Its h RMSE was 0.165177 m for the control and 0.160101 m for the hybrid; q RMSE was 0.592809 versus 0.566938 m²/s. Wet-velocity RMSE in the band worsened from 1.980449 to 2.147265 m/s. At 0.50 s the exact shock is at 1.552567 m. Both predicted profiles have two downward crossings of the frozen depth threshold. The rightmost crossing is at 1.892754 m for the control and 1.885558 m for the hybrid, giving errors of +0.340187 and +0.332991 m. Both steepest-slope diagnostics lie at 1.875 m, +0.322433 m from the exact shock. These fixed finite-grid diagnostics retain the waveform defects; the small crossing difference does not establish a correctly resolved shock.

| Full-horizon mass/boundary result, doubled quadrature | Control | Hybrid |
|---|---:|---:|
| Predicted initial-volume error (m² per unit width) | −0.021883 | −0.042495 |
| Initial-volume error (% of prescribed 5.5 m²) | −0.3979 | −0.7726 |
| Maximum sampled absolute corrected imbalance (%) | 0.4344 | 1.0349 |
| Signed corrected imbalance at 0.50 s (%) | −0.4326 | −1.0349 |
| Maximum absolute boundary-q mismatch (m²/s) | 0.003926 | 0.015246 |
| Maximum corrected-balance change on doubling quadrature (m²) | 1.27e−7 | 5.46e−7 |

Mass uses the prescribed initial volume separately from the predicted initial volume and integrates the right-minus-left flux of the two 1D boundaries. Both prescribed boundary discharges are zero. Base and doubled volume quadratures use 2,000 and 4,000 midpoint cells. The declared 1,001/2,001 boundary-node grids become 1,002/2,002 nodes after inserting every exact reported time; the volume axis contains 52 times, including t=0 and the full 0–0.50 s horizon. The extra nearly coincident time comes from exact float membership of the final grid. Quadrature sensitivity is small here but is not an error bound. Reported mass maxima cover saved times, not continuous-time extrema.

Post-fit data-validation h RMSE was 0.047935 m for the control and 0.048169 m for the hybrid; q RMSE was 0.102045 versus 0.098977 m²/s. This separate window also gives mixed results.

The complete pair process took 758.972 s (12.650 min), including command startup and both run setups. Recorded training wall times were 378.368 s and 377.963 s; their sum is 756.331 s. Those timers include validation, checkpoint/history persistence and memory observations inside the training stage. The full process difference also includes work outside those timers. The observed timing difference does not establish that the hybrid is faster. Mean objective/gradient times were 0.02597 and 0.02631 s per step.

Generated analytical label evaluation took 0.006171 s; coordinate checks and array saving brought saved-array generation to 0.011493 s. The original label CLI process took 1.889 s including startup and metadata. The original final evaluator process took 1.464 s; its recorded evaluation work before report writing was 0.3982 s, including final reference generation of 0.000821 s. The fixed 400×11 inference benchmark used ten warmups and thirty repetitions: median/IQR were 1.573/0.080 ms for the control and 1.631/0.237 ms for the hybrid. Tensor conversion and result transfer are included; model loading, coordinate-rectangle construction, file I/O and metrics are separate. This differs from the core study's 400×50 workload and supplies no solver speedup claim.

The training process-lifetime peak RSS was 411,893,760 bytes (about 393 MiB). It is cumulative across the sequential variants. Current RSS and a process inventory remained unavailable because the sandbox denied `ps`; free memory, system pressure, allocator peaks and thermal telemetry are unestablished. No unavailable Victus, CUDA, MPS, paid service, cloud job, new dependency installation or additional fitting was used.

The executable snapshot contains the frozen Python source, referenced configs, lock/environment files, labels v02, both complete runs, original evaluation and replay helpers. `results/extensions/hybrid/hybrid_e1_seed11_v01.zip` has 85 members, 8,468,200 bytes, SHA256 `f5c5bd34f3619374c98fd1f0d1c2896bcbeac215a163c878e1133f1c8bd05f3a`. CRC/member/extracted hashes passed. An actual saved-checkpoint evaluator ran from the extracted tree: all 16 imported project modules resolved inside its `src`. All 50 arrays across 13 NPZ files and all six scientific JSON sections matched the original exactly. Timings and lifecycle/provenance metadata were retained separately. This was same-host saved-checkpoint replay with the existing locked environment; no training reproduction or process-wide filesystem-read exclusion is claimed. The original checkout remained OS-accessible.

Three saved-array figures, with PNG and PDF exports, are in `results/extensions/hybrid/figures_v01/`: final h/q profiles, held-out errors, and full-horizon corrected mass with both boundary fluxes. Their provenance is in `figure_manifest.json`; the PNG exports were visually checked. Plot export took 16.482 s, including Matplotlib's temporary font-cache creation. This presentation cost is separate from model/evaluation timings.

The original pilot findings, manifest, handoff and protocol remain byte-identical. The v01 label attempt and failed test assertion remain preserved. The original full fits, original evaluator and extracted replay each completed on their first attempt without a scientific-source repair. All 20 protected core modules and the accepted core archive retain their preservation hashes. The core paper, deck, video, coordination records and global implementation status were not edited by this work. No Git staging, commit or remote action was performed. Human participation, recording, native-application, submission and recipient acceptance remain separate.
