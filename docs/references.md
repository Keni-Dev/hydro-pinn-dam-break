# References and source-use notes

## APA 7 reference list

Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T. N. T., James, F., & Cordier, S. (2013). SWASHES: A compilation of shallow water analytic solutions for hydraulic and environmental studies. *International Journal for Numerical Methods in Fluids, 72*(3), 269–300. [https://doi.org/10.1002/fld.3741](https://doi.org/10.1002/fld.3741)

Ketcheson, D. I., LeVeque, R. J., & del Razo, M. J. (2020). *Riemann problems and Jupyter solutions: Theory and approximate solvers for hyperbolic PDEs*. Society for Industrial and Applied Mathematics. [https://doi.org/10.1137/1.9781611976212](https://doi.org/10.1137/1.9781611976212)

Lax, P., & Wendroff, B. (1960). Systems of conservation laws. *Communications on Pure and Applied Mathematics, 13*(2), 217–237. [https://doi.org/10.1002/cpa.3160130205](https://doi.org/10.1002/cpa.3160130205)

Mumtaz, K., Nadeem, M. W., Khan, A., & Lakdawala, Z. (2025). Investigating the use of physics informed neural networks for dam-break scenarios. *PLOS ONE, 20*(9), Article e0332694. [https://doi.org/10.1371/journal.pone.0332694](https://doi.org/10.1371/journal.pone.0332694)

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics, 378*, 686–707. [https://doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing, 43*(5), A3055–A3081. [https://doi.org/10.1137/20M1318043](https://doi.org/10.1137/20M1318043)

## Source-use limits

- The SWASHES corrected author manuscript, arXiv v7, supplies the wet Stoker and dry Ritter benchmark descriptions. The journal citation remains 2013.
- The Riemann-book shallow-water material supports conservative states, characteristic speeds, shocks, and rarefactions. Retained source-comparison records identify the code revision used for independent checks.
- Lax and Wendroff are cited for historical lineage. The full original article was not inspected in the source audit; the implemented two-step predictor–corrector is defined in the project source, with Mumtaz et al. Section 2.2 as its procedural reference.
- Mumtaz et al. motivates the dam-break comparison. This project's analytical E1/E2 references and numerical benchmark are distinct from that study; no exact-replication claim is made.
- Raissi et al. supports the general coordinate-network, automatic-differentiation, collocation, and initial/boundary-loss formulation. It supplies no project-specific accuracy or conservation guarantee.
- Wang et al. motivates gradient-imbalance handling. The project's unweighted RMS, epsilon, clipping, and 100-step EMA update differs from the published algorithm.

Project performance claims use the project's saved runs. Analytical benchmarks and numerical refinement checks do not establish field validation. Included third-party interface-solver source excerpts retain their license notices and provenance. The genuine 2D reference uses a local unsplit finite-volume advance; it is not a PyClaw integrator execution.
