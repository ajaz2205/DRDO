# Novyte Materials Custom RL, Deep Learning, and DFT Inverse Design Workflow

## 1. Executive Summary

Novyte Materials has completed a computational materials discovery workflow in which custom reinforcement learning (RL) and deep learning models are used to generate and prioritize candidate functional crystalline materials, followed by first-principles DFT validation. The demonstrated run includes descriptor construction, RL-guided candidate search, deep learning property prediction, thermodynamic screening, convex-hull assessment, phonon stability checks, elastic stability checks, and thermal/Debye analysis.

This workflow directly maps to the DRDO research activity on AI/ML-based material development for thermal-signature reduction. The completed work proves the core capability: automated material candidate generation, physics-aware screening, and DFT-backed validation. The same framework can be extended toward controlled emissivity, infrared response, and coating-material inverse design by replacing or augmenting the target property from stability-only objectives to spectral/thermal objectives.

## 2. Demonstrated Capability

The completed run demonstrates the following technical capabilities:

1. AI/ML-assisted material candidate generation.
2. Descriptor-based screening using chemical, structural, and energetic features.
3. Custom RL search through a large candidate space.
4. DFT validation using electronic-structure calculations.
5. Formation-energy and convex-hull stability assessment.
6. Phonon-based dynamic stability verification.
7. Elastic tensor and Born-criteria-based mechanical stability assessment.
8. Debye/thermal analysis, including vibrational energy, heat capacity, entropy, and free-energy trends.
9. Identification of computationally stable candidate materials suitable for further property screening and experimental prioritization.

## 3. End-to-End Workflow

The workflow is organized as a design-validate loop:

1. Define candidate chemical space.
2. Convert each candidate into a numerical feature vector.
3. Apply chemical validity filters.
4. Predict stability-related properties using ML models.
5. Use custom RL policies to propose high-value candidates.
6. Validate shortlisted candidates using DFT.
7. Apply thermodynamic, dynamic, mechanical, and thermal checks.
8. Feed validated results back into the dataset.
9. Extend the same loop toward target properties such as emissivity, infrared response, and thermal contrast.

In compact form:

```text
Candidate chemistry -> descriptors -> deep learning prediction -> RL-guided selection
-> DFT validation -> stability/property verdict -> updated search model
```

## 4. Candidate Representation

Each candidate material is represented as a vector:

```text
m = {element set, stoichiometry, site assignment, charge state, structure type}
```

The ML model does not work directly on chemical symbols. Each candidate is converted into a numerical descriptor vector:

```text
x(m) = [r_i, chi_i, q_i, V_i, t, mu, IPE, E_lattice, Delta_chi, E_ref, ...]
```

Where:

- `r_i` = ionic or atomic radius descriptors.
- `chi_i` = electronegativity descriptors.
- `q_i` = oxidation-state or charge descriptors.
- `V_i` = ionic or atomic volume descriptors.
- `t` = geometric tolerance descriptor.
- `mu` = coordination/octahedral fit descriptor.
- `IPE` = ionic packing efficiency.
- `E_lattice` = approximate ionic cohesion descriptor.
- `Delta_chi` = electronegativity contrast between bonded species.
- `E_ref` = reference energetic information where available.

## 5. Chemical Validity Filters

Before expensive calculations, candidates are filtered using basic chemical constraints.

### 5.1 Charge Neutrality

For a candidate with species indexed by `i`, stoichiometric count `n_i`, and oxidation state `q_i`:

```text
Q_total = sum_i(n_i q_i)
```

The charge-neutrality condition is:

```text
Q_total = 0
```

Candidates that violate charge neutrality are penalized or rejected depending on the search stage.

### 5.2 Geometric Tolerance

A tolerance descriptor estimates whether the ions can pack into the desired crystalline framework:

```text
t = (r_large + r_anion) / [sqrt(2) (r_metal + r_anion)]
```

Values close to the stable window are prioritized. Values far outside the window indicate likely distortion, instability, or phase mismatch.

### 5.3 Coordination Fit

The coordination descriptor is:

```text
mu = r_metal / r_anion
```

This checks whether the central metal ion can fit into its local coordination environment.

### 5.4 Ionic Packing Efficiency

Ionic packing efficiency is estimated as:

```text
IPE = V_ions / V_cell
```

Where:

```text
V_ions = sum_i n_i (4/3) pi r_i^3
```

and `V_cell` is the unit-cell volume or an estimated cell volume from the structural model.

## 6. Energy Descriptors

### 6.1 Approximate Lattice Energy

A simplified Coulombic descriptor is used to estimate ionic cohesion:

```text
E_lattice proportional to sum_{i<j} (q_i q_j) / r_ij
```

After including the electrostatic constant and sign convention, more favorable ionic cohesion corresponds to lower lattice energy.

### 6.2 Formation Energy

The DFT or ML-predicted formation energy per atom is:

```text
Delta_E_form = [E_total(candidate) - sum_i n_i mu_i] / N_atoms
```

Where:

- `E_total(candidate)` = total energy of the candidate compound.
- `mu_i` = reference chemical potential of element `i`.
- `n_i` = number of atoms of element `i`.
- `N_atoms` = total atoms in the formula unit or computational cell.

Lower and negative formation energies indicate stronger thermodynamic favorability relative to elemental references.

### 6.3 Convex Hull Stability

Formation energy alone is not enough. A material must also be checked against competing phases. The energy above hull is:

```text
Delta_E_hull = E_form(candidate) - min_decomp sum_j lambda_j E_form(phase_j)
```

Subject to mass balance:

```text
sum_j lambda_j composition_j = composition_candidate
sum_j lambda_j = 1
lambda_j >= 0
```

Interpretation:

- `Delta_E_hull = 0`: thermodynamically stable against decomposition.
- Small positive `Delta_E_hull`: metastable, potentially synthesizable.
- Large positive `Delta_E_hull`: likely unstable.

## 7. Machine Learning Layer

The ML layer estimates expensive properties before full DFT validation.

### 7.1 Property Prediction

For a candidate descriptor vector `x`, the model predicts:

```text
E_form_hat = f_E(x)
G_hat = f_G(x)
S_hat = f_S(x)
```

Where:

- `E_form_hat` = predicted formation energy.
- `G_hat` = predicted Gibbs/free-energy-related stability metric.
- `S_hat` = predicted stability score.

In the demonstrated workflow, deep neural property models are used to prioritize candidates while reducing the number of expensive first-principles calculations.

### 7.2 Penalty-Aware Objective

The screening objective combines energy, structure, novelty, and feasibility:

```text
J_stability(m) =
  w1 E_form_hat
+ w2 G_hat
+ w3 P_charge
+ w4 P_geometry
+ w5 P_packing
+ w6 P_novelty
```

The model seeks candidates with low `J_stability`.

Penalty examples:

```text
P_charge = alpha_Q |Q_total|
```

```text
P_geometry = alpha_t max(0, |t - t_target| - delta_t)^2
```

```text
P_packing = alpha_IPE max(0, |IPE - IPE_target| - delta_IPE)^2
```

Soft penalties allow borderline candidates to remain in the search if their energetic promise is strong.

## 8. Custom Reinforcement Learning Search

Custom reinforcement learning is used to decide which candidate should be evaluated next. The RL controller treats materials discovery as a sequential decision problem where each proposed composition or coating candidate is an action, and the final reward is based on stability, thermal response, emissivity matching, and manufacturability.

At step `t`, the agent observes a state:

```text
s_t = [x(m_t), y_hat(m_t), DFT_status_t, target_IR, constraints, history_t]
```

Where:

- `x(m_t)` = material descriptor vector.
- `y_hat(m_t)` = deep learning predictions for stability and target properties.
- `DFT_status_t` = available first-principles validation results.
- `target_IR` = desired emissivity or thermal-response profile.
- `constraints` = stability, toxicity, cost, and processability limits.
- `history_t` = previously explored candidates and outcomes.

The agent chooses an action:

```text
a_t = policy_theta(s_t)
```

The action may generate a new candidate, modify an existing candidate, substitute an element, change stoichiometry, or adjust a coating formulation.

The reward is tied to the inverse-design objective:

```text
R_t = -J_total(m_t)
```

with:

```text
J_total(m) =
  a1 J_emissivity
+ a2 C_thermal
+ a3 P_stability
+ a4 P_mechanical
+ a5 P_process
+ a6 P_environment
```

The RL objective is:

```text
theta_star = argmax_theta E[sum_t gamma^t R_t]
```

The next candidate is selected by the learned policy:

```text
x_next = argmax_a pi_theta(a | s_t)
```

This is the core inverse search engine: instead of manually testing one material at a time, the custom RL model learns which candidate action is most likely to improve the material objective after accounting for DFT cost, stability risk, and the target IR/thermal response.

## 9. DFT Validation Layer

Shortlisted candidates are validated using first-principles calculations.

### 9.1 Geometry Relaxation

The structure is relaxed to minimize total energy:

```text
min_R,E_cell E_DFT(R, cell)
```

The relaxed structure should have low residual forces and low residual stress.

### 9.2 Self-Consistent Field Calculation

The electronic ground state is solved iteratively:

```text
rho_in -> H[rho] -> psi_n -> rho_out
```

Convergence condition:

```text
|rho_out - rho_in| < tolerance
```

The converged total energy is then used for stability calculations.

### 9.3 Kohn-Sham DFT

The Kohn-Sham total energy can be written conceptually as:

```text
E[rho] = T_s[rho] + E_ext[rho] + E_H[rho] + E_xc[rho] + E_ion-ion
```

Where:

- `T_s` = non-interacting kinetic energy.
- `E_ext` = electron-ion external potential energy.
- `E_H` = Hartree electron-electron energy.
- `E_xc` = exchange-correlation energy.
- `E_ion-ion` = ion-ion repulsion.

The Kohn-Sham equations are:

```text
[-1/2 nabla^2 + V_eff(r)] psi_n(r) = epsilon_n psi_n(r)
```

The electron density is:

```text
rho(r) = sum_n f_n |psi_n(r)|^2
```

## 10. Phonon Stability

Phonons test whether the relaxed structure is dynamically stable.

The dynamical matrix is:

```text
D_{alpha beta}^{ij}(q) =
  1 / sqrt(M_i M_j) sum_R Phi_{alpha beta}^{ij}(R) exp(i q . R)
```

The phonon frequencies are obtained from:

```text
det |D(q) - omega(q)^2 I| = 0
```

Stability condition:

```text
omega(q)^2 >= 0 for all relevant q
```

Imaginary phonon modes indicate a structural instability or distortion path. A candidate with real phonon modes is dynamically stable under small perturbations.

## 11. Elastic Stability

Mechanical stability is checked using the elastic stiffness matrix `C`.

General condition:

```text
u = 1/2 epsilon^T C epsilon > 0
```

for all nonzero strain vectors `epsilon`.

This means the stiffness matrix must be positive definite.

For cubic-like symmetry, the commonly used criteria are:

```text
C11 - C12 > 0
C44 > 0
C11 + 2 C12 > 0
```

Derived mechanical quantities include:

```text
B = bulk modulus
G = shear modulus
E = Young's modulus
nu = Poisson ratio
```

Pugh ratio:

```text
G / B
```

is used as a ductility/brittleness indicator.

## 12. Thermal/Debye Analysis

Thermal behavior is assessed using Debye-type vibrational thermodynamics.

Debye temperature:

```text
theta_D = (hbar / k_B) omega_D
```

Constant-volume heat capacity:

```text
C_V = 9 N k_B (T / theta_D)^3 integral_0^{theta_D/T}
      [x^4 exp(x) / (exp(x) - 1)^2] dx
```

Vibrational free energy:

```text
F_vib(T) = U_vib(T) - T S_vib(T)
```

These outputs help assess temperature response, vibrational robustness, and thermal behavior of candidate materials.

## 13. Extension To Infrared Response And Thermal Signature Reduction

The completed run proves the design-and-validation pipeline. For the DRDO target, the same pipeline can be extended by changing the objective function from stability-only to stability plus spectral thermal response.

### 13.1 Thermal Radiation Objective

Thermal radiance from a surface can be approximated as:

```text
L_lambda(T, m) = epsilon_lambda(m) B_lambda(T)
```

Where:

- `L_lambda` = spectral radiance.
- `epsilon_lambda` = spectral emissivity of material `m`.
- `B_lambda(T)` = Planck blackbody radiance.

Planck radiance:

```text
B_lambda(T) =
  [2 h c^2 / lambda^5] /
  [exp(h c / (lambda k_B T)) - 1]
```

Thermal contrast against a background can be written as:

```text
C_thermal(m) =
  integral_{lambda1}^{lambda2} w(lambda)
  |epsilon_lambda(m) B_lambda(T_object)
   - epsilon_lambda(background) B_lambda(T_background)| d lambda
```

The inverse-design goal is:

```text
min_m C_thermal(m)
```

subject to stability, processability, toxicity, cost, and coating constraints.

### 13.2 Emissivity Target Matching

If a target emissivity spectrum is specified:

```text
J_emissivity(m) =
  integral_{lambda1}^{lambda2} w(lambda)
  [epsilon_lambda(m) - epsilon_target(lambda)]^2 d lambda
```

The complete inverse objective becomes:

```text
J_total(m) =
  a1 J_emissivity
+ a2 C_thermal
+ a3 P_stability
+ a4 P_mechanical
+ a5 P_process
+ a6 P_environment
```

The inverse-design result is:

```text
m_star = argmin_m J_total(m)
```

### 13.3 Optical/Infrared Property From DFT

DFT can provide electronic and optical descriptors used for infrared-response prediction:

```text
epsilon_complex(omega) = epsilon_1(omega) + i epsilon_2(omega)
```

From this, optical constants can be derived:

```text
n(omega), k(omega), R(omega), A(omega)
```

For opaque coatings:

```text
T(omega) approximately 0
A(omega) = 1 - R(omega)
```

By Kirchhoff's law:

```text
epsilon(omega) = A(omega)
```

Thus the same DFT workflow can be extended to estimate emissivity-related descriptors through optical response calculations.

## 14. Inverse Design Loop For DRDO Use Case

For thermal-signature-reduction materials, the inverse design loop becomes:

```text
Define target IR/thermal response
-> generate candidate coating materials
-> predict stability and emissivity descriptors
-> choose next candidates using the custom RL policy
-> validate by DFT optical, phonon, elastic, and thermal calculations
-> rank candidates by thermal contrast and manufacturability
-> recommend top materials for synthesis/coating trials
```

## 15. Mapping To DRDO Requirements

| DRDO Requirement | Matching Novyte Capability |
|---|---|
| AI/ML-based material discovery | Demonstrated custom RL candidate generation and deep learning property prediction |
| Inverse material design | RL policy optimization and objective-driven candidate selection |
| DFT validation | Demonstrated DFT relaxation, SCF, phonon, elastic, and thermal validation |
| Emissivity prediction and optimization | Direct extension using DFT optical constants and emissivity objective functions |
| Thermal signature reduction | Direct extension using radiance/thermal-contrast minimization |
| Infrared-response-controlled coatings | Candidate ranking can include spectral emissivity, mechanical stability, and processability |
| Multispectral camouflage materials | Objective can be expanded across visible, near-infrared, mid-infrared, and long-wave infrared bands |

## 16. What Is Proven And What Is The Next Step

Proven by the completed run:

- AI/ML-assisted generation of candidate materials.
- DFT-backed stability validation.
- Phonon, elastic, and thermal output generation.
- A working computational pipeline for rational materials discovery.

Next extension for DRDO:

- add optical and dielectric-property calculations,
- train emissivity/infrared-response deep learning models,
- define background-matched thermal contrast objectives,
- run inverse design for candidate coating compositions,
- validate top candidates through DFT optical, thermal, and mechanical calculations,
- move shortlisted candidates toward synthesis and coating trials.

## 17. Suggested Technical Statement

Novyte Materials has already demonstrated a validated custom RL/deep learning-DFT materials discovery workflow. The completed run shows how candidate functional materials can be generated using RL-guided search and deep learning property models, then validated using first-principles calculations including energy, hull stability, phonon stability, elastic stability, and thermal/Debye analysis. This same workflow is directly extensible to the DRDO objective by incorporating emissivity, optical response, and thermal-contrast objectives for inverse design of infrared-response-controlled coating materials.
