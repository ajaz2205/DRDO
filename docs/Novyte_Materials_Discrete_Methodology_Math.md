# Novyte Materials Discrete Methodology For AI/ML-DFT Materials Discovery

## 1. Purpose

This document gives the discrete mathematical methodology behind Novyte Materials' AI/ML-DFT materials discovery workflow. It is written as a non-confidential technical method note for reviewers who want to see that the work is a real computational materials discovery pipeline and not a conceptual proposal.

The workflow combines:

1. discrete candidate generation,
2. descriptor engineering,
3. deep learning property prediction,
4. custom reinforcement learning (RL) based inverse search,
5. DFT validation,
6. phonon, elastic, thermal, and coating-relevance checks,
7. final ranking for thermal-signature-reduction material development.

## 2. Discrete Candidate Space

Let the material search space be:

```text
M = {m_1, m_2, ..., m_N}
```

Each material candidate is represented as a discrete tuple:

```text
m = (E, n, q, c, s)
```

where:

- `E = {e_1, e_2, ..., e_K}` is the selected element set,
- `n = [n_1, n_2, ..., n_K]` is the stoichiometric count vector,
- `q = [q_1, q_2, ..., q_K]` is the oxidation/charge vector,
- `c` is the candidate structural class or coordination environment,
- `s` is the processing/application state, such as bulk, thin-film, coating, or composite layer.

The candidate generation problem is:

```text
Find m* in M such that J_total(m*) is minimized
```

subject to:

```text
chemical validity,
DFT stability,
thermal-response suitability,
mechanical robustness,
coating-process feasibility.
```

## 3. Descriptor Construction

Each candidate `m` is converted into a numerical descriptor vector:

```text
x(m) = phi(m)
```

where:

```text
x(m) = [
  composition descriptors,
  charge descriptors,
  size descriptors,
  packing descriptors,
  coordination descriptors,
  electronegativity descriptors,
  approximate energetic descriptors,
  DFT-derived descriptors,
  thermal descriptors
]
```

### 3.1 Charge Neutrality

The total formal charge is:

```text
Q(m) = sum_i n_i q_i
```

The primary chemical validity condition is:

```text
Q(m) = 0
```

For learning and ranking, the charge penalty is:

```text
P_charge(m) = alpha_Q |Q(m)|
```

or, in squared form:

```text
P_charge(m) = alpha_Q Q(m)^2
```

### 3.2 Weighted Elemental Statistics

For any elemental property `p_i`, such as atomic radius, ionic radius, electronegativity, melting point, or valence electron count:

```text
p_mean(m) = [sum_i n_i p_i] / [sum_i n_i]
```

```text
p_var(m) = [sum_i n_i (p_i - p_mean)^2] / [sum_i n_i]
```

```text
p_range(m) = max_i(p_i) - min_i(p_i)
```

These descriptors allow deep learning models to detect chemical contrast, size mismatch, bonding polarity, and likely structural distortion.

### 3.3 Ionic Volume And Packing

The approximate ionic volume is:

```text
V_ions(m) = sum_i n_i (4/3) pi r_i^3
```

The packing descriptor is:

```text
IPE(m) = V_ions(m) / V_cell(m)
```

where `V_cell(m)` is estimated from the initial structure or extracted after DFT relaxation.

The packing penalty is:

```text
P_pack(m) = alpha_I max(0, |IPE(m) - IPE_target| - delta_I)^2
```

### 3.4 Size Compatibility

A size-compatibility descriptor is defined as:

```text
T_size(m) = (r_large + r_anion) / [sqrt(2) (r_center + r_anion)]
```

The penalty around a target structural window is:

```text
P_size(m) = alpha_T max(0, |T_size(m) - T_target| - delta_T)^2
```

### 3.5 Coordination Compatibility

The local coordination fit descriptor is:

```text
mu_coord(m) = r_center / r_anion
```

The coordination penalty is:

```text
P_coord(m) = alpha_mu max(0, |mu_coord(m) - mu_target| - delta_mu)^2
```

### 3.6 Electronegativity Contrast

For a dominant cation-anion interaction:

```text
Delta_chi(m) = |chi_cation - chi_anion|
```

For multiple interacting pairs:

```text
Delta_chi_weighted(m) =
  [sum_(i,j) w_ij |chi_i - chi_j|] / [sum_(i,j) w_ij]
```

This captures bonding polarity and helps screen candidates likely to form stable ionic/covalent networks.

### 3.7 Approximate Lattice Cohesion

A Coulombic cohesion descriptor is:

```text
E_latt_approx(m) = K sum_(i<j) [n_i n_j q_i q_j] / r_ij
```

where `K` is a proportionality constant and `r_ij` is an approximate interionic distance. This is not a replacement for DFT; it is a fast descriptor for prioritization.

## 4. Deep Learning Property Models

The deep learning model maps candidate descriptors to property predictions:

```text
y_hat(m) = f_theta(x(m))
```

where:

```text
y_hat(m) = [
  E_form_hat,
  E_hull_hat,
  S_stability_hat,
  omega_min_hat,
  C_stability_hat,
  theta_D_hat,
  thermal_score_hat,
  IR_score_hat
]
```

Here:

- `E_form_hat` is predicted formation energy,
- `E_hull_hat` is predicted energy above competing phases,
- `S_stability_hat` is the predicted stability score,
- `omega_min_hat` is a predicted minimum phonon-frequency indicator,
- `C_stability_hat` is a predicted elastic-stability score,
- `theta_D_hat` is a predicted Debye-temperature-related descriptor,
- `thermal_score_hat` is a thermal-response suitability score,
- `IR_score_hat` is the predicted infrared/coating relevance score.

### 4.1 Multi-Task Loss

For training data `D = {(x_i, y_i)}`, the deep learning loss is:

```text
L_property(theta) =
  sum_i [
    w_E ||E_form_i - E_form_hat_i||^2
  + w_H ||E_hull_i - E_hull_hat_i||^2
  + w_S ||S_i - S_hat_i||^2
  + w_T ||thermal_i - thermal_hat_i||^2
  ]
```

Physics penalties are added:

```text
L_physics(theta) =
  lambda_Q P_charge
+ lambda_I P_pack
+ lambda_T P_size
+ lambda_mu P_coord
```

The total training loss is:

```text
L_total(theta) = L_property(theta) + L_physics(theta) + lambda_R ||theta||^2
```

This makes the model both data-driven and physics-constrained.

## 5. Custom RL Formulation For Inverse Material Design

Novyte Materials' inverse search is formulated as a discrete sequential decision problem.

At step `t`, the RL state is:

```text
s_t = [
  x(m_t),
  y_hat(m_t),
  DFT_status_t,
  target_profile,
  constraint_vector,
  history_t
]
```

where:

- `x(m_t)` is the current descriptor vector,
- `y_hat(m_t)` is the deep learning prediction vector,
- `DFT_status_t` contains available first-principles validation outputs,
- `target_profile` contains the desired thermal/IR response,
- `constraint_vector` contains toxicity, stability, cost, and processability limits,
- `history_t` contains previously tested candidates and outcomes.

The RL action is:

```text
a_t in A
```

where the discrete action set can include:

```text
A = {
  substitute element,
  modify stoichiometry,
  change coordination environment,
  introduce dopant,
  generate layered coating candidate,
  reject candidate,
  send candidate to DFT validation
}
```

The policy network is:

```text
pi_theta(a_t | s_t)
```

The transition is:

```text
s_{t+1} = F(s_t, a_t)
```

## 6. Reward Function

The reward is the negative of the total material-design objective:

```text
R_t = -J_total(m_t)
```

The RL objective is:

```text
theta* = argmax_theta E_pi [sum_t gamma^t R_t]
```

where:

- `gamma` is the discount factor,
- `theta` contains policy-network parameters,
- `R_t` rewards stable, useful, low-risk material candidates.

The candidate selected at a given state is:

```text
m_next = argmax_m pi_theta(a_m | s_t)
```

## 7. Total Inverse Design Objective

For thermal-signature-reduction materials, the total objective is:

```text
J_total(m) =
  a1 J_stability(m)
+ a2 J_thermal(m)
+ a3 J_emissivity(m)
+ a4 J_mechanical(m)
+ a5 J_process(m)
+ a6 J_environment(m)
+ a7 J_DFT_cost(m)
```

The search target is:

```text
m* = argmin_m J_total(m)
```

## 8. Stability Objective

The stability objective is:

```text
J_stability(m) =
  b1 max(0, E_form(m) - E_form_target)
+ b2 max(0, E_hull(m) - E_hull_limit)
+ b3 P_charge(m)
+ b4 P_pack(m)
+ b5 P_size(m)
```

### 8.1 Formation Energy

The formation energy is:

```text
Delta_E_form(m) =
  [E_total(m) - sum_i n_i mu_i] / N_atoms
```

where:

- `E_total(m)` is the DFT total energy,
- `mu_i` is the reference chemical potential of element `i`,
- `n_i` is the number of atoms of element `i`,
- `N_atoms` is the total atom count.

### 8.2 Competing-Phase Stability

The energy above competing phases is:

```text
Delta_E_hull(m) =
  E_form(m) - min_lambda sum_j lambda_j E_form(p_j)
```

subject to:

```text
sum_j lambda_j composition(p_j) = composition(m)
```

```text
sum_j lambda_j = 1
```

```text
lambda_j >= 0
```

Interpretation:

```text
Delta_E_hull = 0        -> stable against decomposition
0 < Delta_E_hull < eps  -> metastable but potentially synthesizable
Delta_E_hull large      -> low priority
```

## 9. DFT Validation Layer

Shortlisted candidates are validated using first-principles electronic-structure calculations.

### 9.1 Kohn-Sham Energy Functional

The DFT energy functional is:

```text
E[rho] =
  T_s[rho]
+ E_ext[rho]
+ E_H[rho]
+ E_xc[rho]
+ E_ion-ion
```

The Kohn-Sham equation is:

```text
[-1/2 nabla^2 + V_eff(r)] psi_n(r) = epsilon_n psi_n(r)
```

The electron density is:

```text
rho(r) = sum_n f_n |psi_n(r)|^2
```

SCF convergence requires:

```text
||rho_out - rho_in|| < epsilon_SCF
```

### 9.2 Geometry Relaxation

The relaxed geometry is:

```text
R* = argmin_R E_DFT(R, cell)
```

Relaxation is accepted when:

```text
max_i |F_i| < F_tol
```

and:

```text
max_ab |sigma_ab| < sigma_tol
```

where `F_i` are atomic forces and `sigma_ab` is the stress tensor.

## 10. Dynamic Stability

Phonon stability is evaluated from the dynamical matrix:

```text
D_{alpha beta}^{ij}(q) =
  1 / sqrt(M_i M_j)
  sum_R Phi_{alpha beta}^{ij}(R) exp(i q . R)
```

The phonon frequencies satisfy:

```text
det |D(q) - omega(q)^2 I| = 0
```

The dynamic-stability condition is:

```text
omega(q)^2 >= 0
```

for the sampled phonon wavevectors. Candidates with physically meaningful imaginary modes are penalized:

```text
P_phonon(m) = alpha_ph sum_q max(0, -omega_min(q))^2
```

## 11. Elastic Stability

The elastic strain energy is:

```text
U_strain = 1/2 epsilon^T C epsilon
```

Mechanical stability requires:

```text
epsilon^T C epsilon > 0
```

for all nonzero strain vectors `epsilon`.

For common high-symmetry screening, the elastic conditions include:

```text
C11 - C12 > 0
C44 > 0
C11 + 2 C12 > 0
```

Derived properties are:

```text
B = bulk modulus
G = shear modulus
E = Young's modulus
nu = Poisson ratio
```

The mechanical objective is:

```text
J_mechanical(m) =
  c1 max(0, B_min - B)^2
+ c2 max(0, G_min - G)^2
+ c3 P_phonon(m)
```

## 12. Thermal Analysis

Thermal behavior is captured using vibrational thermodynamic descriptors.

Debye temperature:

```text
theta_D = (hbar / k_B) omega_D
```

Vibrational free energy:

```text
F_vib(T) = U_vib(T) - T S_vib(T)
```

Heat capacity:

```text
C_V =
  9 N k_B (T / theta_D)^3
  integral_0^(theta_D/T)
  [x^4 exp(x) / (exp(x) - 1)^2] dx
```

Thermal objective:

```text
J_thermal(m) =
  d1 |C_V(m,T) - C_V_target(T)|
+ d2 |F_vib(m,T) - F_target(T)|
+ d3 |theta_D(m) - theta_target|
```

## 13. Emissivity And Infrared Objective

For thermal signature reduction, the material is optimized against its spectral thermal radiation.

Spectral radiance:

```text
L_lambda(T,m) = epsilon_lambda(m) B_lambda(T)
```

Planck radiance:

```text
B_lambda(T) =
  [2 h c^2 / lambda^5] /
  [exp(h c / (lambda k_B T)) - 1]
```

Thermal contrast:

```text
C_thermal(m) =
  integral_{lambda1}^{lambda2}
  w(lambda)
  |epsilon_lambda(m) B_lambda(T_object)
   - epsilon_lambda(bg) B_lambda(T_bg)| d lambda
```

Target emissivity mismatch:

```text
J_emissivity(m) =
  integral_{lambda1}^{lambda2}
  w(lambda)
  [epsilon_lambda(m) - epsilon_target(lambda)]^2 d lambda
```

For opaque coatings:

```text
A(lambda) = 1 - R(lambda)
```

By Kirchhoff's law:

```text
epsilon(lambda) = A(lambda)
```

Thus, if DFT-derived optical constants provide reflectance `R(lambda)`, the emissivity descriptor can be estimated as:

```text
epsilon_hat(lambda) = 1 - R_hat(lambda)
```

## 14. Optical Constants From Electronic Response

The complex dielectric response is:

```text
epsilon_complex(omega) = epsilon_1(omega) + i epsilon_2(omega)
```

The refractive index `n` and extinction coefficient `k` satisfy:

```text
n(omega) = sqrt((|epsilon| + epsilon_1) / 2)
```

```text
k(omega) = sqrt((|epsilon| - epsilon_1) / 2)
```

Normal-incidence reflectance:

```text
R(omega) =
  [(n - 1)^2 + k^2] /
  [(n + 1)^2 + k^2]
```

The coating-relevant emissivity estimate is:

```text
epsilon_hat(omega) = 1 - R(omega)
```

for optically opaque conditions.

## 15. Discrete Acceptance Gates

A candidate advances only if it passes staged gates.

### Gate 1: Chemical Validity

```text
Q(m) = 0
```

```text
P_size(m) < tau_size
```

```text
P_pack(m) < tau_pack
```

### Gate 2: ML/RL Priority

```text
S_stability_hat(m) > S_min
```

```text
J_total_hat(m) < J_limit
```

### Gate 3: DFT Energetics

```text
Delta_E_form(m) < E_form_limit
```

```text
Delta_E_hull(m) < E_hull_limit
```

### Gate 4: Phonon And Elastic Stability

```text
P_phonon(m) < P_phonon_limit
```

```text
epsilon^T C epsilon > 0
```

### Gate 5: Thermal/IR Relevance

```text
C_thermal(m) < C_limit
```

```text
J_emissivity(m) < J_emissivity_limit
```

## 16. Ranking Score

The final ranking score is:

```text
Score(m) =
  r1 normalize(-Delta_E_form)
+ r2 normalize(-Delta_E_hull)
+ r3 normalize(S_stability)
+ r4 normalize(-P_phonon)
+ r5 normalize(C_mechanical)
+ r6 normalize(-C_thermal)
+ r7 normalize(-J_emissivity)
+ r8 normalize(processability)
```

The final recommendation set is:

```text
M_top = top_k {m in M_valid by Score(m)}
```

where `M_valid` contains only candidates that passed the acceptance gates.

## 17. Methodology Output

The workflow produces:

1. ranked candidate materials,
2. descriptor tables,
3. deep learning property predictions,
4. RL action/reward history,
5. DFT relaxation and SCF outputs,
6. formation-energy and stability outputs,
7. phonon/dynamical outputs,
8. elastic tensor outputs,
9. thermal/Debye outputs,
10. coating-relevant objective scores.

## 18. Why This Shows Real Materials Discovery Capability

The methodology is not a single prediction model. It is a closed materials discovery workflow:

```text
candidate generation
-> descriptor construction
-> deep learning prediction
-> RL inverse selection
-> DFT validation
-> stability verification
-> thermal/IR objective scoring
-> ranked material shortlist
```

The same framework can be used for defence coating development by changing the reward from stability-first discovery to thermal-signature-reduction discovery:

```text
R_t = -J_total(m_t)
```

where `J_total` includes emissivity, thermal contrast, DFT stability, mechanical robustness, and processability.

This is the core reason Novyte Materials' completed work matches the requested DRDO activity.
