# Vlasov-Poisson System Solver using PINNs

This repository contains an implementation of a **Physics-Informed Neural Network (PINN)** to solve the **Vlasov-Poisson System**, a fundamental equation in plasma physics.

## 1. Problem Formulation

The **Vlasov-Poisson system** describes the evolution of a distribution function $f(x, v, t)$ in a self-consistent electrostatic field:

$\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + E \frac{\partial f}{\partial v} = 0$

where $E(x, t)$ is determined by the **Poisson equation**:

$\frac{\partial E}{\partial x} = \rho - 1$

with charge density:

$\rho(x, t) = \int f(x, v, t) \, dv$

### PINN Approach:
- A neural network approximates $f(x, v, t)$.
- The **loss function** enforces:
  - **Vlasov equation residual:** $\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + E \frac{\partial f}{\partial v} = 0$
  - **Poisson equation residual:** $\frac{\partial E}{\partial x} = \rho - 1$
  - **Boundary & Initial conditions**

## 2. Implementation Details

- **Neural Network Inputs:** $(x, v, t)$
- **Outputs:** $f(x, v, t)$
- **Loss Terms:**
  - **Physics Loss:** Enforces Vlasov & Poisson equations.
  - **Data Loss:** Ensures correct initial/boundary conditions.
- **Checkpointing:** Model checkpoints are saved after training.

## 3. Dependencies

Ensure you have the following installed:

```bash
pip install torch numpy matplotlib tqdm
