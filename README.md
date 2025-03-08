# Solving the Vlasov-Poisson System Using a Physics-Informed Neural Network (PINN)

## **Introduction**
The **Vlasov-Poisson system** describes the evolution of a **collisionless plasma** under electrostatic forces. It consists of two coupled equations:
1. The **Vlasov equation**, which governs the evolution of the distribution function \( f(x, v, t) \).
2. The **Poisson equation**, which determines the self-consistent electric field.

## **Mathematical Formulation**
The **Vlasov equation** for an electron plasma is:
$\[\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + E \frac{\partial f}{\partial v} = 0\]$

where:
- $\( f(x, v, t) \)$ is the **electron distribution function**,
- $\( E(x, t) \)$ is the **electric field**.

The **Poisson equation** relates the electric field to the charge density:

$\[\frac{\partial E}{\partial x} = \rho_e - 1\]$

### **Why is There a −1 in the Poisson Equation?**
This comes from **Gauss’s law**:

$\[\nabla \cdot \mathbf{E} = \rho\]$

In a **normalized plasma**, the total charge density consists of:
1. **Electrons**, which contribute a density $\( \rho_e \)$.
2. **A uniform background of ions**, which is assumed to have a constant density $\( \rho_{\text{background}} = 1 \)$.

Thus, the **modified Poisson equation** becomes:

$\[\frac{\partial E}{\partial x} = \rho_e - 1\]$

This ensures that in equilibrium $(\( \rho_e = 1 \))$, the net electric field is **zero**, preventing unphysical charge accumulation.

---

## **PINN Approach to Solve the Vlasov-Poisson System**
A **Physics-Informed Neural Network (PINN)** is used to solve this system by minimizing a loss function derived from the governing equations.

### **Neural Network Model**
I used a **Deep Ritz architecture** with Fourier embeddings.

