import jax
from jax import grad, vmap, jit, jacrev, jacfwd
import jax.numpy as jnp
from jax.experimental.ode import odeint as jodeint
from functools import partial
import optax
from optax import adam
from scipy.optimize import differential_evolution, dual_annealing, minimize, NonlinearConstraint, Bounds, fsolve
from jax.config import config
config.update("jax_enable_x64", True)
from cyipopt import minimize_ipopt
import pandas as pd
import matplotlib.pyplot as plt

# Code of simple version of problem (ideal K values used, only reactor temperature, separator temperature, and
# separator pressure optimised

class Species:
    heat_of_formation_298 = jnp.array([-110.525, -393.509, -200.660, 0.0, -241.818]) * 1000  # J
    A = jnp.array([3.376, 5.457, 2.211, 3.249, 3.47, 1.702, 3.28, 3.639])
    B = jnp.array([0.557, 1.045, 12.216, 0.422, 1.45, 9.081, 0.593, 0.506]) * 10 ** -3
    C = jnp.array([0.0, 0.0, -3.45, 0.0, 0.0, -2.164, 0.0, 0.0]) * 10 ** -6
    D = jnp.array([-0.031, -1.157, 0.0, 0.083, 0.121, 0.0, 0.04, -0.227]) * 10 ** 5
    T_0 = 298  # K
    R = 8.314  # J/mol.K

    def heat_of_formation(self, T):
        T_0 = self.T_0
        delta_constants_i = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, -1],
                                       [0, 0, 0, -2, 0, 0, 0, -0.5],
                                       [0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, -1, 0, 0, 0, -0.5]])
        delta_A = delta_constants_i @ jnp.transpose(self.A)
        delta_B = delta_constants_i @ jnp.transpose(self.B)
        delta_C = delta_constants_i @ jnp.transpose(self.C)
        delta_D = delta_constants_i @ jnp.transpose(self.D)
        delta_Cp = delta_A + delta_B / 2 * T_0 * (T / T_0 + 1) + delta_C / 3 * T_0 ** 2 * (
                (T / T_0) ** 2 + T / T_0 + 1) + delta_D / T / T_0
        heat_of_formation = self.heat_of_formation_298 + delta_Cp * self.R * (T - T_0)
        return heat_of_formation

    def specific_heat(self, T, y_i):
        A = self.A[0:7]
        B = self.B[0:7]
        C = self.C[0:7]
        D = self.D[0:7]
        Cp_i = (A + B * T + C * T ** 2 + D / (T ** 2)) * self.R
        Cp = sum(y_i * Cp_i)
        return Cp

    def K_values(self, T, P):
        A = jnp.array([6.72527, 7.58828, 7.9701, 6.14858, 8.05573, 6.84566, 6.72531])
        B = jnp.array([295.228, 861.82, 1521.12, 80.948, 1723.64, 435.621, 285.573])
        C = jnp.array([268.243, 271.883, 234.0, 277.532, 233.076, 271.361, 270.087])
        Psat_i = 10 ** (A - B / (C + T - 273.15)) * 133.322 * 10 ** -5  # bar
        K_values = Psat_i / P
        return K_values

class Reactor:
    # constant parameters
    species = Species()
    R = 8.314472  # m3/bar/K/mol
    P = 80.0  # bar
    e_cat = 0.285  # void fraction of catalyst bed
    p_cat = 1190.0  # kg/m3, density of catalyst bed
    a = 1.0  # activity of catalyst,
    D_i = 0.04  # m, internal diameter of self tube
    MW_i = jnp.array(
        [28.0097, 44.0087, 32.04106, 2.01568, 18.01468, 16.04206,
         28.0134])  # molecular weights (CO, CO2, CH3OH, H2, H20, CH4, N2)
    U_shell = 118.44  # W/m2K, overall heat transfer coefficient from self tube-side to shell-side
    z = 7.0  # m, length of PFR
    volume_t = jnp.pi * D_i ** 2 / 4 * z
    A = jnp.pi * D_i ** 2 / 4  # cross-sectional area of self tube
    n_i = 1.0
    T_shell = 490.0  # K, temperature of boiling water on the shell side
    Nt = 1620.0

    def __init__(self, no_ode_steps=100):
        self.no_ode_steps = no_ode_steps

    def kinetic_model(self, T):
        k_constants = jnp.array([[1.07, 36696], [1.22 * 10 ** 10, -94765]])
        K_constants = jnp.array([[3453.38, 0.0], [0.499, 17197], [6.62 * 10 ** -11, 124119]])  # K by A/B
        Keq_constants = jnp.array([[3066.0, 10.592], [-2073.0, -2.029]])  # K by A/B

        k = k_constants[:, 0] * jnp.exp(k_constants[:, 1] / self.R / T)
        K = K_constants[:, 0] * jnp.exp(K_constants[:, 1] / self.R / T)
        Keq = 10.0 ** (Keq_constants[:, 0] / T - Keq_constants[:, 1])
        return jnp.concatenate((k, K, Keq))

    def rate_expression(self, T, F_i):
        p = self.partial_pressure(F_i)
        p_CO = p[0]
        p_CO2 = p[1]
        p_CH3OH = p[2]
        p_H2 = p[3]
        p_H20 = p[4]

        reaction_equilibrium_constants = self.kinetic_model(T)  # kMeOH, kRWGS, K1, K2, K3, K1eq, K2eq
        k_CH3OH = reaction_equilibrium_constants[0]
        k_RWGS = reaction_equilibrium_constants[1]
        K_1 = reaction_equilibrium_constants[2]
        K_2 = reaction_equilibrium_constants[3]
        K_3 = reaction_equilibrium_constants[4]
        K_1eq = reaction_equilibrium_constants[5]
        K_2eq = reaction_equilibrium_constants[6]

        rate_CH3OH = k_CH3OH * p_CO2 * p_H2 * (1 - (1 / K_1eq) * p_H20 * p_CH3OH / (p_H2 ** 3 * p_CO2)) / (
                1 + K_1 * (p_H20 / p_H2) + K_2 * p_H2 ** 0.5 + K_3 * p_H20) ** 3
        rate_RWGS = k_RWGS * p_CO2 * (1 - 1 / K_2eq * (p_H20 * p_CO) / (p_CO2 * p_H2)) / (
                1 + K_1 * (p_H20 / p_H2) + K_2 * p_H2 ** 0.5 + K_3 * p_H20)

        stoich_coefficient_CH3OH = jnp.array([0.0, -1.0, 1.0, -3.0, 1.0, 0.0, 0.0])
        stoich_coefficient_RWGS = jnp.array([1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0])
        r_i = (stoich_coefficient_CH3OH * rate_CH3OH + stoich_coefficient_RWGS * rate_RWGS)
        return r_i

    def molar_fraction(self, F_i):
        y_i = F_i / jnp.sum(F_i)
        return y_i

    def partial_pressure(self, F_i):
        p = F_i / jnp.sum(F_i) * self.P
        return p

    def total_molar_t(self, F_i):
        return jnp.sum(F_i)

    def var_0(self, T_0, F_i0):
        return jnp.concatenate((F_i0, T_0))

    def pfr(self, var, z):
        F_i = var[0:7]
        T = var[7]
        ft = self.total_molar_t(F_i)
        y_i = self.molar_fraction(F_i)
        r_i = self.rate_expression(T, F_i)
        Cp = self.species.specific_heat(T, y_i)
        Hf = self.species.heat_of_formation(T)
        dFdz = self.Nt * self.A * self.p_cat * (1 - self.e_cat) * r_i
        dTdz = (jnp.sum(self.n_i * r_i[0:5] * -Hf) * self.p_cat * self.a + self.U_shell * 4 / self.D_i * (
                self.T_shell - T)) / ft / Cp * self.A * self.Nt * (1 - self.e_cat)
        return jnp.concatenate((dFdz, dTdz[None]))

    def mass_energy_balance(self, T_0, F_i0):
        Z = jnp.linspace(0.0, self.z, self.no_ode_steps)
        var_0 = self.var_0(T_0, F_i0)
        sol = jodeint(self.pfr, var_0, Z)
        return sol

class Separator:
    species = Species()
    learning_rate = 1e-2
    optimizer = adam(learning_rate)

    def __init__(self, no_optim_steps=100, no_RR_loop_steps=50):
        self.no_optim_steps = no_optim_steps
        self.no_RR_loop_steps = no_RR_loop_steps

    def rachford_rice_call(self, F_i, K_i):
        def rachford_rice_equation_intermofV(V):
            return self.rachford_rice_equation(F_i, K_i, V)

        def deriv_rachford_rice_equation_intermofV(V):
            return self.deriv_rachford_rice_equation(F_i, K_i, V)

        def rachfordrice_loop(V, n):
            info_of_interest = V, n
            F = self.total_flowrate(F_i)
            V_new = (V / F + rachford_rice_equation_intermofV(V) / deriv_rachford_rice_equation_intermofV(V)) * F
            return V_new, info_of_interest

        F = self.total_flowrate(F_i)
        V0 = jnp.array([0.9 * F])
        ns = jnp.arange(self.no_RR_loop_steps)
        V_final, info_along_the_way = jax.lax.scan(rachfordrice_loop, V0, ns)
        return V_final[0]

    def rachford_rice_equation(self, F_i, K_i, V):
        z_i = self.molar_fraction(F_i)
        F = self.total_flowrate(F_i)
        RR = z_i * (K_i - 1) / (1 + V / F * (K_i - 1))
        RR = jnp.sum(RR)
        return RR

    def deriv_rachford_rice_equation(self, F_i, K_i, V):
        z_i = self.molar_fraction(F_i)
        F = self.total_flowrate(F_i)
        RR = z_i * (K_i - 1) ** 2 / (1 + V / F * (K_i - 1)) ** 2
        RR = jnp.sum(RR)
        return RR

    def mass_balance(self, F_i, T, P):
        Ki = self.species.K_values(T, P)
        V = self.rachford_rice_call(F_i, Ki)
        F = self.total_flowrate(F_i)
        z_i = self.molar_fraction(F_i)
        L = F - V
        x_i = F * z_i / (L + V * Ki)
        y_i = x_i * Ki
        F_vap = y_i * V
        F_liq = x_i * L
        return F_vap, F_liq

    def molar_fraction(self, F_i):
        y_i = F_i / jnp.sum(F_i)
        return y_i

    def total_flowrate(self, F_i):
        return jnp.sum(F_i)

class MixSplit:
    def mix(self, F_1, F_2):
        return F_1 + F_2

    def split(self, F_i, split_factor):
        F_out1 = split_factor * F_i
        F_out2 = (1 - split_factor) * F_i
        return F_out1, F_out2

class Process:
    reactor = Reactor()
    separator = Separator()
    mixsplit = MixSplit()

    MW_i = jnp.array([28.0097, 44.0087, 32.04106, 2.01568, 18.01468, 16.04206,
                      28.0134])  # molecular weights (CO, CO2, CH3OH, H2, H20, CH4, N2)
    F_i0 = jnp.array([10727.9, 23684.2, 756.7, 9586.5, 108.8, 4333.1,
                      8072.0]) / 3600 * 1000 / MW_i  # mol/s, molar flow of species
    split_factor = 0.5  # constant as value to optimised

    @partial(jax.jit, static_argnums=(0))
    def objective_function(self, x):
        T_reactor = jnp.array([x[0]])
        T_separator = jnp.array([x[1]])
        P_separator = jnp.array([x[2]])
        F6 = x[3:10]
        F1 = self.F_i0
        F2 = self.mixsplit.mix(F1, F6)
        F3 = self.reactor.mass_energy_balance(T_reactor, F2)[-1, 0:7]
        F4, F7 = self.separator.mass_balance(F3, T_separator, P_separator)
        output = -F7[2]
        return output

    @partial(jax.jit, static_argnums=(0))
    def equality_constraints(self, x):
        T_reactor = jnp.array([x[0]])
        T_separator = jnp.array([x[1]])
        P_separator = jnp.array([x[2]])
        F6 = x[3:10]
        F1 = self.F_i0
        F2 = self.mixsplit.mix(F1, F6)
        F3 = self.reactor.mass_energy_balance(T_reactor, F2)[-1, 0:7]
        F4, F7 = self.separator.mass_balance(F3, T_separator, P_separator)
        F6_solve, F5 = self.mixsplit.split(F4, self.split_factor)
        balance = F4 - F5 - F6
        return balance

    @partial(jax.jit, static_argnums=(0))
    def grad_equality_constraints(self, x):
        output = jacrev(self.equality_constraints)(x)
        return output

    @partial(jax.jit, static_argnums=(0))
    def grad_objective_function(self, x):
        output = grad(self.objective_function)(x)
        return output

    @partial(jax.jit, static_argnums=(0))
    def hess_objective_function(self, x):
        return jacrev(jacrev(self.objective_function))(x)

    @partial(jax.jit, static_argnums=(0))
    def hess_equality_constraints(self, x):
        return jacrev(self.grad_equality_constraints)(x)

    def stream_information(self, x):
        T_reactor = jnp.array([x[0]])
        T_separator = jnp.array([x[1]])
        P_separator = jnp.array([x[2]])
        F6 = x[3:10]
        F1 = self.F_i0
        F2 = self.mixsplit.mix(F1, F6)
        F3 = self.reactor.mass_energy_balance(T_reactor, F2)[-1, 0:7]
        F4, F7 = self.separator.mass_balance(F3, T_separator, P_separator)
        F6_solve, F5 = self.mixsplit.split(F4, self.split_factor)
        stream_table = pd.DataFrame(
            {"F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6, "F6_SOLVE": F6_solve, "F7": F7})
        return stream_table

if __name__ == '__main__':
    process = Process()
    reactor = Reactor()
    separator = Separator()
    species = Species()
    mixsplit = MixSplit()

    # Solve for initial guess
    T_reactor = jnp.array([520.0])
    T_separator = jnp.array([320.0])
    P_separator = jnp.array([60.0])
    def mass_balance(x):
        F6 = x[0:7]
        F1 = process.F_i0
        F2 = mixsplit.mix(F1, F6)
        F3 = reactor.mass_energy_balance(T_reactor, F2)[-1, 0:7]
        F4, F7 = separator.mass_balance(F3, T_separator, P_separator)
        F6_solve, F5 = mixsplit.split(F4, process.split_factor)
        balance = F4 - F5 - F6
        return balance
    x0_fsolve = jnp.array([4.09224872e+01,
       9.22305520e+01, 3.43026095e+00, 1.00000001e+02, 3.04671303e-01,
       7.22343879e+01, 7.86732938e+01])
    x_solve = fsolve(mass_balance, x0_fsolve)
    UB = x_solve*1.5
    x0 = jnp.concatenate((T_reactor, T_separator, P_separator, x_solve))

    # Ipopt
    cons_ipopt = [{'type': 'eq', 'fun': process.equality_constraints, 'jac': process.grad_equality_constraints}]
    bons = [(500, 583), (308, 323), (50, 75), (0, UB[0]), (0, UB[1]), (0, UB[2]), (0, UB[3]), (0, UB[4]),
             (0, UB[5]), (0, UB[6])]
    result_ipopt = minimize_ipopt(process.objective_function, x0=x0, bounds=bons, constraints=cons_ipopt,
                                  options={'disp': 5})
    result_ipopt_jac = minimize_ipopt(process.objective_function, jac=process.grad_objective_function, x0=x0,
                                      bounds=bons, constraints=cons_ipopt)
    print(result_ipopt)
    print(result_ipopt_jac)
    data_ipopt_jac = process.stream_information(result_ipopt["x"])
    data_ipopt_jac.to_csv("data ipopt jac.csv")
    data_ipopt = process.stream_information(result_ipopt["x"])
    data_ipopt.to_csv("data ipopt.csv")

    # Differential Evolution
    cons_DE = NonlinearConstraint(process.equality_constraints,
                                  jnp.array([-10**-10, -10**-10, -10**-10, -10**-10, -10**-10, -10**-10, -10**-10]),
                                  jnp.array([10**-10, 10**-10, 10**-10, 10**-10, 10**-10, 10**-10, 10**-10]))
    result_DE = differential_evolution(process.objective_function, bounds=bons, constraints=cons_DE, x0=x0, disp=True)
    print(result_DE)

