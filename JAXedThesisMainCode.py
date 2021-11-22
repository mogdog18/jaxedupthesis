import jax
import scipy.misc
from jax import grad, vmap, jit, jacrev, jacfwd
import jax.numpy as jnp
from jax.experimental.ode import odeint as jodeint
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import differential_evolution, dual_annealing, minimize, NonlinearConstraint, Bounds, fsolve
from jax.config import config
import chex
import numpy as np

config.update("jax_enable_x64", True)
from cyipopt import minimize_ipopt
# jax.config.update('jax_disable_jit', True)
import pandas as pd
import numdifftools as nd

# Class containing thermodynamic data of species
class Species:
    heat_of_formation_298 = jnp.array([-110.525, -393.509, -200.660, 0.0, -241.818]) * 1000  # J
    A = jnp.array([3.376, 5.457, 2.211, 3.249, 3.47, 1.702, 3.28, 3.639])
    B = jnp.array([0.557, 1.045, 12.216, 0.422, 1.45, 9.081, 0.593, 0.506]) * 10 ** -3
    C = jnp.array([0.0, 0.0, -3.45, 0.0, 0.0, -2.164, 0.0, 0.0]) * 10 ** -6
    D = jnp.array([-0.031, -1.157, 0.0, 0.083, 0.121, 0.0, 0.04, -0.227]) * 10 ** 5
    T_0 = 298  # K
    R = 8.314472  # J/mol.K
    Tc = jnp.array([132.9, 304.1, 512.6, 33.0, 647.3, 190.4, 126.2])  # critical temperature, K
    Pc = jnp.array([35.0, 73.8, 80.9, 12.9, 221.2, 46.0, 33.9])  # critical pressure, bar
    w_i = jnp.array([0.066, 0.239, 0.556, -0.216, 0.344, 0.011, 0.039])  # acentric factor

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

    def fugacity(self, T, P, y_i, x_i):     # function to determine fugacity coefficient
        k_i = 0.37464 + 1.5322 * self.w_i - 0.26992 * self.w_i ** 2
        alpha_i = (1 + k_i * (1 - (T / self.Tc) ** 0.5)) ** 2
        R = self.R * 10 ** -5
        a_i = jnp.array([(0.457235 * R ** 2 * self.Tc ** 2 / self.Pc * alpha_i)])
        b_i = 0.077796 * R * self.Tc / self.Pc
        b_v = jnp.sum(y_i * b_i)
        b_l = jnp.sum(x_i * b_i)
        binary_parameter = jnp.array([[0.0, 0.0, 0.0, 0.0919, 0.0, 0.03, 0.033],
                                      [0.0, 0.0, 0.022, -0.1622, 0.0063, 0.0793, -0.0222],
                                      [0.0, 0.022, 0.0, 0.0, -0.0778, 0.0, -0.2141],
                                      [0.0919, -0.1622, 0.0, 0.0, 0.0, 0.0263, 0.0711],
                                      [0.0, 0.0063, -0.0778, 0.0, 0.0, 0.0, 0.0],  # 0.5619, 0.8014
                                      [0.03, 0.0793, 0.0, 0.0263, 0.0, 0.0, 0.0289],
                                      [0.033, -0.0222, -0.2141, 0.0711, 0.0, 0.0289, 0.0]])
        a_i_matrix = jnp.transpose(a_i ** 0.5) @ (a_i ** 0.5)
        a_ij = (1 - binary_parameter) * a_i_matrix
        y_i = jnp.array([y_i])
        x_i = jnp.array([x_i])
        a_v = jnp.sum(jnp.sum(jnp.transpose(y_i) @ y_i * a_ij))
        a_l = jnp.sum(jnp.sum(jnp.transpose(x_i) @ x_i * a_ij))
        A_l = a_l * P / (R ** 2 * T ** 2)
        A_v = a_v * P / (R ** 2 * T ** 2)
        B_l = b_l * P / R / T
        B_v = b_v * P / R / T
        alpha_l = -(1 - B_l)[0]
        beta_l = (A_l - 3 * B_l ** 2 - 2 * B_l)[0]
        gama_l = -(A_l * B_l - B_l ** 2 - B_l ** 3)[0]
        alpha_v = -(1 - B_v)[0]
        beta_v = (A_v - 3 * B_v ** 2 - 2 * B_v)[0]
        gama_v = -(A_v * B_v - B_v ** 2 - B_v ** 3)[0]
        coefficient_l = jnp.array([1.0, alpha_l, beta_l, gama_l])
        coefficient_v = jnp.array([1.0, alpha_v, beta_v, gama_v])
        roots_l = jnp.roots(coefficient_l, strip_zeros=False)
        roots_v = jnp.roots(coefficient_v, strip_zeros=False)
        Z_l = jnp.real(jnp.sort_complex(roots_l)[0])
        Z_v = jnp.real(
            jnp.max(jnp.where(condition=jnp.isreal(roots_v), x=roots_v, y=jnp.ones_like(roots_v) * -jnp.inf)))
        log_term_l = jnp.log(Z_l - B_l)
        log_term_v = jnp.log(Z_v - B_v)
        phi_l_i = jnp.exp(b_i / b_l * (Z_l - 1) - log_term_l - A_l / (2 * 2 ** 0.5 * B_l) * (
                2 / a_l * jnp.transpose(a_ij @ jnp.transpose(x_i)) -
                b_i / b_l) * jnp.log((Z_l + (1 + 2 ** 0.5) * B_l) / (Z_l + (1 - 2 ** 0.5) * B_l)))
        phi_v_i = jnp.exp(b_i / b_v * (Z_v - 1) - log_term_v - A_v / (2 * 2 ** 0.5 * B_v) * (
                2 / a_v * jnp.transpose(a_ij @ jnp.transpose(y_i)) -
                b_i / b_v) * jnp.log((Z_v + (1 + 2 ** 0.5) * B_v) / (Z_v + (1 - 2 ** 0.5) * B_v)))
        return jnp.reshape(phi_v_i, (7)), jnp.reshape(phi_l_i, (7))

    def specific_heat(self, T, y_i):
        A = self.A[0:7]
        B = self.B[0:7]
        C = self.C[0:7]
        D = self.D[0:7]
        Cp_i = (A + B * T + C * T ** 2 + D / (T ** 2)) * self.R
        Cp = sum(y_i * Cp_i)
        return Cp

    def K_values_ideal(self, T, P):
        A = jnp.array([6.72527, 7.58828, 7.9701, 6.14858, 8.05573, 6.84566, 6.72531])
        B = jnp.array([295.228, 861.82, 1521.12, 80.948, 1723.64, 435.621, 285.573])
        C = jnp.array([268.243, 271.883, 234.0, 277.532, 233.076, 271.361, 270.087])
        Psat_i = 10 ** (A - B / (C + T - 273.15)) * 133.322 * 10 ** -5  # bar
        K_values = Psat_i / P
        return K_values

    def K_values(self, T, P, y_i, x_i):
        phi_v_i, phi_l_i = self.fugacity(T, P, y_i, x_i)
        K_i = phi_l_i / phi_v_i
        return K_i

# class containing functions describing reactor model
class Reactor:
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

    # Model validation - conditions
    # P = 68.2
    # T_shell = 485

    def __init__(self, no_ode_steps=1000):
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

    def mass_energy_balance(self, T_0, F_i0):
        Z = jnp.linspace(0.0, self.z, self.no_ode_steps)
        var_0 = self.var_0(T_0, F_i0)

        def pfr(var, z):
            F_i = var[0:7]
            T = var[7]
            ft = self.total_molar_t(F_i)
            y_i = self.molar_fraction(F_i)
            r_i = self.rate_expression(T, F_i)
            Cp = self.species.specific_heat(T, y_i)
            Hf = self.species.heat_of_formation(T)
            dFdz = self.Nt * self.A * self.p_cat * (1 - self.e_cat) * r_i
            dTdz = (jnp.sum(self.n_i * r_i[0:5] * -Hf) * self.p_cat * self.a + self.U_shell * 4 / self.D_i *
                    (self.T_shell - T)) / ft / Cp * self.A * self.Nt * (1 - self.e_cat)
            return jnp.concatenate((dFdz, dTdz[None]))

        sol = jodeint(pfr, var_0, Z)
        return sol

# class containing functions describing separator model
class Separator:
    species = Species()

    def __init__(self, no_loop_steps=20, no_RR_loop_steps=20):
        self.no_loop_steps = no_loop_steps
        self.no_RR_loop_steps = no_RR_loop_steps

    def separator_loop(self, F_i, T, P):
        z_i = self.molar_fraction(F_i)
        K_i_a = self.species.K_values_ideal(T, P)
        F = self.total_flowrate(F_i)
        V_a = self.rachford_rice_call(F_i, K_i_a, jnp.array([0.9 * F]))
        carry0 = jnp.concatenate((K_i_a, V_a))

        def separator_inner_loop(carry, x):  # loop to iteratively find K and V
            info_of_interest = carry, x
            K_i_a = carry[0:7]
            V_a = carry[7]
            L = F - V_a
            x_i = F * z_i / (L + V_a * K_i_a)
            y_i = K_i_a * x_i
            K_i_b = self.species.K_values(T, P, y_i, x_i)
            V_b = self.rachford_rice_call(F_i, K_i_a, V_a)
            K_i_b = jnp.reshape(K_i_b, (7))
            carry = jnp.concatenate((K_i_b, jnp.array([V_b])))
            return carry, info_of_interest

        xs = jnp.arange(self.no_loop_steps)
        x_final, info_along_the_way = jax.lax.scan(separator_inner_loop, carry0, xs)
        K_i = x_final[0:7]
        V = x_final[7]
        return K_i, V, info_along_the_way

    def rachford_rice_call(self, F_i, K_i, V):  # loop to find V using Newton's method that leads to RR=0
        def rachford_rice_equation_intermofV(V):
            return self.rachford_rice_equation(F_i, K_i, V)

        def deriv_rachford_rice_equation_intermofV(V):
            return self.deriv_rachford_rice_equation(F_i, K_i, V)

        def rachfordrice_loop(V, n):
            info_of_interest = V, n
            F = self.total_flowrate(F_i)
            V_new = (V / F + rachford_rice_equation_intermofV(V) / deriv_rachford_rice_equation_intermofV(V)) * F
            return V_new, info_of_interest

        V0 = jnp.array([V])
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
        K_i, V, info = self.separator_loop(F_i, T, P)
        K_i = jnp.reshape(K_i, (7))
        F = self.total_flowrate(F_i)
        z_i = self.molar_fraction(F_i)
        L = F - V
        x_i = F * z_i / (L + V * K_i)
        y_i = x_i * K_i
        F_vap = y_i * V
        F_liq = x_i * L
        return F_vap, F_liq

    def molar_fraction(self, F_i):
        y_i = F_i / jnp.sum(F_i)
        return y_i

    def total_flowrate(self, F_i):
        return jnp.sum(F_i)

# class containing functions describing splitting or mixing of streams
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
    species = Species()
    MW_i = jnp.array([28.0097, 44.0087, 32.04106, 2.01568, 18.01468, 16.04206,
                      28.0134])  # molecular weights (CO, CO2, CH3OH, H2, H20, CH4, N2)
    F_i0 = jnp.array([10727.9, 23684.2, 756.7, 9586.5, 108.8, 4333.1,
                      8071.9]) / 3600 * 1000 / MW_i  # mol/s, molar flow of species
    hours_in_year = 8760.0
    seconds_in_year = 3600 * hours_in_year
    feed_carbon_content = F_i0[0] + F_i0[1] # sum of CO and CO2 molar flowrate
    R = 8.314472  # J/mol.K
    T1 = jnp.array([480.0])  # temperature of feed
    U_hx = 850.0  # heat transfer coefficient of heat exchangers
    T_MPS = jnp.array([648.0])  # temperature of MPS utility
    T_CW_in = jnp.array([303.0])  # temperature in of CW utility
    T_CW_out = jnp.array([318.0])  # temperature out of CW utility

    def __init__(self, epsilon=10**-6):
        self.epsilon = epsilon  # epsilon (step size) for ND

    @partial(jax.jit, static_argnums=(0))
    def objective_function(self, x):  # methanol production obj func
        F6 = x[35:42]
        output = -F6[2]
        return output

    @partial(jax.jit, static_argnums=(0))
    def objective_function_cost(self, x):  # cost obj func
        F2 = x[0:7]
        F4 = x[7:14]
        F7 = x[14:21]
        F8 = x[21:28]
        F9 = x[28:35]
        F6 = x[35:42]
        F3 = F2
        F5 = F4
        F10 = F9
        T_reactor = jnp.array([x[42]])
        T_separator = jnp.array([x[43]])
        P_separator = jnp.array([x[44]])
        ratio_CO_CO2 = jnp.array([x[46]])
        split_factor = jnp.array([x[47]])
        T_reactor_out = jnp.array([x[45]])

        F1 = jnp.array([self.feed_carbon_content * ratio_CO_CO2[0], self.feed_carbon_content * (1 - ratio_CO_CO2)[0],
                        self.F_i0[2], self.F_i0[3], self.F_i0[4], self.F_i0[5], self.F_i0[6]])
        T_reactor_average = (T_reactor + T_reactor_out) / 2
        F2_y_i = self.molar_fraction(F2)
        Cp_reactor = self.species.specific_heat(T_reactor_average, F2_y_i)
        F1_fraction_F2 = self.total_molar(F1) / self.total_molar(F2)
        F10_fraction_F2 = self.total_molar(F10) / self.total_molar(F2)
        T_2 = F1_fraction_F2 * self.T1 + F10_fraction_F2 * T_separator
        Cp_2 = self.species.specific_heat(T_2, F2_y_i)
        F4_y_i = self.molar_fraction(F4)
        T_preflash_cooler_average = (T_reactor_out + T_separator) / 2
        Cp_preflash_cooler = self.species.specific_heat(T_preflash_cooler_average, F4_y_i)

        methanol_income = F6[2] * 0.021 * self.seconds_in_year
        purge_income = self.total_molar(F8) * 0.331 * 6 * 10 ** -9 * self.seconds_in_year
        steam_income = self.total_molar(F3) * Cp_reactor * (
                T_reactor_out - T_reactor) * 6 * 10 ** -9 * self.seconds_in_year
        compressor_energy_cost = self.total_molar(F9) * self.R * T_separator * (
                (self.reactor.P / P_separator) ** 0.2908 - 1) / \
                                 0.2908 * 16.8 / 277778 * self.hours_in_year
        prereactor_heater_energy_cost = self.total_molar(F2) * Cp_2 * (
                T_reactor - T_2) * 8.22 * 10 ** -9 * self.seconds_in_year
        preflash_cooler_energy_cost = self.total_molar(F4) * Cp_preflash_cooler * (
                T_reactor_out - T_separator) * 0.44 * 10 ** -9 * self.seconds_in_year
        catalyst_capital_cost = 10 * self.reactor.p_cat * self.reactor.A * self.reactor.Nt * self.reactor.z * self.reactor.e_cat
        prereactor_heater_capital_cost = 7296 * (self.total_molar(F2) * Cp_2 * (T_reactor - T_2) / (
                self.U_hx * 1.0 * self.lmtd(self.T_MPS, self.T_MPS, T_2, T_reactor))) ** 0.65
        preflash_cooler_capital_cost = 7296 * (
                self.total_molar(F4) * Cp_preflash_cooler * (T_reactor_out - T_separator) / (
                self.U_hx * 0.9 * self.lmtd(T_reactor_out, T_separator, self.T_CW_in, self.T_CW_out))) ** 0.65
        compressor_capital_cost = 1293 * 517.3 * 3.11 * (1 / 745.7 * self.total_molar(F9) * self.R * T_separator * (
                (self.reactor.P / P_separator) ** 0.2908 - 1) / \
                                                         0.2908) ** 0.82 / 280
        income = methanol_income + purge_income + steam_income
        operating_expenses = compressor_energy_cost + prereactor_heater_energy_cost + preflash_cooler_energy_cost
        capital_costs = catalyst_capital_cost + compressor_capital_cost + prereactor_heater_capital_cost + preflash_cooler_capital_cost  # + reactor_capital_cost
        TAC = income - operating_expenses - capital_costs / 3
        return -TAC[0]

    @partial(jax.jit, static_argnums=(0))
    def grad_objective_function_cost(self, x):
        return grad(self.objective_function_cost)(x)

    def num_grad_objective_function_cost(self, x):
        return scipy.optimize.approx_fprime(x, self.objective_function_cost, self.epsilon)

    @partial(jax.jit, static_argnums=(0))
    def equality_constraints(self, x):  # constraint - mass & energy balances across each unit
        F2 = x[0:7]
        F4 = x[7:14]
        F7 = x[14:21]
        F8 = x[21:28]
        F9 = x[28:35]
        F6 = x[35:42]
        F3 = F2
        F5 = F4
        F10 = F9
        T_reactor = jnp.array([x[42]])
        T_separator = jnp.array([x[43]])
        P_separator = jnp.array([x[44]])
        ratio_CO_CO2 = jnp.array([x[46]])
        split_factor = jnp.array([x[47]])
        T_reactor_out = jnp.array([x[45]])
        F1 = jnp.array([self.feed_carbon_content * ratio_CO_CO2[0], self.feed_carbon_content * (1 - ratio_CO_CO2)[0],
                        self.F_i0[2], self.F_i0[3], self.F_i0[4], self.F_i0[5], self.F_i0[6]])
        mixer_balance = F2 - self.mixsplit.mix(F1, F10)
        reactor_mass_balance = F4 - self.reactor.mass_energy_balance(T_reactor, F3)[-1, 0:7]
        reactor_e_balance = T_reactor_out - self.reactor.mass_energy_balance(T_reactor, F3)[-1, 7]
        F7_calc, F6_calc = self.separator.mass_balance(F5, T_separator, P_separator)
        separator_v_balance = F7 - F7_calc
        separator_l_balance = F6 - F6_calc
        F9_calc, F8_calc = self.mixsplit.split(F7, split_factor)
        splitter_balance_1 = F9 - F9_calc
        splitter_balance_2 = F8_calc - F8
        return jnp.concatenate((mixer_balance, reactor_mass_balance, separator_v_balance, separator_l_balance,
                                splitter_balance_1, splitter_balance_2, reactor_e_balance))

    @partial(jax.jit, static_argnums=(0))
    def grad_equality_constraints(self, x):
        output = jacrev(self.equality_constraints)(x)
        return output

    def num_grad_equality_constraints(self, x):
        output = nd.Jacobian(self.equality_constraints, step=self.epsilon)(x)
        return output

    @partial(jax.jit, static_argnums=(0))
    def grad_objective_function(self, x):
        output = grad(self.objective_function)(x)
        return output

    def num_grad_objective_function(self, x):
        output = scipy.optimize.approx_fprime(x, self.objective_function, self.epsilon)
        return output

    @partial(jax.jit, static_argnums=(0))
    def hess_objective_function(self, x):
        return jacrev(jacrev(self.objective_function))(x)

    @partial(jax.jit, static_argnums=(0))
    def hess_equality_constraints(self, x):
        return jacrev(self.grad_equality_constraints)(x)

    def stream_information(self, optimizer_result, method):  # function used to organise result of optimisation
        x = optimizer_result["x"]
        nfev = jnp.array([optimizer_result["nfev"]])
        nit = jnp.array([optimizer_result["nit"]])
        if optimizer_result["status"] == True:
            success = jnp.array([1])
        else:
            success = jnp.array([0])
        if "ipopt" in method:
            njev = jnp.array([optimizer_result["njev"]])
            optimizer_performance = pd.DataFrame({"no.function evaluations": nfev, "no.iterations": nit,
                                                  "no.Jacobean evaluations": njev, "success": success})
        else:
            optimizer_performance = pd.DataFrame({"no.function evaluations": nfev, "no.iterations": nit,
                                                  "success": success})
        F2 = x[0:7]
        F4 = x[7:14]
        F7 = x[14:21]
        F8 = x[21:28]
        F9 = x[28:35]
        F6 = x[35:42]
        F3 = F2
        F5 = F4
        F10 = F9
        T_reactor = jnp.array([x[42]])
        T_separator = jnp.array([x[43]])
        P_separator = jnp.array([x[44]])
        ratio_CO_CO2 = jnp.array([x[46]])
        split_factor = jnp.array([x[47]])
        T_reactor_out = jnp.array([x[45]])
        F1 = jnp.array([self.feed_carbon_content * ratio_CO_CO2[0], self.feed_carbon_content * (1 - ratio_CO_CO2)[0],
                        self.F_i0[2], self.F_i0[3], self.F_i0[4], self.F_i0[5], self.F_i0[6]])
        s_ratio = jnp.array([(F3[3] - F3[1]) / (F3[1] + F3[0])])
        stream_table = pd.DataFrame(
            {"F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6, "F7": F7, "F8": F8, "F9": F9, "F10": F10})
        operating_conditions = pd.DataFrame({"Reactor inlet temperature": T_reactor,
                                             "Separator temperature": T_separator, "Separator pressure": P_separator,
                                             "Fraction recycle": split_factor,
                                             "Feed fraction CO": ratio_CO_CO2, "s ratio": s_ratio, "T reactor out":
                                                 T_reactor_out})
        name_stream_table = "stream table - " + method
        name_operating_conditions = "operating conditions - " + method
        name_optimizer_performance = "optimizer performance - " + method
        stream_table.to_csv("%s.csv" % name_stream_table, index=False)
        operating_conditions.to_csv("%s.csv" % name_operating_conditions, index=False)
        optimizer_performance.to_csv("%s.csv" % name_optimizer_performance, index=False)
        return stream_table, operating_conditions, optimizer_performance

    def total_molar(self, F_i):
        return jnp.sum(F_i)

    def molar_fraction(self, F_i):
        y_i = F_i / jnp.sum(F_i)
        return y_i

    def lmtd(self, T_hot_in, T_hot_out, T_cold_in, T_cold_out):
        delta_Ti = T_hot_in - T_cold_out
        delta_Tj = T_hot_out - T_cold_in
        lmtd = (delta_Ti - delta_Tj) / jnp.log(delta_Ti / delta_Tj)
        return lmtd

    def F6_conversion(self, x):  # overall conversion
        F6 = x[35:42]
        carbon_in = self.F_i0[0] + self.F_i0[1] + self.F_i0[5] + self.F_i0[2]
        output = F6[2] / carbon_in
        return output

    def F4_conversion(self, x):
        F4 = x[7:14]
        carbon_in = self.F_i0[0] + self.F_i0[1] + self.F_i0[5] + self.F_i0[2]
        output = F4[2] / carbon_in
        return output

if __name__ == '__main__':
    process = Process(epsilon=10 **-8)
    reactor = Reactor()
    separator = Separator()
    species = Species()
    mixsplit = MixSplit()

    # Base Case Initial Guess - taken from BaseCase_and_UpperBound code
    x = jnp.array(
        [1.51638438e+02, 2.32481901e+02, 1.28277974e+01, 2.32651356e+03, 2.42826475e+00, 1.49902708e+02, 1.60015287e+02,
         9.05145757e+01, 1.68114085e+02, 1.38319476e+02, 2.01116239e+03, 6.67960808e+01,
         1.49902708e+02, 1.60015287e+02, 9.04954308e+01, 1.65979990e+02,
         1.25352771e+01, 2.01082532e+03, 1.50124123e+00, 1.49745027e+02,
         1.59950438e+02, 4.52477154e+01, 8.29899952e+01, 6.26763854e+00,
         1.00541266e+03, 7.50620617e-01, 7.48725133e+01, 7.99752192e+01,
         4.52477154e+01, 8.29899952e+01, 6.26763854e+00, 1.00541266e+03,
         7.50620617e-01, 7.48725133e+01, 7.99752192e+01, 1.91448682e-02,
         2.13409470e+00, 1.25784199e+02, 3.37069196e-01, 6.52948396e+01,
         1.57681585e-01, 6.48489600e-02, 500, 308, 60.0, 5.47420626e+02, 0.529, 0.5])
    bons = [(0, 5.12095147e+02), (0, 6.71121273e+02), (0, 5.48100450e+01), (0, 9.67574347e+03),
            (0, 8.86013966e+00), (0, 7.48512296e+02), (0, 7.99687990e+02), (0, 3.74186230e+02),
            (0, 4.68736806e+02), (0, 3.95103428e+02), (0, 8.79277224e+03), (0, 2.11244607e+02),
            (0, 7.48512296e+02), (0, 7.99687990e+02), (0, 3.74142127e+02), (0, 4.65171826e+02),
            (0, 5.21121591e+01), (0, 8.79192709e+03), (0, 6.88106424e+00), (0, 7.48064883e+02),
            (0, 7.99509818e+02), (0, 7.48284253e+01), (0, 9.30343651e+01), (0, 1.04224318e+01),
            (0, 1.75838542e+03), (0, 1.37621285e+00), (0, 1.49612977e+02), (0, 1.59901964e+02),
            (0, 2.99313701e+02), (0, 3.72137461e+02), (0, 4.16897273e+01), (0, 7.03354167e+03),
            (0, 5.50485139e+00), (0, 5.98451906e+02), (0, 6.39607854e+02), (0, 4.41037707e-02),
            (0, 3.56498002e+00), (0, 3.42991269e+02), (0, 8.45154629e-01), (0, 2.04363542e+02),
            (0, 4.47413281e-01), (0, 1.78172716e-01),
            (500, 583), (308, 323), (50, 75), (500, 583), (0.0, 1.0), (0.01, 0.8)]
    # Ipopt - methanol production
    cons_ipopt = [{'type': 'eq', 'fun': process.equality_constraints, 'jac': process.num_grad_equality_constraints}]
    cons_ipopt_jac = [{'type': 'eq', 'fun': process.equality_constraints, 'jac': process.grad_equality_constraints}]
    result_ipopt_jac = minimize_ipopt(process.objective_function, jac=process.grad_objective_function, x0=x,
                                      bounds=bons, constraints=cons_ipopt_jac, options={'disp': 5, "output_file":
            "Ipopt JAX File Print", "file_print_level": 8})
    result_ipopt = minimize_ipopt(process.objective_function, jac=process.num_grad_objective_function, x0=x,
                                  bounds=bons, constraints=cons_ipopt,
                                  options={'disp': 5, "output_file": "Ipopt File Print", "file_print_level": 8})
    S_table_ipopt, OC_table_ipopt, OP_table_ipopt = process.stream_information(result_ipopt, "ipopt")
    S_table_ipopt_jac, OC_table_ipopt_jac, OP_table_ipopt_jac = process.stream_information(result_ipopt_jac,
                                                                                           "ipopt with jac")
    print(result_ipopt)
    print(result_ipopt_jac)
    print(process.F4_conversion(result_ipopt_jac["x"]))
    print(process.F6_conversion(result_ipopt_jac["x"]))
    # Ipopt - cost
    result_ipopt_jac = minimize_ipopt(process.objective_function_cost, jac=process.grad_objective_function_cost, x0=x,
                                      bounds=bons, constraints=cons_ipopt_jac, options={'disp': 5, "output_file":
            "Ipopt JAX File Print", "file_print_level": 8})
    result_ipopt = minimize_ipopt(process.objective_function_cost, jac=process.num_grad_objective_function_cost, x0=x,
                                  bounds=bons, constraints=cons_ipopt,
                                  options={'disp': 5, "output_file": "Ipopt File Print", "file_print_level": 8})
    S_table_ipopt, OC_table_ipopt, OP_table_ipopt = process.stream_information(result_ipopt, "ipopt")
    S_table_ipopt_jac, OC_table_ipopt_jac, OP_table_ipopt_jac = process.stream_information(result_ipopt_jac,
                                                                                           "ipopt with jac")
    print(result_ipopt)
    print(result_ipopt_jac)
    print(process.F6_conversion(result_ipopt_jac["x"]))
    print(process.F4_conversion(result_ipopt_jac["x"]))

    # S_table_DE, OC_table_DE, OP_table_DE = process.stream_information(result_DE, "differential evolution")
    # history_DE = jnp.asarray(callback.hist)
    # plt.figure()
    # plt.plot(history_DE[:, 0:3], 'o', markersize=1 * 0.5)
    # plt.xlabel("Iteration")
    # plt.ylabel("Temperature (K)/ Pressure (bar)")
    # plt.legend(["T reactor", "T separator", "P separator"])
    # plt.savefig("DE")
    #
    # plt.figure()
    # plt.plot(history_DE[:, 0], 'o', markersize=1 * 0.5)
    # plt.xlabel("Iteration")
    # plt.ylabel("Temperature (K)")
    # plt.savefig("DE T reactor")
    #
    # # # reactor = Reactor()
    # # # initial_temperature = jnp.array([523.0])
    # # # flow_reactor_out = reactor.mass_energy_balance(initial_temperature, process.F_i0)[-1, 0:7]
    # # separator = Separator()
    # T_separator = jnp.array([330.0])
    # P_separator = jnp.array([60.0])
    #
    # flow_coco = jnp.array([386.45788, 1279.0561, 831.93083, 4372.4646, 133.64492, 423.92416, 129.28562])
    # print(grad(separator.rachford_rice_equation)(T_separator, P_separator,flow_coco, jnp.array([1398])))
    # print(jnp.sum(flow_coco))
    # v_param = separator.rachford_rice_call(flow_coco, T_separator, P_separator)
    # print(v_param)
    # # print(v_param)
    # print(separator.mass_balance(flow_coco, T_separator, P_separator))
    # print(separator.rachford_rice_equation(T_separator, P_separator, flow_coco, v_param))
    # #
    # # initial_temperature = jnp.array([523.0])  # K
    # reactor = Reactor()
    # sol = reactor.mass_energy_balance(initial_temperature, process.F_i0)
    # Z = jnp.linspace(0.0, reactor.z, reactor.no_ode_steps)
    #
    # # Temperature vs length plots
    # plt.figure()
    # plt.plot(Z, sol[:, 7], 'ro', markersize=1 * 0.5)
    # plt.xlabel("Length of reactor (m)")
    # plt.ylabel("Temperature (K)")
    # plt.savefig("T vs length")
    # #
    # # Methanol vs length plots
    # plt.figure()
    # plt.plot(Z, sol[:, 2])
    # plt.xlabel("reactor length (m)")
    # plt.ylabel("Methanol molar flow (mol/s)")
    # plt.savefig("CH3OH mole flow vs length")

    # optimization
    # start_point = jnp.array([540.0])
    # T_values = reactor.loop(start_point)
    # methanol_flowrate = vmap(reactor.objective_function)(jnp.array(T_values)) * -1
    # methanol_flowrate = jnp.reshape(methanol_flowrate, [1, reactor.no_optim_steps])*-1

    # fig, axs = plt.subplots(2)
    # fig.suptitle('Flow/T vs length')
    # axs[0].plot(jnp.linspace(0,reactor.no_optim_steps, reactor.no_optim_steps),methanol_flowrate)
    # axs[0].set(ylabel = "Methanol flowrate out (mol/S)")
    # axs[1].plot(jnp.linspace(0,reactor.no_optim_steps, reactor.no_optim_steps),T_values)
    # plt.xlabel("iteration")
    # axs[1].set(ylabel = "Temperature in (K)")
    # plt.savefig("flowrate vs iteration")

    # F_i = jnp.array([51.019131, 148.22169, 177.31822, 1914.9441, 56.058626, 149.78939, 159.95742])
    # T = jnp.array([330.0])
    # P = jnp.array([60.0])
    # y_i = jnp.array([0.020771939, 0.0595115259, 0.01268075, 0.77971855, 0.0013109827, 0.060903278, 0.065102974])
    # x_i = jnp.array([0.00011514431, 0.010469182, 0.72283337, 0.0033709576, 0.26127321, 0.0013214924, 0.00061663844])
    # x = jnp.array([520.0, 330.0, 60, 0.7, 0.3, 3.9295303, 15.028284, 3.1656873, 174.12167, 0.28198171, 13.202387, 14.106927])    F_i = jnp.array([51.019131, 148.22169, 177.31822, 1914.9441, 56.058626, 149.78939, 159.95742])
    # T = jnp.array([330.0])
    # P = jnp.array([60.0])
    # y_i = jnp.array([0.020771939, 0.0595115259, 0.01268075, 0.77971855, 0.0013109827, 0.060903278, 0.065102974])
    # x_i = jnp.array([0.00011514431, 0.010469182, 0.72283337, 0.0033709576, 0.26127321, 0.0013214924, 0.00061663844])
    # x = jnp.array([520.0, 330.0, 60, 0.7, 0.3, 3.9295303, 15.028284, 3.1656873, 174.12167, 0.28198171, 13.202387, 14.106927])
