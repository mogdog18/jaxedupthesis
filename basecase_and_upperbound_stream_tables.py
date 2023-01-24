import jax
import jax.numpy as jnp
from scipy.optimize import differential_evolution, dual_annealing, minimize, NonlinearConstraint, Bounds, fsolve
from jax.config import config
import numpy as np
config.update("jax_enable_x64", True)
import pandas as pd
from models import Reactor, Separator, Species, Process, MixSplit

# code containing functions to determine the base case and value of the upper bound of recycle ratio

MW_i = jnp.array([28.0097, 44.0087, 32.04106, 2.01568, 18.01468, 16.04206, 28.0134])  # molecular weights (CO, CO2,
# CH3OH, H2, H20, CH4, N2)
F1 = jnp.array(
    [10727.9, 23684.2, 756.7, 9586.5, 108.8, 4333.1, 8071.9]) / 3600 * 1000 / MW_i  # mol/s, molar flow of species
process = Process()
mixsplit = MixSplit()
separator = Separator()
reactor = Reactor()

def massbalance_process(x):
    F2 = x[0:7]
    F4 = x[7:14]
    F7 = x[14:21]
    F8 = x[21:28]
    F9 = x[28:35]
    F6 = x[35:42]
    F3 = F2
    F5 = F4
    F10 = F9
    T_reactor_out = jnp.array([x[42]])
    mixer_balance = F2 - mixsplit.mix(F1, F10)
    reactor_mass_balance = F4 - reactor.mass_energy_balance(T_reactor, F3)[-1, 0:7]
    reactor_e_balance = T_reactor_out - reactor.mass_energy_balance(T_reactor, F3)[-1, 7]
    F7_calc, F6_calc = separator.mass_balance(F5, T_separator, P_separator)
    separator_v_balance = F7 - F7_calc
    separator_l_balance = F6 - F6_calc
    F9_calc, F8_calc = mixsplit.split(F7, split_factor)
    splitter_balance_1 = F9 - F9_calc
    splitter_balance_2 = F8_calc - F8
    return jnp.concatenate((mixer_balance, reactor_mass_balance, separator_v_balance, separator_l_balance,
                            splitter_balance_1, splitter_balance_2, reactor_e_balance))

if __name__ == '__main__':
    # Base Case
    T_reactor = jnp.array([500])
    T_separator = jnp.array([308])
    P_separator = jnp.array([60])
    ratio_CO_CO2 = jnp.array([0.529])
    split_factor = jnp.array([0.5])
    x0 = jnp.array([153.57546, 200.4892, 16.838237, 2277.6904, 2.6612953, 149.79524, 159.96206,
                    51.289886, 147.46577, 172.14724, 1914.049, 55.684727, 149.79524, 159.96206,
                    51.268206, 145.09568, 20.556532, 1913.4411, 1.9674096, 149.53383, 159.84199,
                    25.634103, 72.547839, 10.278266, 956.72054, 0.98370482, 74.766917, 79.920993,
                    25.634103, 72.547839, 10.278266, 956.72054, 0.98370482, 74.766917, 79.920993,
                    0.021679606, 2.3700932, 151.59071, 0.60791527, 53.717317, 0.26140885, 0.12006883, 512])
    x = fsolve(massbalance_process, x0)
    F2 = x[0:7]
    F4 = x[7:14]
    F7 = x[14:21]
    F8 = x[21:28]
    F9 = x[28:35]
    F6 = x[35:42]
    F3 = F2
    F5 = F4
    F10 = F9
    T_reactor_out = x[42]
    stream_table_BC = pd.DataFrame(
        {"F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6, "F7": F7, "F8": F8, "F9": F9, "F10": F10})
    stream_table_BC.to_csv("BaseCase_StreamTable.csv", index=False)
    print(x)
    x0 = np.ones([48])
    x0[0:42] = x[0:42]
    x0[42] = T_reactor
    x0[43] = T_separator
    x0[44] = P_separator
    x0[45] = x[42]
    x0[46] = ratio_CO_CO2
    x0[47] = split_factor
    print(x0)
    print(process.objective_function(x0))
    print(process.objective_function_cost(x0))
    conversion = F6[2]/(F1[0] + F1[1] + F1[2] + F1[5])
    print(conversion)

    # Upper Bound
    split_factor = jnp.array([0.8])
    x0 = jnp.array([221.67304, 298.6591, 41.420616, 4764.6484, 5.7503546, 373.87905, 399.6497,
                    117.18886, 215.18574, 229.37815, 4305.26, 89.223709, 373.87905, 399.6497,
                    117.16545, 213.39836, 43.576111, 4304.5878, 5.0908618, 373.56338, 399.51079,
                    93.732357, 170.71869, 34.860889, 3443.6703, 4.0726894, 298.85071, 319.60863,
                    93.732357, 170.71869, 34.860889, 3443.6703, 4.0726894, 298.85071, 319.60863,
                    0.023418158, 1.7873847, 185.80204, 0.67218755, 84.132848, 0.31566401, 0.12006883, 0.13891171])
    x = fsolve(massbalance_process, x0)
    UB = x*2.0
    print(UB)
    F2 = x[0:7]
    F4 = x[7:14]
    F7 = x[14:21]
    F8 = x[21:28]
    F9 = x[28:35]
    F6 = x[35:42]
    F3 = F2
    F5 = F4
    F10 = F9
    stream_table_UB = pd.DataFrame(
        {"F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6, "F7": F7, "F8": F8, "F9": F9, "F10": F10})
    stream_table_UB.to_csv("UpperBound_StreamTable.csv", index=False)

