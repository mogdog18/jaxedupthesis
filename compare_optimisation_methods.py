import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from JAXedThesisMainCode import Reactor, Separator, Species, Process, MixSplit
from cyipopt import minimize_ipopt

process = Process()
reactor = Reactor()
separator = Separator()
species = Species()
mixsplit = MixSplit()

# BaseCase conditions taken from BaseCase_and_UpperBound code
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

if __name__ == '__main__':
    # Ipopt - Cost Objective Function
    cons_ipopt = [{'type': 'eq', 'fun': process.equality_constraints, 'jac': process.num_grad_equality_constraints}]
    cons_ipopt_jac = [{'type': 'eq', 'fun': process.equality_constraints, 'jac': process.grad_equality_constraints}]
    result_ipopt_jac_cost = minimize_ipopt(process.objective_function_cost, jac=process.grad_objective_function_cost,
                                           x0=x,
                                           bounds=bons, constraints=cons_ipopt_jac, options={'disp': 5, "output_file":
            "Ipopt JAX File Print", "file_print_level": 8})
    result_ipopt_cost = minimize_ipopt(process.objective_function_cost, jac=process.num_grad_objective_function_cost,
                                       x0=x,
                                       bounds=bons, constraints=cons_ipopt,
                                       options={'disp': 5, "output_file": "Ipopt File Print", "file_print_level": 8})
    S_table_ipopt_cost, OC_table_ipopt_cost, OP_table_ipopt_cost = process.stream_information(result_ipopt_cost,
                                                                                              "ipopt cost")
    S_table_ipopt_jac_cost, OC_table_ipopt_jac_cost, OP_table_ipopt_jac_cost = process.stream_information(
        result_ipopt_jac_cost,
        "ipopt with jac cost")
    # Ipopt - methanol production
    Process = Process()
    result_ipopt_jac = minimize_ipopt(process.objective_function, jac=process.grad_objective_function, x0=x,
                                      bounds=bons, constraints=cons_ipopt_jac, options={'disp': 5, "output_file":
            "Ipopt JAX File Print", "file_print_level": 8})
    result_ipopt = minimize_ipopt(process.objective_function, jac=process.num_grad_objective_function, x0=x,
                                  bounds=bons, constraints=cons_ipopt,
                                  options={'disp': 5, "output_file": "Ipopt File Print", "file_print_level": 8})
    S_table_ipopt, OC_table_ipopt, OP_table_ipopt = process.stream_information(result_ipopt, "ipopt")
    S_table_ipopt_jac, OC_table_ipopt_jac, OP_table_ipopt_jac = process.stream_information(result_ipopt_jac,
                                                                                           "ipopt with jac")
    print(result_ipopt_jac_cost)
    print(result_ipopt_cost)
    print(result_ipopt_jac)
    print(result_ipopt)
    labels = ["iterations", "function evaluations", "Jacobian evaluations"]
    x_pos = jnp.arange(len(labels))
    fig, ax = plt.subplots()
    width = 0.35
    ND_data = [OP_table_ipopt["no.iterations"][0], OP_table_ipopt["no.function evaluations"][0],
               OP_table_ipopt["no.Jacobean evaluations"][0]]
    print(ND_data)
    AD_data = [OP_table_ipopt_jac["no.iterations"][0], OP_table_ipopt_jac["no.function evaluations"][0],
               OP_table_ipopt_jac["no.Jacobean evaluations"][0]]
    print(AD_data)
    ND_data_cost = [OP_table_ipopt_cost["no.iterations"][0], OP_table_ipopt_cost["no.function evaluations"][0],
                    OP_table_ipopt_cost["no.Jacobean evaluations"][0]]
    print(ND_data_cost)
    AD_data_cost = [OP_table_ipopt_jac_cost["no.iterations"][0], OP_table_ipopt_jac_cost["no.function evaluations"][0],
                    OP_table_ipopt_jac_cost["no.Jacobean evaluations"][0]]
    print(AD_data_cost)

    fig, ax = plt.subplots()
    ax.bar(x_pos, ND_data, width, alpha=0.5, color='#E182FD')
    ax.bar(x_pos + width, AD_data, width, alpha=0.5, color='#6E98F8')
    ax.set_xticks((x_pos + width / 2))
    ax.set_xticklabels(labels, fontsize=11)
    plt.legend(["Numerical", "Automatic"], loc="best")
    ax.set_ylabel("Number", fontsize=11)
    plt.savefig("num_vs_ad_OP - methanol.svg", format='svg', dpi=1200)

    fig, ax = plt.subplots()
    ax.bar(x_pos, ND_data_cost, width, alpha=0.5, color='#E182FD')
    ax.bar(x_pos + width, AD_data_cost, width, alpha=0.5, color='#6E98F8')
    ax.set_xticks((x_pos + width / 2))
    ax.set_xticklabels(labels, fontsize=11)
    plt.legend(["Numerical", "Automatic"], loc="best")
    ax.set_ylabel("Number", fontsize=11)
    plt.savefig("num_vs_ad_OP - cost.svg", format='svg', dpi=1200)

    labels = ["methanol", "cost"]
    x_pos = jnp.arange(len(labels))
    fig, ax = plt.subplots()
    ND_times = [10.683, 326.028]
    AD_times = [5.035, 44.395]
    width = 0.35
    ax.bar(x_pos, ND_times, width, alpha=0.5, color='#E182FD')
    ax.bar(x_pos + width, AD_times, width, alpha=0.5, color='#6E98F8')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    plt.legend(["Numerical", "Automatic"], loc="best")
    ax.set_ylabel("Total time taken (s)", fontsize=11)
    plt.savefig("num_vs_ad_time.svg", format='svg', dpi=1200)
