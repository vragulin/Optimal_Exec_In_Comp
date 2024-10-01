# Optimal_Exec_In_Comp

## Description
This project offers tools and analytics to optimize execution strategies in competitive markets using advanced algorithms based on Quadratic Programming.

The analytics cover 
1) Optimal Execution Strategies
2) Best Response (to an Adversary) Strategies
3) Two Trader Equilibrium Strategies solvers / simulations.

The theoretical framework is based on the Chriss(2024) series of papers. The main papers are:

[1] Optimal Position Building Strategies in Competition:  https://arxiv.org/pdf/2409.03586

[2] Position-Building in Competition with Real-World Constraints: https://arxiv.org/abs/2409.15459

[3] Position-Building in Competition with Realistic Transactions Costs:  (TBA)


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/vragulin/Optimal_Exec_In_Comp.git
    cd Optimal_Exec_In_Comp
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

The repository contains two sets of scripts.  The scripts in the `optimizer_paper/` directory were used to produce charts and analytics for the Chriss(2024)[2] paper and use the Scipy.optimize.minimize general solver to find the optimal trading strategies. 

The scripts in the `optimizer_qp/` directory have the same functionality but enable the use of a considerably faster specialized QP solver, DAQP (https://darnstrom.github.io/daqp/start/).

![](https://github.com/vragulin/Optimal_Exec_In_Comp/blob/main/readme_figures/QP_v_SciPy_solver_times_log.png)
![](https://github.com/vragulin/Optimal_Exec_In_Comp/blob/main/readme_figures/Speed_Ratio_QP_v_SciPy.png)

Both sets of scripts (optimizer_paper and optimizer_qp) have the same structure and can be run in the same way.  The main scripts are:
1. Caclulate the optimal response to a known trading strategy by an Adversary:
    ```sh
    python optimizer_paper/opt_cost.py
    python optimizer_paper/opt_cost_complet_plots_paper.py
    ```
![](https://github.com/vragulin/Optimal_Exec_In_Comp/blob/main/readme_figures/best-reponse-eager-completion_sigma3-4x4-grid.png)

2. Run two trader optimization scripts with and without constraints:
    ```sh
    python optimizer_paper/tte_optimizer.py
    python optimizer_paper/tte_optimizer_constraints.py
    python optimizer_paper/opt_cost_equil2trader_channel_eager_plots_paper.py
       ```
![](https://github.com/vragulin/Optimal_Exec_In_Comp/blob/main/readme_figures/tte_k25_l1.png)

3. Produce 2x5 plots of simulation data (the simulation data should have already been generated by one of the
    optimization scripts above and saved in the `results/` directory):
    ```sh
    python optimizer/2x5_StateSpace_plot.py
    ```

# Creating your own optimization problems and trading strategies

The set of trading strategies implemented is defined in the `optimizer/trading_funcs` directory. The main families are:
- risk-neutral
- risk-averse (based on sinh())
- eager (exponential)
- parabolic
- equilibrium (two-trader)

See Chriss(2024) papers for details.
Alternatively, your own bespoke trading strategy can be defined via a callback function.


## Project Structure

- `optimizer_paper/`: Main scripts used to solve trading strategy optimizations and generate plots for the Chriss(2024) papers.
- `optimizer_qp/`: Faster optimziation scripts using the QP solver.
- `lin_propagator/`: Linear Propagator model scripts (best response, etc.)
- `cost_function/`: Modules for calculating trading costs in competitive settings.
- `fourier/`: Directory containing helper functions for approximating trading trajectories with Fourier and Sine Series.
- `results/`: Directory containing simulation results.
- `disc_sin_transform/`: Examples of using the discrete sine transform to approximate trading trajectories.
- `solver_examples/`: Examples of using general-purpose and QP optimization solvers
- `bvp_solver_examples/`: Examples of using the boundary value problem solver to approximate trading trajectories.
- `Mathematica/`: Mathematica notebooks for testing sub-problems and testing the code.
- `Representation/`: (Beta-version) code for encapsulation trading strategies in a class
- `tests/`: Unit tests for the codebase (pytest).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes to vlad@manifoldinsights.co.uk.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License
. See the `LICENSE` file for details.
