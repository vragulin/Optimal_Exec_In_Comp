# Optimal_Exec_In_Comp

## Description
This project aims to optimize execution strategies in competitive markets using advanced algorithms.

The analytics cover 
1) Optimal Execution
2) Best Response (to an Adversary) Strategies
3) Two Trader Equilibrium Strtegies.analytics for the Chriss(2024) series of papers

The theoretical framework is based on the Chriss(2024) series of papers. The main papers are:


*   Optimal Position Building Strategies in Competition:  https://arxiv.org/pdf/2409.03586
*  Position-Building in Competition with Real-World Constraints: TBA


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

1. Ensure that the simulation results are available in the `results` directory as specified in `config.py`.

2. Run the main script to generate the plots:
    ```sh
    python optimizer/2x5_StateSpace_const_plot.py
    ```

## Project Structure

- `optimizer/`: Main scripts used to solve trading strategy optimizations and generate plots for the Chriss(2024) papers.
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