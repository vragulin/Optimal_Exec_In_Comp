# Optimal_Exec_In_Comp
Optimal Execution/ Best Response / Two Trader Equilibrium analytics for the Chriss(2024) series of papers


1.  Optimal Position Building Strategies in Competition:  https://arxiv.org/pdf/2409.03586
2.  Position-Building in Competition with Real-World Constraints: TBA


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

- `optimizer/2x5_StateSpace_const_plot.py`: Main script to generate 2x5 plots using saved simulation results.
- `optimizer/2x5_StateSpace_plot_chg_N.py`: Script to generate plots comparing analytic solutions and Fourier approximations.
- `cost_function/`: Directory containing cost function approximation scripts.
- `fourier/`: Directory containing Fourier transformation scripts.
- `trading_funcs/`: Directory containing trading-related functions.
- `config.py`: Configuration file specifying locations and file name formats for results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License
. See the `LICENSE` file for details.