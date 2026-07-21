import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

output_folder = sys.argv[1]
file_path = os.path.join(output_folder, "Convergence_Data", "relative_residue.dat")

def process_residues(input_file):
    # Read the residue file
    df = pd.read_csv(input_file, sep=r"\s+")

    # Create the plot
    plt.figure(figsize=(12, 7))

    residuals_to_plot = [
        "DENSITY_RESIDUE",
        "X-MOMENTUM_RESIDUE",
        "Y-MOMENTUM_RESIDUE",
        "Z-MOMENTUM_RESIDUE",
        "ENERGY_RESIDUE",
    ]

    for column in residuals_to_plot:
        if column in df.columns:
            plt.plot(df["ITERATION"], df[column],
                     label=column, linewidth=1.5)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Relative Residue")
    plt.title("Solver Convergence History")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save the plot in the folder passed through argv[1]
    save_path = os.path.join(output_folder, "convergence_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()

# Execute
process_residues(file_path)
