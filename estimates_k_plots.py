import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from estimates_k_to_csv import generate_rows


def plot_k_results(
    epsilon_nominator_values,
    N_values,
    k_values,
    target_delta=1e-9,
    outdir=Path("."),
):
    outdir.mkdir(parents=True, exist_ok=True)

    for eps_nom in epsilon_nominator_values:
        epsilon = eps_nom / 100.0
        for N in N_values:
            print(f"[Plot] epsilon={epsilon:.2f}, N={N}")

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            plotted_any = False

            for idx, k_chunks in enumerate(k_values):
                ax = axes[idx]
                rows = generate_rows(epsilon, N, k_chunks, target_delta=target_delta)
                if not rows:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        fontsize=12,
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"k={k_chunks}")
                    ax.set_xlabel("k2")
                    ax.set_ylabel("k1")
                    continue

                rows.sort(key=lambda r: r[3])
                k2_values = [r[3] for r in rows]
                k1_values = [r[4] for r in rows]
                ax.plot(k2_values, k1_values, marker="o", markersize=3, linewidth=1.6)
                ax.set_title(f"k={k_chunks}")
                ax.set_xlabel("k2")
                ax.set_ylabel("Max k1")
                ax.grid(True, alpha=0.3)
                plotted_any = True

            if not plotted_any:
                plt.close(fig)
                print(f"⚠️  No curves for ε={epsilon:.2f}, N={N}")
                continue

            fig.suptitle(
                f"Max k1 vs k2 (epsilon={epsilon:.2f}, N={N})",
                fontsize=14,
                weight="bold",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            out_path = outdir / f"k_curves_eps{eps_nom}_N{N}.png"
            fig.savefig(out_path.as_posix(), dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    N_values = [2500, 5000, 10000, 100000]
    epsilon_nominator_values = [5, 10]
    k_values = [8, 16, 32]
    plot_k_results(epsilon_nominator_values, N_values, k_values, outdir=Path("."))
