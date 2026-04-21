import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def summary_exp4(df):

    # -----------------------------
    # 1. Define columns
    # -----------------------------
    param_cols = ["p", "k", "n", "sl"]   # 'su' is redundant
    method_cols = [c for c in df.columns if c not in param_cols+["su"]]

    # -----------------------------
    # 2. Best and worst method per row
    # -----------------------------
    df["best_method"] = df[method_cols].idxmax(axis=1)
    df["worst_method"] = df[method_cols].idxmin(axis=1)

    # -----------------------------
    # 3. Global ranking
    # -----------------------------
    best_counts = df["best_method"].value_counts()
    worst_counts = df["worst_method"].value_counts()

    print("\n=== METHODS WITH MOST WINS ===")
    print(best_counts)

    print("\n=== METHODS WITH MOST LOSSES ===")
    print(worst_counts)

    # -----------------------------
    # 4. Ranking by parameter
    # -----------------------------
    print("\n=== METHOD WINS CONDITIONED ON PARAMETERS ===")
    for p in param_cols:
        print(f"\n--- Dependence on parameter '{p}' ---")
        print(df.groupby(p)["best_method"].value_counts(normalize=True))

    # -----------------------------
    # 5. Heatmaps of mean performance
    # -----------------------------
    for p in param_cols:
        pivot = df.groupby(p)[method_cols].mean()

        plt.figure(figsize=(10, 6))
        plt.imshow(pivot, cmap="viridis", aspect="auto")
        plt.colorbar(label="Mean method value")
        plt.xticks(range(len(method_cols)), method_cols, rotation=90)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(f"Mean performance of methods by parameter '{p}'")
        plt.tight_layout()
        plt.savefig(f"output/experiment_4_heatmap_mean_by_{p}.png")
        plt.close()

    # -----------------------------
    # 6. Pairwise dominance matrix
    # -----------------------------
    pair_results = {}

    for m1 in method_cols:
        for m2 in method_cols:
            if m1 == m2:
                continue
            pair_results[(m1, m2)] = (df[m1] > df[m2]).mean()

    pair_df = pd.DataFrame.from_dict(pair_results, orient="index", columns=["P(m1 > m2)"])

    print("\n=== PAIRWISE METHOD DOMINANCE ===")
    print(pair_df.sort_values("P(m1 > m2)", ascending=False))

    # -----------------------------
    # 7. Plot: wins per method
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(best_counts.index, best_counts.values)
    plt.title("Number of wins per method")
    plt.ylabel("Wins")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("output/experiment_4_method_wins.png")
    plt.close()

    # -----------------------------
    # 8. Plot: losses per method
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(worst_counts.index, worst_counts.values, color="red")
    plt.title("Number of losses per method")
    plt.ylabel("Losses")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("output/experiment_4_method_losses.png")
    plt.close()

    # -----------------------------
    # 9. Dominance region maps
    # -----------------------------
    print("\n=== GENERATING DOMINANCE REGION MAPS ===")

    param_pairs = [
        ("p", "k"),
        ("p", "n"),
        ("p", "sl"),
        ("k", "n"),
        ("k", "sl"),
        ("n", "sl")
    ]

    for x_param, y_param in param_pairs:

        plt.figure(figsize=(8, 6))

        # Scatter plot: color = best method
        methods = df["best_method"].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
        color_map = dict(zip(methods, colors))

        for m in methods:
            subset = df[df["best_method"] == m]
            plt.scatter(subset[x_param], subset[y_param],
                        label=m, s=20, color=color_map[m])

        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.title(f"Dominance regions: best method by ({x_param}, {y_param})")
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"output/experiment_4_dominance_{x_param}_{y_param}.png")
        plt.close()

    # -----------------------------
    # 10. Count how many times each method equals k
    # -----------------------------
    equal_to_k_counts = (df[method_cols].eq(df["k"], axis=0)).sum()

    print("\n=== NUMBER OF TIMES EACH METHOD EQUALS 'k' ===")
    print(equal_to_k_counts)

    # Save to file
    equal_to_k_counts.to_csv("output/experiment_4_method_equal_to_k_counts.csv")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(equal_to_k_counts.index, equal_to_k_counts.values, color="purple")
    plt.title("Number of times each method equals 'k'")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("output/experiment_4_method_equal_to_k.png")
    plt.close()

    print("\nSummary report completed. Figures saved in 'output/' directory.")