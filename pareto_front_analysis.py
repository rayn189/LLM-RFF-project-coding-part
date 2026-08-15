from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


EXCEL_PATH = Path(r"C:\Users\RAYNES\Desktop\SURF\deployment_summary\ppo_epochs.xlsx")
OUT_DIR = EXCEL_PATH.parent / "pareto_front_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_excel(EXCEL_PATH, sheet_name="ppo_epochs")
    required = ["epoch", "val_acc", "delay_ms_per_sample", "params_m", "complexity_score"]
    df = df.dropna(subset=required).copy()
    return df


def pareto_front(df, x_col, y_col, maximize_x=True, maximize_y=True):
    """Return non-dominated points for a 2D objective pair."""
    rows = []
    for i, a in df.iterrows():
        dominated = False
        for j, b in df.iterrows():
            if i == j:
                continue
            better_or_equal_x = b[x_col] >= a[x_col] if maximize_x else b[x_col] <= a[x_col]
            better_or_equal_y = b[y_col] >= a[y_col] if maximize_y else b[y_col] <= a[y_col]
            strictly_better = (b[x_col] > a[x_col] if maximize_x else b[x_col] < a[x_col]) or (
                b[y_col] > a[y_col] if maximize_y else b[y_col] < a[y_col]
            )
            if better_or_equal_x and better_or_equal_y and strictly_better:
                dominated = True
                break
        if not dominated:
            rows.append(a)

    exact_front = pd.DataFrame(rows)
    if not exact_front.empty:
        exact_front = exact_front.sort_values(x_col, ascending=not maximize_x).reset_index(drop=True)
    return exact_front


def pareto_front_3d(df, objectives):
    """Return the true 3D non-dominated set.

    Each objective is a tuple of (column_name, maximize_bool).
    A point is dominated if another point is at least as good in all
    objectives and strictly better in at least one objective.
    """
    rows = []
    for i, a in df.iterrows():
        dominated = False
        for j, b in df.iterrows():
            if i == j:
                continue
            better_or_equal = True
            strictly_better = False
            for col, maximize in objectives:
                if maximize:
                    if b[col] < a[col]:
                        better_or_equal = False
                        break
                    if b[col] > a[col]:
                        strictly_better = True
                else:
                    if b[col] > a[col]:
                        better_or_equal = False
                        break
                    if b[col] < a[col]:
                        strictly_better = True
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            rows.append(a)

    front = pd.DataFrame(rows)
    if not front.empty:
        front = front.sort_values(["val_acc", "delay_ms_per_sample", "complexity_score"], ascending=[False, True, True]).reset_index(drop=True)
    return front


def plot_front(df, front_df, x_col, y_col, xlabel, ylabel, title, filename, x_max=False, y_max=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], s=28, alpha=0.65, label="All PPO epochs")

    if not front_df.empty:
        plt.scatter(front_df[x_col], front_df[y_col], s=48, color="crimson", label="Pareto front")
        ordered = front_df.sort_values(x_col, ascending=not x_max)
        plt.plot(ordered[x_col], ordered[y_col], color="crimson", linewidth=2)
        for _, row in ordered.iterrows():
            plt.annotate(
                int(row["epoch"]),
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color="crimson",
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved {out_path}")


def select_knee_point(front_df):
    """Select the knee point on the 3D Pareto front via normalized distance to the ideal point.

    Higher val_acc is better; lower delay/complexity are better.
    The ideal point after normalization is (1, 1, 1).
    """
    if front_df.empty:
        return None

    df = front_df.copy().reset_index(drop=True)
    acc = df["val_acc"].astype(float)
    delay = df["delay_ms_per_sample"].astype(float)
    comp = df["complexity_score"].astype(float)

    def normalize_max(series):
        min_v = series.min()
        max_v = series.max()
        if max_v == min_v:
            return pd.Series([1.0] * len(series), index=series.index)
        return (series - min_v) / (max_v - min_v)

    def normalize_min(series):
        min_v = series.min()
        max_v = series.max()
        if max_v == min_v:
            return pd.Series([1.0] * len(series), index=series.index)
        return (max_v - series) / (max_v - min_v)

    acc_n = normalize_max(acc)
    delay_n = normalize_min(delay)
    comp_n = normalize_min(comp)

    distances = ((1.0 - acc_n) ** 2 + (1.0 - delay_n) ** 2 + (1.0 - comp_n) ** 2) ** 0.5
    knee_idx = distances.idxmin()
    knee_row = df.loc[knee_idx].copy()
    knee_row["knee_distance"] = float(distances.loc[knee_idx])
    knee_row["acc_norm"] = float(acc_n.loc[knee_idx])
    knee_row["delay_norm"] = float(delay_n.loc[knee_idx])
    knee_row["comp_norm"] = float(comp_n.loc[knee_idx])
    return knee_row


def plot_front_3d(df, front_df, knee_point, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        df["val_acc"],
        df["delay_ms_per_sample"],
        df["complexity_score"],
        s=20,
        alpha=0.35,
        color="#4C78A8",
        label="All PPO epochs",
    )
    if not front_df.empty:
        ax.scatter(
            front_df["val_acc"],
            front_df["delay_ms_per_sample"],
            front_df["complexity_score"],
            s=70,
            color="crimson",
            label="3D Pareto front",
        )
        ordered = front_df.sort_values(["val_acc", "delay_ms_per_sample", "complexity_score"], ascending=[False, True, True])
        ax.plot(
            ordered["val_acc"],
            ordered["delay_ms_per_sample"],
            ordered["complexity_score"],
            color="crimson",
            linewidth=2.5,
        )
        for _, row in ordered.iterrows():
            ax.text(
                row["val_acc"],
                row["delay_ms_per_sample"],
                row["complexity_score"],
                str(int(row["epoch"])),
                fontsize=8,
                color="crimson",
            )
    if knee_point is not None:
        ax.scatter(
            [knee_point["val_acc"]],
            [knee_point["delay_ms_per_sample"]],
            [knee_point["complexity_score"]],
            s=160,
            color="gold",
            edgecolor="black",
            marker="*",
            label="Knee point",
            depthshade=False,
        )
        ax.text(
            knee_point["val_acc"],
            knee_point["delay_ms_per_sample"],
            knee_point["complexity_score"],
            f"K{int(knee_point['epoch'])}",
            fontsize=10,
            color="black",
        )
    ax.set_xlabel("Accuracy (val_acc)")
    ax.set_ylabel("Delay per sample (ms)")
    ax.set_zlabel("Complexity")
    ax.set_title("3D Pareto Scatter: Accuracy-Delay-Complexity")
    ax.view_init(elev=28, azim=-58)
    ax.legend()
    plt.tight_layout()
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = load_data()

    # If params_m is constant, prefer complexity_score for deployment Pareto.
    if df["params_m"].nunique() <= 1:
        print("params_m is constant across epochs; using complexity_score for Pareto plotting.")

    # Fig 1: accuracy vs complexity, accuracy maximize, complexity minimize
    front_ac = pareto_front(df, "val_acc", "complexity_score", maximize_x=True, maximize_y=False)
    plot_front(
        df,
        front_ac,
        "val_acc",
        "complexity_score",
        "Accuracy (val_acc)",
        "Complexity",
        "Pareto Front: Accuracy vs Complexity",
        "pareto_accuracy_complexity.png",
        x_max=True,
        y_max=False,
    )

    # Fig 2: accuracy vs delay, accuracy maximize, delay minimize
    front_ad = pareto_front(df, "val_acc", "delay_ms_per_sample", maximize_x=True, maximize_y=False)
    plot_front(
        df,
        front_ad,
        "val_acc",
        "delay_ms_per_sample",
        "Accuracy (val_acc)",
        "Delay per sample (ms)",
        "Pareto Front: Accuracy vs Delay",
        "pareto_accuracy_delay.png",
        x_max=True,
        y_max=False,
    )

    # Fig 3: complexity vs delay, both minimize
    front_cd = pareto_front(df, "complexity_score", "delay_ms_per_sample", maximize_x=False, maximize_y=False)
    plot_front(
        df,
        front_cd,
        "complexity_score",
        "delay_ms_per_sample",
        "Complexity",
        "Delay per sample (ms)",
        "Pareto Front: Complexity vs Delay",
        "pareto_complexity_delay.png",
        x_max=False,
        y_max=False,
    )

    # True 3D Pareto front: maximize accuracy, minimize delay, minimize complexity.
    front_3d = pareto_front_3d(
        df,
        [
            ("val_acc", True),
            ("delay_ms_per_sample", False),
            ("complexity_score", False),
        ],
    )
    knee_point = select_knee_point(front_3d)
    plot_front_3d(df, front_3d, knee_point, "pareto_3d_scatter.png")

    # Save fronts to CSV for traceability.
    front_ac.to_csv(OUT_DIR / "front_accuracy_complexity.csv", index=False)
    front_ad.to_csv(OUT_DIR / "front_accuracy_delay.csv", index=False)
    front_cd.to_csv(OUT_DIR / "front_complexity_delay.csv", index=False)
    front_3d.to_csv(OUT_DIR / "front_3d_candidates.csv", index=False)

    if knee_point is not None:
        pd.DataFrame([knee_point]).to_csv(OUT_DIR / "knee_point.csv", index=False)

    print("Pareto front counts:")
    print(f"  Accuracy-Complexity: {len(front_ac)}")
    print(f"  Accuracy-Delay: {len(front_ad)}")
    print(f"  Complexity-Delay: {len(front_cd)}")
    print(f"  3D Pareto front: {len(front_3d)}")
    if knee_point is not None:
        print("Knee point selected:")
        print(
            f"  epoch={int(knee_point['epoch'])}, val_acc={knee_point['val_acc']:.6f}, "
            f"delay_ms_per_sample={knee_point['delay_ms_per_sample']:.6f}, "
            f"complexity_score={knee_point['complexity_score']:.6f}, "
            f"knee_distance={knee_point['knee_distance']:.6f}"
        )
    print(f"Plots saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
