"""
RealLift Core — Centralized class for Causal Inference Lift Measurement.

This module defines the RealLift class, which acts as the single entry point
for all data ingestion, date standardization, and pipeline operations.
"""

import pandas as pd
import numpy as np
import warnings


class DoEResult(dict):
    """
    Wrapper around the Design of Experiments result dict.

    Behaves exactly like a regular dict (full backward compatibility)
    but adds convenience methods for visualization.
    """

    def plot_power_analysis(
        self,
        scenario=0,
        durations=None,
        max_mde=0.05,
        n_points=200,
        power_target=0.80,
        alpha=0.05,
        figsize=(12, 7),
    ):
        """
        Power Analysis — Power vs Real Impact (%) for different durations.

        Plots how statistical power grows as the true effect size increases,
        with one curve per experiment duration.  A horizontal target line
        (default 80 %) and intersection markers show the MDE at each
        duration.

        Parameters
        ----------
        scenario : int
            Scenario index to use for the consolidated sigma / ac_factor.
        durations : list of int, optional
            Experiment durations (days) to plot.  Defaults to a sensible
            spread based on the ``experiment_days`` used in the DoE.
        max_mde : float
            Maximum MDE on the x-axis (default ``0.05`` = 5 %).
        n_points : int
            Resolution of each curve.
        power_target : float
            Target power threshold line (default ``0.80``).
        alpha : float
            Significance level (default ``0.05``).
        figsize : tuple
            Figure size.
        """
        from scipy.stats import norm
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        scenarios = self.get("scenarios", [])
        if not scenarios:
            print("  [plot_power_analysis] No scenarios found in the DoE result.")
            return
        if scenario >= len(scenarios):
            raise IndexError(
                f"Scenario {scenario} out of range (max {len(scenarios) - 1})"
            )

        s = scenarios[scenario]
        dur = s.get("duration")
        if dur is None:
            print("  [plot_power_analysis] No duration data in this scenario.")
            return

        cons_summary = dur.get("consolidated", {}).get("summary", {})
        sigma = cons_summary.get("sigma")
        ac_factor = cons_summary.get("ac_factor", 1.0)

        if sigma is None:
            print("  [plot_power_analysis] Could not extract sigma from consolidated duration.")
            return

        # ── Resolve durations to plot ────────────────────────────────────
        if durations is None:
            cluster_results = dur.get("cluster_results", [])
            if cluster_results:
                first_cr = cluster_results[0]
                curve = first_cr.get("power_curve", first_cr.get("mde_curve"))
                if curve is not None and "days" in curve.columns:
                    d_min = int(curve["days"].min())
                    d_max = int(curve["days"].max())
                else:
                    d_min, d_max = 21, 60
            else:
                d_min, d_max = 21, 60

            step = max(7, (d_max - d_min) // 4)
            durations = list(range(d_min, d_max + 1, step))
            if durations[-1] != d_max:
                durations.append(d_max)

        # ── Compute power curves ─────────────────────────────────────────
        z_alpha = norm.ppf(1 - alpha / 2)
        mde_range = np.linspace(0, max_mde, n_points)

        palette = [
            "#EF4444",  # red-500
            "#F59E0B",  # amber-500
            "#10B981",  # emerald-500
            "#06B6D4",  # cyan-500
            "#3B82F6",  # blue-500
            "#8B5CF6",  # violet-500
        ]

        # ── Black premium theme ──────────────────────────────────────────
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            for i, d in enumerate(sorted(durations)):
                color = palette[i % len(palette)]
                n_eff = max(d * ac_factor, 1)

                powers = []
                for m in mde_range:
                    delta = np.log(1 + m)
                    se = sigma / np.sqrt(n_eff)
                    z = delta / se if se > 0 else 0
                    pwr = norm.cdf(z - z_alpha)
                    powers.append(pwr)

                powers = np.array(powers)

                ax.plot(
                    mde_range * 100,
                    powers * 100,
                    color=color,
                    linewidth=2.2,
                    label=f"{d} days",
                )

                # ── Intersection marker at power_target ──────────────────
                cross_idx = np.searchsorted(powers, power_target)
                if 0 < cross_idx < len(powers):
                    p0, p1 = powers[cross_idx - 1], powers[cross_idx]
                    m0, m1 = mde_range[cross_idx - 1], mde_range[cross_idx]
                    if p1 != p0:
                        m_cross = m0 + (power_target - p0) / (p1 - p0) * (m1 - m0)
                    else:
                        m_cross = m0
                    ax.plot(
                        m_cross * 100,
                        power_target * 100,
                        "o",
                        color=color,
                        markersize=8,
                        markeredgecolor="white",
                        markeredgewidth=1.5,
                        zorder=5,
                    )
                    ax.annotate(
                        f"{m_cross * 100:.1f}%",
                        xy=(m_cross * 100, power_target * 100),
                        xytext=(0, 12),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color=color,
                    )

            # ── Target line ──────────────────────────────────────────────
            ax.axhline(
                power_target * 100,
                color="#94A3B8",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
            )
            ax.text(
                max_mde * 100 * 0.97,
                power_target * 100 + 1.5,
                f"Target: {power_target:.0%}",
                color="#94A3B8",
                fontsize=9,
                ha="right",
                va="bottom",
            )

            # ── Axes ─────────────────────────────────────────────────────
            ax.set_xlim(0, max_mde * 100)
            ax.set_ylim(0, 105)
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
            ax.set_xlabel("Real Impact (%)", fontsize=12, color="white")
            ax.set_ylabel("Power", fontsize=12, color="white")

            # ── Title + scenario subtitle ────────────────────────────────
            pct = s.get("pct_treatment", 0)
            n_treat = s.get("n_treatment", "?")
            treat_pool = s.get("treatment_pool", [])
            treat_str = ", ".join(treat_pool) if treat_pool else f"{n_treat} geos"

            fig.text(
                0.5, 0.99,
                "Power Analysis",
                ha="center", va="top",
                fontsize=16, fontweight="bold", color="white",
            )
            fig.text(
                0.5, 0.95,
                f"Minimum Detectable Effect (MDE) for {power_target:.0%} Power",
                ha="center", va="top",
                fontsize=12, fontweight="bold", color="white",
            )
            fig.text(
                0.5, 0.915,
                f"Scenario {scenario}  •  {pct:.0%} treatment  •  {treat_str}",
                ha="center", va="top",
                fontsize=10, color="#94A3B8",
            )

            ax.grid(True, linestyle="--", alpha=0.15, color="white")
            ax.tick_params(colors="#CBD5E1", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")

            ax.legend(
                loc="lower right",
                fontsize=10,
                framealpha=0.4,
                edgecolor="#333333",
                facecolor="#111111",
                labelcolor="white",
            )

            plt.tight_layout(rect=[0, 0, 1, 0.88])
            plt.show()

    def plot_scenario_comparison(
        self,
        target_days=28,
        power_target=0.80,
        alpha=0.05,
        figsize=(10, 6),
    ):
        """
        Bar chart comparing MDE vs Treatment Size across all scenarios.
        """
        from scipy.stats import norm
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        scenarios = self.get("scenarios", [])
        if not scenarios:
            print("  [plot_scenario_comparison] No scenarios found.")
            return

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power_target)
        z_mult = z_alpha + z_beta

        x_labels = []
        mdes = []

        for i, s in enumerate(scenarios):
            pct = s.get("pct_treatment", 0)
            x_labels.append(f"Scen {i}\n({pct:.0%})")
            
            dur = s.get("duration", {})
            cons_summary = dur.get("consolidated", {}).get("summary", {})
            sigma = cons_summary.get("sigma")
            ac_factor = cons_summary.get("ac_factor", 1.0)
            
            if sigma is None:
                mdes.append(np.nan)
                continue
                
            n_eff = max(target_days * ac_factor, 1)
            delta_log = z_mult * sigma / np.sqrt(n_eff)
            mde = np.exp(delta_log) - 1
            mdes.append(mde * 100)

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            bars = ax.bar(x_labels, mdes, color="#06B6D4", alpha=0.8, edgecolor="#22D3EE", linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.annotate(
                        f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        color="white", fontweight="bold", fontsize=10
                    )

            ax.set_ylabel(f"MDE @ {target_days} days (%)", fontsize=12, color="white")
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
            
            fig.text(0.5, 0.95, "Scenario Comparison", ha="center", va="top", fontsize=16, fontweight="bold", color="white")
            fig.text(0.5, 0.90, f"Cost vs. Sensitivity Trade-off ({power_target:.0%} Power)", ha="center", va="top", fontsize=12, color="#94A3B8")

            ax.grid(True, axis="y", linestyle="--", alpha=0.15, color="white")
            ax.tick_params(colors="#CBD5E1", labelsize=11)
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout(rect=[0, 0, 1, 0.86])
            plt.show()

    def plot_donor_weights(self, scenario=0, top_n=15, figsize=(10, 8)):
        """
        Horizontal bar chart of aggregated donor pool weights for a scenario.
        """
        import matplotlib.pyplot as plt
        from collections import defaultdict

        scenarios = self.get("scenarios", [])
        if not scenarios or scenario >= len(scenarios):
            print("  [plot_donor_weights] Invalid scenario.")
            return

        clusters = scenarios[scenario].get("clusters", [])
        
        weight_map = defaultdict(float)
        for cl in clusters:
            controls = cl.get("control", [])
            weights = cl.get("control_weights", [])
            if len(controls) == len(weights):
                for c, w in zip(controls, weights):
                    weight_map[c] += w

        if not weight_map:
            print("  [plot_donor_weights] No weights found (maybe matched_did?).")
            return

        sorted_donors = sorted(weight_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
        labels = [x[0] for x in sorted_donors][::-1]
        values = [x[1] for x in sorted_donors][::-1]

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            bars = ax.barh(labels, values, color="#10B981", alpha=0.8, edgecolor="#34D399")
            
            for bar in bars:
                width = bar.get_width()
                ax.annotate(
                    f"{width:.2f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left", va="center",
                    color="white", fontsize=9
                )

            ax.set_xlabel("Aggregated Weight in Synthetic Control", fontsize=12, color="white")
            
            pct = scenarios[scenario].get("pct_treatment", 0)
            fig.text(0.5, 0.95, "Donor Pool Composition", ha="center", va="top", fontsize=16, fontweight="bold", color="white")
            fig.text(0.5, 0.91, f"Scenario {scenario}  •  Top {top_n} Donors", ha="center", va="top", fontsize=12, color="#94A3B8")

            ax.grid(True, axis="x", linestyle="--", alpha=0.15, color="white")
            ax.tick_params(colors="#CBD5E1", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout(rect=[0, 0, 1, 0.88])
            plt.show()

    def plot_validation_quality(self, scenario=0, figsize=(8, 8)):
        """
        Scatter plot of R² Train vs R² Test to identify overfitting risk.
        """
        import matplotlib.pyplot as plt

        scenarios = self.get("scenarios", [])
        if not scenarios or scenario >= len(scenarios):
            print("  [plot_validation_quality] Invalid scenario.")
            return

        val_df = scenarios[scenario].get("validation")
        if val_df is None or "r2_train" not in val_df.columns:
            print("  [plot_validation_quality] Validation metrics not available.")
            return

        r2_tr = val_df["r2_train"].values
        r2_te = val_df["r2_test"].values
        
        # Color based on degradation (train - test)
        degradations = np.clip(r2_tr - r2_te, 0, 1)
        
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            # Diagonal reference line
            ax.plot([0, 1], [0, 1], color="#64748B", linestyle="--", alpha=0.6, label="Perfect Stability")
            
            # Scatter points
            scatter = ax.scatter(
                r2_tr, r2_te, 
                c=degradations, cmap="YlOrRd", 
                s=100, alpha=0.9, edgecolor="white", linewidth=1,
                zorder=5
            )

            # Threshold lines
            ax.axhline(0.6, color="#EF4444", linestyle=":", alpha=0.5, label="Min Acceptable R²")
            ax.axvline(0.6, color="#EF4444", linestyle=":", alpha=0.5)

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("R² (Training - In Time)", fontsize=12, color="white")
            ax.set_ylabel("R² (Validation - Out of Time)", fontsize=12, color="white")
            
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("R² Degradation (Train - Test)", color="white")
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            fig.text(0.5, 0.95, "Design Stability (Cross-Validation)", ha="center", va="top", fontsize=16, fontweight="bold", color="white")
            fig.text(0.5, 0.91, f"Scenario {scenario}  •  Each point is a cluster", ha="center", va="top", fontsize=12, color="#94A3B8")

            ax.grid(True, linestyle="--", alpha=0.15, color="white")
            ax.tick_params(colors="#CBD5E1", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")
                
            ax.legend(loc="upper left", framealpha=0.4, facecolor="#111111", edgecolor="#333333", labelcolor="white")

            plt.tight_layout(rect=[0, 0, 1, 0.88])
            plt.show()

    def plot_duration_mde_tradeoff(self, durations=None, power_target=0.80, alpha=0.05, figsize=(12, 7)):
        """
        Plot MDE vs Days for multiple scenarios to visualize the time trade-off.
        """
        from scipy.stats import norm
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        scenarios = self.get("scenarios", [])
        if not scenarios:
            print("  [plot_duration_mde_tradeoff] No scenarios found.")
            return

        if durations is None:
            durations = list(range(14, 61, 7))  # 14 to 60 days, weekly steps

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power_target)
        z_mult = z_alpha + z_beta

        palette = ["#3B82F6", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#06B6D4"]

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            for i, s in enumerate(scenarios):
                pct = s.get("pct_treatment", 0)
                dur = s.get("duration", {})
                cons_summary = dur.get("consolidated", {}).get("summary", {})
                sigma = cons_summary.get("sigma")
                ac_factor = cons_summary.get("ac_factor", 1.0)

                if sigma is None:
                    continue

                color = palette[i % len(palette)]
                mdes = []
                
                for d in durations:
                    n_eff = max(d * ac_factor, 1)
                    delta_log = z_mult * sigma / np.sqrt(n_eff)
                    mdes.append((np.exp(delta_log) - 1) * 100)

                ax.plot(
                    durations, mdes, 
                    color=color, linewidth=2.5, marker="o", markersize=6,
                    label=f"Scenario {i} ({pct:.0%} tx)"
                )

            ax.set_xlabel("Experiment Duration (Days)", fontsize=12, color="white")
            ax.set_ylabel("Minimum Detectable Effect (%)", fontsize=12, color="white")
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))

            fig.text(0.5, 0.95, "Duration vs. Sensitivity Trade-off", ha="center", va="top", fontsize=16, fontweight="bold", color="white")
            fig.text(0.5, 0.91, f"MDE required to achieve {power_target:.0%} Power across Scenarios", ha="center", va="top", fontsize=12, color="#94A3B8")

            ax.grid(True, linestyle="--", alpha=0.15, color="white")
            ax.tick_params(colors="#CBD5E1", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")

            ax.legend(loc="upper right", framealpha=0.4, facecolor="#111111", edgecolor="#333333", labelcolor="white")

            plt.tight_layout(rect=[0, 0, 1, 0.88])
            plt.show()

    def plot_cluster_fits(self, scenario=0, figsize=(14, 4)):
        """
        Plot the Treatment vs Synthetic time series for each cluster in the scenario.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not hasattr(self, "_df") or self._df is None:
            print("  [plot_cluster_fits] No underlying dataframe attached to this DoEResult.")
            return

        scenarios = self.get("scenarios", [])
        if not scenarios or scenario >= len(scenarios):
            print("  [plot_cluster_fits] Invalid scenario.")
            return

        clusters = scenarios[scenario].get("clusters", [])
        if not clusters:
            print("  [plot_cluster_fits] No clusters found in this scenario.")
            return

        df = self._df.copy()
        date_col = getattr(self, "_date_col", None)
        if date_col is None or date_col not in df.columns:
            # Try to guess date column
            date_col = df.select_dtypes(include=['datetime64']).columns[0] if len(df.select_dtypes(include=['datetime64']).columns) > 0 else df.columns[0]

        df = df.sort_values(date_col).set_index(date_col)

        n_clusters = len(clusters)
        cols = 2
        rows = (n_clusters + 1) // cols
        if rows == 0: rows = 1

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows), squeeze=False)
            fig.patch.set_facecolor("black")

            axes_flat = axes.flatten()

            for i, cl in enumerate(clusters):
                ax = axes_flat[i]
                ax.set_facecolor("black")

                treat_geos = cl.get("treatment", [])
                ctrl_geos = cl.get("control", [])
                weights = cl.get("control_weights", [])

                if not treat_geos or not ctrl_geos or not weights:
                    continue

                # Compute series
                y_treat = df[treat_geos].mean(axis=1) if len(treat_geos) > 1 else df[treat_geos[0]]
                X_ctrl = df[ctrl_geos].values
                w_arr = np.array(weights)

                # Weights in DoE are derived on normalized volume scale.
                # We must normalize X, apply weights, then re-scale to y_treat's mean.
                y_mean = y_treat.mean()
                if y_mean == 0: y_mean = 1e-10
                
                X_mean = X_ctrl.mean(axis=0)
                X_mean = np.where(X_mean == 0, 1e-10, X_mean)

                X_norm = X_ctrl / X_mean
                y_synth = (X_norm @ w_arr) * y_mean

                # Plot
                ax.plot(df.index, y_treat, color="#06B6D4", linewidth=2, label="Treatment")
                ax.plot(df.index, y_synth, color="#10B981", linewidth=2, linestyle="--", label="Synthetic")

                treat_str = ", ".join(treat_geos)
                ax.set_title(f"Cluster {i} ({treat_str})", color="white", fontsize=11, fontweight="bold")
                
                ax.grid(True, linestyle="--", alpha=0.15, color="white")
                ax.tick_params(colors="#CBD5E1", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color("#1E1E1E")
                
                # Format dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

                if i == 0:
                    ax.legend(loc="best", framealpha=0.4, facecolor="#111111", edgecolor="#333333", labelcolor="white")

            # Hide unused subplots
            for j in range(i + 1, len(axes_flat)):
                fig.delaxes(axes_flat[j])

            fig.text(0.5, 0.98, "Pre-Intervention Fit (Treatment vs Synthetic)", ha="center", va="top", fontsize=16, fontweight="bold", color="white")
            fig.text(0.5, 0.95, f"Scenario {scenario}", ha="center", va="top", fontsize=12, color="#94A3B8")

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()

    def plot_consolidated_fit(self, scenario=0, figsize=(12, 5)):
        """
        Plot the aggregated Treatment vs Synthetic time series (summing all clusters).
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not hasattr(self, "_df") or self._df is None:
            print("  [plot_consolidated_fit] No underlying dataframe attached to this DoEResult.")
            return

        scenarios = self.get("scenarios", [])
        if not scenarios or scenario >= len(scenarios):
            print("  [plot_consolidated_fit] Invalid scenario.")
            return

        clusters = scenarios[scenario].get("clusters", [])
        if not clusters:
            print("  [plot_consolidated_fit] No clusters found.")
            return

        df = self._df.copy()
        date_col = getattr(self, "_date_col", None)
        if date_col is None or date_col not in df.columns:
            date_col = df.select_dtypes(include=['datetime64']).columns[0] if len(df.select_dtypes(include=['datetime64']).columns) > 0 else df.columns[0]

        df = df.sort_values(date_col).set_index(date_col)

        agg_treat = np.zeros(len(df))
        agg_synth = np.zeros(len(df))

        for cl in clusters:
            treat_geos = cl.get("treatment", [])
            ctrl_geos = cl.get("control", [])
            weights = cl.get("control_weights", [])

            if treat_geos and ctrl_geos and weights:
                # Same normalization logic
                y_treat = df[treat_geos].mean(axis=1) if len(treat_geos) > 1 else df[treat_geos[0]]
                X_ctrl = df[ctrl_geos].values
                w_arr = np.array(weights)

                y_mean = y_treat.mean()
                if y_mean == 0: y_mean = 1e-10
                
                X_mean = X_ctrl.mean(axis=0)
                X_mean = np.where(X_mean == 0, 1e-10, X_mean)

                X_norm = X_ctrl / X_mean
                y_synth_vals = (X_norm @ w_arr) * y_mean
                
                agg_treat += y_treat.values
                agg_synth += y_synth_vals

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            ax.plot(df.index, agg_treat, color="#06B6D4", linewidth=2.5, label="Total Treatment")
            ax.plot(df.index, agg_synth, color="#10B981", linewidth=2.5, linestyle="--", label="Total Synthetic")

            ax.set_ylabel("Metric Volume", fontsize=12, color="white")

            fig.text(0.5, 0.98, "Consolidated Fit (Total Treatment vs Synthetic)", ha="center", va="top", fontsize=16, fontweight="bold", color="white")
            pct = scenarios[scenario].get("pct_treatment", 0)
            fig.text(0.5, 0.92, f"Scenario {scenario}  •  Aggregated across {len(clusters)} clusters ({pct:.0%} tx)", ha="center", va="top", fontsize=12, color="#94A3B8")

            ax.grid(True, linestyle="--", alpha=0.15, color="white")
            ax.tick_params(colors="#CBD5E1", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

            ax.legend(loc="best", fontsize=11, framealpha=0.4, facecolor="#111111", edgecolor="#333333", labelcolor="white")

            plt.tight_layout(rect=[0, 0, 1, 0.90])
            plt.show()

class ExperimentResult(dict):
    """
    Wrapper for experiment results. Inherits from dict for backward compatibility,
    but provides visualization methods.
    """
    def plot_cluster_effects(self, post_only=False, figsize=(14, 4)):
        """
        Plot the Treatment vs Synthetic time series for each cluster in the experiment.
        If post_only=True, zooms in to show only the test period.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.dates as mdates

        def human_format(x, pos):
            if abs(x) >= 1e9: return f'{x/1e9:.1f}B'
            if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
            if abs(x) >= 1e3: return f'{x/1e3:.1f}k'
            return f'{x:,.0f}'

        formatter = ticker.FuncFormatter(human_format)

        results = self.get("results", [])
        if not results:
            print("  [plot_cluster_effects] No experiment results found.")
            return

        date_col = getattr(self, "_date_col", None)

        n_clusters = len(results)
        cols = 2
        rows = (n_clusters + 1) // cols
        if rows == 0: rows = 1

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows), squeeze=False)
            fig.patch.set_facecolor("black")
            axes_flat = axes.flatten()

            for i, res in enumerate(results):
                ax = axes_flat[i]
                ax.set_facecolor("black")

                syn = res.get("synthetic", {})
                plot_data = syn.get("plotting_data", {})
                
                if not plot_data or "y" not in plot_data:
                    continue

                y_treat = plot_data["y"]
                y_synth = plot_data["synthetic"]
                t_idx = plot_data["treatment_idx"]
                treat_geo = plot_data["treatment_geo"]
                geo_name = ", ".join(treat_geo) if isinstance(treat_geo, list) else treat_geo

                df = syn.get("df")
                if df is not None:
                    d_col = date_col if date_col and date_col in df.columns else df.columns[0]
                    dates = df[d_col]
                else:
                    dates = list(range(len(y_treat)))

                if post_only:
                    y_treat = y_treat[t_idx:]
                    y_synth = y_synth[t_idx:]
                    dates = dates.iloc[t_idx:] if hasattr(dates, 'iloc') else dates[t_idx:]

                ax.plot(dates, y_treat, color="#06B6D4", linewidth=2, label="Treatment")
                ax.plot(dates, y_synth, color="#10B981", linewidth=2, linestyle="--", label="Synthetic")
                
                # Intervention line only makes sense if we show pre-period
                if not post_only:
                    v_line_pos = dates.iloc[t_idx] if hasattr(dates, 'iloc') else t_idx
                    ax.axvline(v_line_pos, linestyle=":", color="#EF4444", linewidth=2, alpha=0.8, label="Intervention")

                ax.set_title(f"Cluster {i} ({geo_name})", color="white", fontsize=11, fontweight="bold")
                
                ax.grid(True, linestyle="--", alpha=0.15, color="white")
                ax.tick_params(colors="#CBD5E1", labelsize=9)
                ax.yaxis.set_major_formatter(formatter)
                
                for spine in ax.spines.values():
                    spine.set_color("#1E1E1E")
                
                if hasattr(dates, 'iloc'):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

                if i == 0:
                    ax.legend(loc="best", framealpha=0.4, facecolor="#111111", edgecolor="#333333", labelcolor="white")

            for j in range(i + 1, len(axes_flat)):
                fig.delaxes(axes_flat[j])

            fig.text(0.5, 0.98, "Treatment Effect (Treatment vs Synthetic)", ha="center", va="top", fontsize=16, fontweight="bold", color="white")
            subtitle = "Test Period Only" if post_only else "Red dotted line indicates start of treatment"
            fig.text(0.5, 0.95, subtitle, ha="center", va="top", fontsize=12, color="#94A3B8")

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()

    def plot_consolidated_effect(self, post_only=False, figsize=(12, 5)):
        """
        Plot the aggregated Treatment vs Synthetic time series for the entire experiment.
        If post_only=True, zooms in to show only the test period.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.dates as mdates

        def human_format(x, pos):
            if abs(x) >= 1e9: return f'{x/1e9:.1f}B'
            if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
            if abs(x) >= 1e3: return f'{x/1e3:.1f}k'
            return f'{x:,.0f}'

        formatter = ticker.FuncFormatter(human_format)

        results = self.get("results", [])
        if not results:
            print("  [plot_consolidated_effect] No experiment results found.")
            return

        date_col = getattr(self, "_date_col", None)
        
        # We need to find the first valid df to get dates and length
        ref_df = None
        d_col = date_col
        for res in results:
            df = res.get("synthetic", {}).get("df")
            if df is not None:
                ref_df = df
                if not d_col or d_col not in ref_df.columns:
                    d_col = ref_df.columns[0]
                break
                
        if ref_df is None:
            print("  [plot_consolidated_effect] No dataframes attached to results.")
            return

        dates = ref_df[d_col]
        agg_treat = np.zeros(len(dates))
        agg_synth = np.zeros(len(dates))
        t_idx = None

        for res in results:
            plot_data = res.get("synthetic", {}).get("plotting_data", {})
            if plot_data and "y" in plot_data:
                agg_treat += np.array(plot_data["y"])
                agg_synth += np.array(plot_data["synthetic"])
                t_idx = plot_data["treatment_idx"]

        if post_only and t_idx is not None:
            dates = dates.iloc[t_idx:] if hasattr(dates, 'iloc') else dates[t_idx:]
            agg_treat = agg_treat[t_idx:]
            agg_synth = agg_synth[t_idx:]

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            ax.plot(dates, agg_treat, color="#06B6D4", linewidth=2.5, label="Total Treatment")
            ax.plot(dates, agg_synth, color="#10B981", linewidth=2.5, linestyle="--", label="Total Synthetic")
            
            if not post_only and t_idx is not None:
                v_line_pos = dates.iloc[t_idx] if hasattr(dates, 'iloc') else t_idx
                ax.axvline(v_line_pos, linestyle=":", color="#EF4444", linewidth=2, alpha=0.8, label="Intervention")

            ax.set_ylabel("Metric Volume", fontsize=12, color="white")
            ax.yaxis.set_major_formatter(formatter)

            fig.text(0.5, 0.98, "Consolidated Treatment Effect", ha="center", va="top", fontsize=16, fontweight="bold", color="white")
            fig.text(0.5, 0.92, f"Aggregated across {len(results)} clusters", ha="center", va="top", fontsize=12, color="#94A3B8")

            ax.grid(True, linestyle="--", alpha=0.15, color="white")
            ax.tick_params(colors="#CBD5E1", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

            ax.legend(loc="best", fontsize=11, framealpha=0.4, facecolor="#111111", edgecolor="#333333", labelcolor="white")

            plt.tight_layout(rect=[0, 0, 1, 0.90])
            plt.show()

    def plot_lift_distributions(self, show_null=False, figsize=(15, 6)):
        """
        Plot the bootstrap distributions of the Absolute and Percentual Lifts.
        Shows histograms with KDE curves and 95% Confidence Intervals.
        If show_null=True, overlays the theoretical Null Hypothesis (H0) distribution.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import scipy.stats as stats
        from reallift.geo.bootstrap import bootstrap_significance

        def human_format(x, pos):
            if abs(x) >= 1e9: return f'{x/1e9:.1f}B'
            if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
            if abs(x) >= 1e3: return f'{x/1e3:.1f}k'
            return f'{x:,.0f}'

        results = self.get("results", [])
        if not results:
            print("  [plot_lift_distributions] No experiment results found.")
            return

        # Calculate consolidated arrays
        try:
            post_len = len(results[0]["synthetic"]["plotting_data"]["post_real"])
            consolidated_post_real_arr = np.zeros(post_len)
            consolidated_post_synth_arr = np.zeros(post_len)
            
            for res in results:
                consolidated_post_real_arr += res["synthetic"]["plotting_data"]["post_real"]
                consolidated_post_synth_arr += res["synthetic"]["plotting_data"]["post_synth"]
                
            consolidated_effect_arr = consolidated_post_real_arr - consolidated_post_synth_arr
            
            # Re-run bootstrap for consolidated to get the raw distribution arrays
            cons_boot = bootstrap_significance(consolidated_effect_arr, consolidated_post_synth_arr, conf_level=0.95)
            
            boot_abs = cons_boot["boot_totals_abs"]
            boot_pct = cons_boot["boot_totals_pct"] * 100  # Convert to percentage
            
            ci_l_abs = cons_boot["ci_lower_total_abs"]
            ci_u_abs = cons_boot["ci_upper_total_abs"]
            ci_l_pct = cons_boot["ci_lower_total_pct"] * 100
            ci_u_pct = cons_boot["ci_upper_total_pct"] * 100
            
            # True observed values
            tot_obs = np.sum(consolidated_post_real_arr)
            tot_exp = np.sum(consolidated_post_synth_arr)
            obs_lift_abs = tot_obs - tot_exp
            obs_lift_pct = (obs_lift_abs / tot_exp * 100) if tot_exp != 0 else 0
            
            std_abs = np.std(boot_abs)
            std_pct = np.std(boot_pct)
            
        except Exception as e:
            print(f"  [plot_lift_distributions] Could not generate bootstrap distributions: {e}")
            return

        with plt.style.context("dark_background"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            fig.patch.set_facecolor("black")
            
            def plot_dist(ax, data, ci_l, ci_u, mean_val, title, is_pct=False):
                ax.set_facecolor("black")
                ax.grid(True, linestyle='--', alpha=0.15, color="white")
                for spine in ax.spines.values():
                    spine.set_color("#1E1E1E")
                
                # Histogram
                count, bins, _ = ax.hist(data, bins=40, density=True, color="#06B6D4", alpha=0.3, edgecolor=None)
                
                # KDE Curves
                try:
                    std_data = np.std(data)
                    
                    if show_null:
                        # Construct Null Distribution by centering the empirical variance at 0
                        null_data = data - np.mean(data)
                        null_kde = stats.gaussian_kde(null_data)
                        x_null = np.linspace(min(null_data) - std_data, max(null_data) + std_data, 200)
                        
                        ax.plot(x_null, null_kde(x_null), color="#EF4444", linewidth=2, label="Null Hypothesis (H0)")
                        ax.fill_between(x_null, 0, null_kde(x_null), color="#EF4444", alpha=0.15)
                        ax.axvline(0, color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.8)
                    else:
                        ax.axvline(0, color="#EF4444", linestyle="-", linewidth=2, label="Null Effect (0)")

                    # Observed Effect Distribution (H1)
                    kde = stats.gaussian_kde(data)
                    x_eval = np.linspace(min(data) - std_data, max(data) + std_data, 200)
                    ax.plot(x_eval, kde(x_eval), color="#10B981", linewidth=2.5, label="Observed Lift (H1)")
                    
                    # Shade CI region for Observed Lift
                    x_fill = np.linspace(ci_l, ci_u, 100)
                    ax.fill_between(x_fill, 0, kde(x_fill), color="#10B981", alpha=0.25)
                except Exception:
                    if not show_null:
                        ax.axvline(0, color="#EF4444", linestyle="-", linewidth=2, label="Null Effect (0)")
                
                # Vertical Lines
                ax.axvline(mean_val, color="#FBBF24", linestyle="--", linewidth=2, label="Point Estimate")
                ax.axvline(ci_l, color="#94A3B8", linestyle=":", linewidth=2, label="95% CI")
                ax.axvline(ci_u, color="#94A3B8", linestyle=":", linewidth=2)
                
                ax.set_title(title, color="white", fontweight="bold", fontsize=12)
                ax.tick_params(colors="#CBD5E1", labelsize=10)
                
                if not is_pct:
                    ax.xaxis.set_major_formatter(ticker.FuncFormatter(human_format))
                else:
                    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=1))
                    
                ax.legend(loc="upper right", framealpha=0.4, facecolor="#111111", edgecolor="#333333", labelcolor="white", fontsize=9)

            plot_dist(ax1, boot_abs, ci_l_abs, ci_u_abs, obs_lift_abs, "Absolute Lift Distribution")
            plot_dist(ax2, boot_pct, ci_l_pct, ci_u_pct, obs_lift_pct, "Percentual Lift Distribution", is_pct=True)

            # Add text boxes with metrics
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#111111", edgecolor="#333333", alpha=0.8)
            
            # Formatting helpers for text
            def fmt(val): return f"{val:,.2f}"
            def fmt_pct(val): return f"{val:.2f}%"

            text_abs = (
                f"Observed Output: {fmt(tot_obs)}\n"
                f"Expected Output: {fmt(tot_exp)}\n"
                f"--------------------------------------\n"
                f"Absolute Lift: {fmt(obs_lift_abs)}\n"
                f"95% CI: [{fmt(ci_l_abs)}, {fmt(ci_u_abs)}]\n"
                f"Std Deviation: {fmt(std_abs)}"
            )
            ax1.text(0.03, 0.95, text_abs, transform=ax1.transAxes, fontsize=10,
                     verticalalignment='top', color="white", bbox=bbox_props, family='monospace')

            text_pct = (
                f"Percentual Lift: {fmt_pct(obs_lift_pct)}\n"
                f"95% CI: [{fmt_pct(ci_l_pct)}, {fmt_pct(ci_u_pct)}]\n"
                f"Std Deviation: {fmt_pct(std_pct)}"
            )
            ax2.text(0.03, 0.95, text_pct, transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top', color="white", bbox=bbox_props, family='monospace')

            fig.suptitle("Consolidated Treatment Effect (Bootstrap Distribution)", color="white", fontsize=16, fontweight="bold")
            fig.text(0.5, 0.92, "Empirical distributions generated via Moving Block Bootstrap (MBB)", ha="center", va="top", fontsize=12, color="#94A3B8")

            plt.tight_layout(rect=[0, 0, 1, 0.90])
            plt.show()

class RealLift:
    """
    Central orchestrator for the RealLift causal inference pipeline.

    Loads, parses, and validates data once in the constructor. All downstream
    operations (cleaning, design, simulation, experiment analysis) operate
    on the pre-validated, in-memory DataFrame — eliminating redundant I/O
    and ensuring consistent date parsing.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to a CSV file or a pandas DataFrame.
    date_col : str
        Name of the date column.
    start_date : str, optional
        Start of the analysis window (YYYY-MM-DD). Stored as default for methods.
    end_date : str, optional
        End of the analysis window (YYYY-MM-DD). Stored as default for methods.
    dayfirst : bool, default True
        Whether to interpret the first value in ambiguous dates as the day.
    verbose : bool, default True
        Whether to print summary information on initialization.

    Examples
    --------
    >>> from reallift import RealLift
    >>> rl = RealLift("data.csv", date_col="dia", start_date="2025-11-01")
    >>> rl.clean()
    >>> doe = rl.design()
    >>> sim = rl.simulate(treatment_geos=["sao_paulo"], lift=0.05, days=20)
    >>> results = rl.run(treatment_start_date="2026-03-16", doe=doe, scenario=0)
    """

    def __init__(
        self,
        data,
        date_col: str,
        start_date: str = None,
        end_date: str = None,
        dayfirst: bool = True,
        verbose: bool = True,
    ):
        self.date_col = date_col
        self.start_date = start_date
        self.end_date = end_date
        self.dayfirst = dayfirst
        self._verbose = verbose
        self._filepath = None

        # ── Load Data ──
        if isinstance(data, str):
            self._filepath = data
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(
                f"'data' must be a file path (str) or pd.DataFrame, got {type(data).__name__}"
            )

        # ── Validate date column ──
        if date_col not in df.columns:
            raise ValueError(
                f"Date column '{date_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # ── Parse dates (single, centralized parsing) ──
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(
                df[date_col].astype(str),
                format="mixed",
                dayfirst=dayfirst,
                errors="coerce",
            )

        n_nat = df[date_col].isna().sum()
        if n_nat > 0:
            if verbose:
                print(f"  [Warning] Dropped {n_nat} rows with unparseable dates.")
            df = df.dropna(subset=[date_col])

        df = df.sort_values(date_col).reset_index(drop=True)

        # ── Detect geos ──
        self.geos = [col for col in df.columns if col != date_col]
        if len(self.geos) < 2:
            raise ValueError(
                f"Need at least 2 geo columns, found {len(self.geos)}: {self.geos}"
            )

        # ── Store full DataFrame (no period filtering here) ──
        self._df_raw = df.copy()  # immutable baseline — never mutated
        self.df = df

        if verbose:
            date_min = df[date_col].min().strftime("%Y-%m-%d")
            date_max = df[date_col].max().strftime("%Y-%m-%d")
            print(f"\n  RealLift initialized")
            print(f"  {'-' * 50}")
            print(f"  Date column : {date_col}")
            print(f"  Date range  : {date_min} -> {date_max}")
            print(f"  Rows        : {len(df):,}")
            print(f"  Geos        : {len(self.geos)}")
            if start_date or end_date:
                sd = start_date or date_min
                ed = end_date or date_max
                print(f"  Window      : {sd} -> {ed}")
            print(f"  {'-' * 50}\n")

    # ──────────────────────────────────────────────────────────────────────
    # Public Methods
    # ──────────────────────────────────────────────────────────────────────

    def clean(
        self,
        imputation_method: str = "constant",
        constant_value: float = 1e-3,
        verbose: bool = None,
        plot: bool = False,
        save_csv: bool = True,
        save_pdf: bool = False,
        file_name: str = "cleaned_geo_data.csv",
        pdf_name: str = "cleaning_report.pdf",
        max_zero_rate: float = None,
        top_n_geos: int = None,
        keep_top_quantiles: int = None,
        exclude_geos: list = None,
        quantile_bins: int = None,
        logo: str = None,
    ) -> pd.DataFrame:
        """
        Clean and validate the geo data. Wraps ``clean_geo_data``.

        Updates ``self.df`` with the cleaned result and returns it.
        """
        from .utils.data_cleaning import clean_geo_data

        if verbose is None:
            verbose = self._verbose

        # Always rebuild from the pristine baseline so that each
        # call to clean() is idempotent (no accumulated mutations).
        fresh_df = self._df_raw.copy()

        result = clean_geo_data(
            data=fresh_df,
            date_col=self.date_col,
            imputation_method=imputation_method,
            constant_value=constant_value,
            verbose=verbose,
            plot=plot,
            save_csv=save_csv,
            save_pdf=save_pdf,
            file_name=file_name,
            pdf_name=pdf_name,
            max_zero_rate=max_zero_rate,
            top_n_geos=top_n_geos,
            keep_top_quantiles=keep_top_quantiles,
            exclude_geos=exclude_geos,
            quantile_bins=quantile_bins,
            start_date=self.start_date,
            end_date=self.end_date,
            logo=logo,
        )

        # Update internal state with cleaned result
        self.df = result
        self.geos = [c for c in result.columns if c != self.date_col]

        # Track the cleaned CSV path for downstream functions
        if save_csv:
            self._filepath = file_name

        return result

    def design(
        self,
        geos=None,
        pct_treatment=None,
        fixed_treatment=None,
        mde=None,
        experiment_days=None,
        n_folds=5,
        search_mode="ranking",
        experiment_type="synthetic_control",
        use_elasticnet=False,
        check_ghost_lift=True,
        n_jobs=None,
        verbose=None,
        save_pdf=False,
        pdf_name="doe_report.pdf",
        logo=None,
    ) -> "DoEResult":
        """
        Run Design of Experiments. Wraps ``design_of_experiments``.

        Returns a :class:`DoEResult` (dict subclass) that supports
        ``doe.plot_mde()`` for visual MDE analysis.
        """
        from .pipelines.design import design_of_experiments
        from .config.defaults import DEFAULT_EXPERIMENT_DAYS

        if verbose is None:
            verbose = self._verbose
        if experiment_days is None:
            experiment_days = DEFAULT_EXPERIMENT_DAYS

        raw = design_of_experiments(
            filepath=self._filepath,
            date_col=self.date_col,
            start_date=self.start_date,
            end_date=self.end_date,
            geos=geos,
            pct_treatment=pct_treatment,
            fixed_treatment=fixed_treatment,
            mde=mde,
            experiment_days=experiment_days,
            n_folds=n_folds,
            search_mode=search_mode,
            experiment_type=experiment_type,
            use_elasticnet=use_elasticnet,
            check_ghost_lift=check_ghost_lift,
            n_jobs=n_jobs,
            verbose=verbose,
            save_pdf=save_pdf,
            pdf_name=pdf_name,
            logo=logo,
            df=self.df,
        )
        
        result = DoEResult(raw)
        result._df = self.df
        result._date_col = self.date_col
        return result

    def simulate(
        self,
        treatment_geos,
        days=None,
        start_date=None,
        end_date=None,
        lift=0.05,
        trend_slope=None,
        seasonality_amplitude=None,
        seasonality_period=None,
        noise_std=None,
        random_seed=42,
        plot=True,
        log_scale=False,
        save_csv=False,
        file_name="simulated_intervention.csv",
        as_integer=False,
        verbose=None,
    ) -> pd.DataFrame:
        """
        Generate simulated post-intervention data. Wraps ``generate_simulated_intervention``.

        Updates ``self.df`` with the combined (pre + simulated post) DataFrame.
        """
        from .simulation.geo_simulation import generate_simulated_intervention

        if verbose is None:
            verbose = self._verbose

        result = generate_simulated_intervention(
            filepath=self._filepath,
            treatment_geos=treatment_geos,
            days=days,
            start_date=start_date,
            end_date=end_date,
            lift=lift,
            date_col=self.date_col,
            trend_slope=trend_slope,
            seasonality_amplitude=seasonality_amplitude,
            seasonality_period=seasonality_period,
            noise_std=noise_std,
            random_seed=random_seed,
            plot=plot,
            log_scale=log_scale,
            save_csv=save_csv,
            file_name=file_name,
            as_integer=as_integer,
            verbose=verbose,
            df=self.df,
        )

        # Update internal state with combined pre+post data
        self.df = result
        self.geos = [c for c in result.columns if c != self.date_col]

        if save_csv:
            self._filepath = file_name

        return result

    def run(
        self,
        treatment_start_date=None,
        treatment_end_date=None,
        doe=None,
        scenario=None,
        start_date=None,
        end_date=None,
        geos=None,
        n_treatment=1,
        fixed_treatment=None,
        mde=0.015,
        experiment_days=None,
        n_folds=5,
        random_state=None,
        conf_level=0.95,
        plot=False,
        verbose=None,
        ignore_treatment_start=False,
        ignore_treatment_end=False,
        perform_backtesting=None,
    ) -> dict:
        """
        Run the full GeoLift experiment analysis. Wraps ``run_geo_experiment``.

        Parameters
        ----------
        perform_backtesting : dict, optional
            When provided, runs a **pre-experiment backtest** instead of a real
            analysis.  The method simulates an intervention on the last *N* days
            of the pre-test history (Real History Backtest mode) and then feeds
            the resulting data into the standard experiment pipeline.

            Accepted keys:

            * ``lift`` (float, default ``0.0``): fractional lift to inject into
              the treatment geos (e.g. ``0.05`` = +5 %).  Use ``0.0`` to run a
              placebo / null-hypothesis check.
            * ``days`` (int, default ``28``): number of historical days to
              carve out as the simulated post-intervention window.

            The treatment geos are resolved automatically from *doe/scenario*
            or *fixed_treatment*.

            Example::

                results = rl.run(
                    doe=doe,
                    scenario=0,
                    perform_backtesting={"lift": 0.0, "days": 28},
                )
        """
        from .pipelines.experiment import run_geo_experiment
        from .config.defaults import DEFAULT_EXPERIMENT_DAYS

        if verbose is None:
            verbose = self._verbose
        if experiment_days is None:
            experiment_days = DEFAULT_EXPERIMENT_DAYS

        # ── Determine the DataFrame to analyse ──
        run_df = self.df

        if perform_backtesting is not None:
            from .simulation.geo_simulation import generate_simulated_intervention

            bt_lift = perform_backtesting.get("lift", 0.0)
            bt_days = perform_backtesting.get("days", 28)

            # --- Resolve treatment geos ---
            if doe is not None and scenario is not None:
                clusters = doe["scenarios"][scenario]["clusters"]
                treatment_geos = []
                for c in clusters:
                    t = c["treatment"]
                    if isinstance(t, list):
                        treatment_geos.extend(t)
                    else:
                        treatment_geos.append(t)
            elif fixed_treatment is not None:
                treatment_geos = list(fixed_treatment)
            else:
                raise ValueError(
                    "perform_backtesting requires treatment geos — provide "
                    "'doe' + 'scenario' or 'fixed_treatment'."
                )

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"  BACKTESTING MODE")
                print(f"  Injected lift : {bt_lift:+.2%}")
                print(f"  Window        : last {bt_days} days of pre-test history")
                print(f"  Treatment geos: {treatment_geos}")
                print(f"{'=' * 60}\n")

            # Simulate using negative-days (real history backtest)
            run_df = generate_simulated_intervention(
                treatment_geos=treatment_geos,
                days=-bt_days,
                lift=bt_lift,
                date_col=self.date_col,
                plot=plot,
                verbose=verbose,
                df=self.df,
            )

            # Derive treatment_start_date from the backtest split
            treatment_start_date = (
                run_df[self.date_col].iloc[-bt_days].strftime("%Y-%m-%d")
            )
            # End date = last row of simulated data
            treatment_end_date = (
                run_df[self.date_col].iloc[-1].strftime("%Y-%m-%d")
            )

        # treatment_start_date is required for the experiment pipeline
        if treatment_start_date is None:
            raise ValueError(
                "treatment_start_date is required (or use perform_backtesting "
                "to derive it automatically)."
            )

        raw_results = run_geo_experiment(
            filepath=self._filepath,
            date_col=self.date_col,
            treatment_start_date=treatment_start_date,
            treatment_end_date=treatment_end_date,
            doe=doe,
            scenario=scenario,
            start_date=start_date or self.start_date,
            end_date=end_date or self.end_date,
            geos=geos,
            n_treatment=n_treatment,
            fixed_treatment=fixed_treatment,
            mde=mde,
            experiment_days=experiment_days,
            n_folds=n_folds,
            random_state=random_state,
            conf_level=conf_level,
            plot=plot,
            verbose=verbose,
            ignore_treatment_start=ignore_treatment_start,
            ignore_treatment_end=ignore_treatment_end,
            df=run_df,
        )
        
        result = ExperimentResult(raw_results)
        result._df = self.df
        result._date_col = self.date_col
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self):
        n_rows = len(self.df) if self.df is not None else 0
        n_geos = len(self.geos) if self.geos else 0
        return f"RealLift(date_col='{self.date_col}', rows={n_rows}, geos={n_geos})"
