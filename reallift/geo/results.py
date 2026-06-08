import numpy as np
import pandas as pd


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
        max_mde=None,
        power_target=0.80,
        alpha=0.05,
        figsize=(12, 7),
    ):
        """
        Power Analysis â€” Power vs Real Impact (%) for different durations.

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

        # â”€â”€ Resolve durations to plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # Auto max_mde: compute the MDE at power_target for the shortest duration
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power_target)
        if max_mde is None:
            d_min_for_mde = min(durations)
            n_eff_min = max(d_min_for_mde * ac_factor, 1)
            delta_log = (z_alpha + z_beta) * sigma / np.sqrt(n_eff_min)
            mde_auto = np.exp(delta_log) - 1
            max_mde = max(0.05, round(mde_auto * 1.5, 2))

        mde_range = np.linspace(0, max_mde, 200)

        palette = [
            "#EF4444",  # red-500
            "#F59E0B",  # amber-500
            "#10B981",  # emerald-500
            "#06B6D4",  # cyan-500
            "#3B82F6",  # blue-500
            "#8B5CF6",  # violet-500
        ]

        # â”€â”€ Black premium theme â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

                # â”€â”€ Intersection marker at power_target â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

            # â”€â”€ Target line â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

            # â”€â”€ Axes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            ax.set_xlim(0, max_mde * 100)
            ax.set_ylim(0, 105)
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
            ax.set_xlabel("Real Impact (%)", fontsize=12, color="white")
            ax.set_ylabel("Power", fontsize=12, color="white")

            # â”€â”€ Title + scenario subtitle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            pct = s.get("pct_treatment", 0)
            treat_str = ", ".join(
                cl["treatment"][0]
                for cl in s.get("clusters", [])
                if cl.get("treatment")
            ) or f"{s.get('n_treatment', '?')} geos"

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
        Scatter plot of RÂ² Train vs RÂ² Test to identify overfitting risk.
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
            ax.axhline(0.6, color="#EF4444", linestyle=":", alpha=0.5, label="Min Acceptable RÂ²")
            ax.axvline(0.6, color="#EF4444", linestyle=":", alpha=0.5)

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("RÂ² (Training - In Time)", fontsize=12, color="white")
            ax.set_ylabel("RÂ² (Validation - Out of Time)", fontsize=12, color="white")
            
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("RÂ² Degradation (Train - Test)", color="white")
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

    def plot_cluster_distributions(self, scenario=0, figsize=(14, 4)):
        """
        Plot the empirical distribution (KDE + histogram) of daily KPI values
        for Treatment vs Synthetic over the pre-period, one panel per cluster.

        Useful for checking that the synthetic control matches the treatment
        distribution — not just its trajectory — before running the experiment.

        Parameters
        ----------
        scenario : int
            Scenario index (default 0).
        figsize : tuple
            Size per cluster row (width, height).
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import scipy.stats as stats

        PALETTE = ["#06B6D4", "#F59E0B", "#10B981", "#EF4444",
                   "#8B5CF6", "#3B82F6", "#F97316", "#EC4899"]

        if not hasattr(self, "_df") or self._df is None:
            print("  [plot_cluster_distributions] No underlying dataframe attached.")
            return

        scenarios = self.get("scenarios", [])
        if not scenarios or scenario >= len(scenarios):
            print("  [plot_cluster_distributions] Invalid scenario.")
            return

        clusters = scenarios[scenario].get("clusters", [])
        if not clusters:
            print("  [plot_cluster_distributions] No clusters found.")
            return

        df = self._df.copy()
        date_col = getattr(self, "_date_col", None)
        if date_col is None or date_col not in df.columns:
            date_col = df.select_dtypes(include=["datetime64"]).columns[0]
        df = df.sort_values(date_col).set_index(date_col)

        def human_fmt(x, pos):
            if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
            if abs(x) >= 1e3: return f"{x/1e3:.0f}k"
            return f"{x:,.0f}"

        n_clusters = len(clusters)
        cols = 2
        rows = (n_clusters + 1) // cols or 1

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                rows, cols,
                figsize=(figsize[0], figsize[1] * rows),
                squeeze=False,
            )
            fig.patch.set_facecolor("black")
            axes_flat = axes.flatten()

            for i, cl in enumerate(clusters):
                ax = axes_flat[i]
                ax.set_facecolor("black")
                ax.grid(True, linestyle="--", alpha=0.15, color="white")
                for spine in ax.spines.values():
                    spine.set_color("#1E1E1E")

                treat_geos = cl.get("treatment", [])
                ctrl_geos  = cl.get("control", [])
                weights    = cl.get("control_weights", [])
                if not treat_geos or not ctrl_geos or not weights:
                    continue

                color = PALETTE[i % len(PALETTE)]

                y_treat = (
                    df[treat_geos].mean(axis=1)
                    if len(treat_geos) > 1 else df[treat_geos[0]]
                ).values
                X_ctrl = df[ctrl_geos].values
                w_arr  = np.array(weights)
                y_mean = y_treat.mean() or 1e-10
                X_mean = np.where(X_ctrl.mean(axis=0) == 0, 1e-10, X_ctrl.mean(axis=0))
                y_synth = (X_ctrl / X_mean @ w_arr) * y_mean

                # Histogram (background)
                ax.hist(y_treat, bins=35, density=True,
                        color=color, alpha=0.20, label="_nolegend_")
                ax.hist(y_synth, bins=35, density=True,
                        color="#10B981", alpha=0.15, label="_nolegend_")

                # KDE curves
                try:
                    kde_t  = stats.gaussian_kde(y_treat)
                    kde_s  = stats.gaussian_kde(y_synth)
                    xmin   = min(y_treat.min(), y_synth.min())
                    xmax   = max(y_treat.max(), y_synth.max())
                    xr     = np.linspace(xmin, xmax, 300)
                    ax.plot(xr, kde_t(xr), color=color,    linewidth=2.5, label="Treatment")
                    ax.plot(xr, kde_s(xr), color="#10B981", linewidth=2.5,
                            linestyle="--", label="Synthetic")
                except Exception:
                    pass

                # Mean lines
                ax.axvline(y_treat.mean(), color=color,    linestyle=":",
                           linewidth=1.5, alpha=0.8)
                ax.axvline(y_synth.mean(), color="#10B981", linestyle=":",
                           linewidth=1.5, alpha=0.8)

                treat_str = ", ".join(treat_geos)
                ax.set_title(
                    f"C{i} — {treat_str}",
                    color="white", fontweight="bold", fontsize=11,
                )
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(human_fmt))
                ax.tick_params(colors="#CBD5E1", labelsize=9)

                try:
                    from scipy.stats import wasserstein_distance
                    wdist = wasserstein_distance(y_treat, y_synth)
                    wdist_pct = wdist / (y_treat.mean() or 1e-10) * 100
                    wdist_str = f"{wdist_pct:.1f}%"
                except Exception:
                    wdist_str = "N/A"

                bbox = dict(boxstyle="round,pad=0.4", facecolor="#111111",
                            edgecolor="#333333", alpha=0.8)
                info = (
                    f"μ={y_treat.mean():,.0f}\n"
                    f"Treatment  σ={y_treat.std():,.0f}\n"
                    f"Synthetic  σ={y_synth.std():,.0f}\n"
                    f"Wasserstein={wdist_str}"
                )
                ax.text(0.97, 0.95, info, transform=ax.transAxes, fontsize=9,
                        va="top", ha="right", color="white",
                        bbox=bbox, family="monospace")

                if i == 0:
                    ax.legend(loc="upper left", framealpha=0.4,
                              facecolor="#111111", edgecolor="#333333",
                              labelcolor="white", fontsize=9)

            for j in range(i + 1, len(axes_flat)):
                fig.delaxes(axes_flat[j])

            fig.text(0.5, 0.98, "Pre-Period KPI Distribution (Treatment vs Synthetic)",
                     ha="center", va="top", fontsize=15, fontweight="bold", color="white")
            fig.text(0.5, 0.95, f"Scenario {scenario}  •  Daily values over pre-period",
                     ha="center", va="top", fontsize=11, color="#94A3B8")
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()

    def plot_consolidated_distribution(self, scenario=0, figsize=(12, 5)):
        """
        Plot the empirical distribution (KDE + histogram) of daily KPI values
        for the consolidated (summed) Treatment vs Synthetic over the pre-period.

        Parameters
        ----------
        scenario : int
            Scenario index (default 0).
        figsize : tuple
            Figure size.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import scipy.stats as stats

        if not hasattr(self, "_df") or self._df is None:
            print("  [plot_consolidated_distribution] No underlying dataframe attached.")
            return

        scenarios = self.get("scenarios", [])
        if not scenarios or scenario >= len(scenarios):
            print("  [plot_consolidated_distribution] Invalid scenario.")
            return

        clusters = scenarios[scenario].get("clusters", [])
        if not clusters:
            print("  [plot_consolidated_distribution] No clusters found.")
            return

        df = self._df.copy()
        date_col = getattr(self, "_date_col", None)
        if date_col is None or date_col not in df.columns:
            date_col = df.select_dtypes(include=["datetime64"]).columns[0]
        df = df.sort_values(date_col).set_index(date_col)

        agg_treat = np.zeros(len(df))
        agg_synth = np.zeros(len(df))

        for cl in clusters:
            treat_geos = cl.get("treatment", [])
            ctrl_geos  = cl.get("control", [])
            weights    = cl.get("control_weights", [])
            if not treat_geos or not ctrl_geos or not weights:
                continue
            y_t = (
                df[treat_geos].mean(axis=1)
                if len(treat_geos) > 1 else df[treat_geos[0]]
            ).values
            X_ctrl = df[ctrl_geos].values
            w_arr  = np.array(weights)
            y_mean = y_t.mean() or 1e-10
            X_mean = np.where(X_ctrl.mean(axis=0) == 0, 1e-10, X_ctrl.mean(axis=0))
            agg_treat += y_t
            agg_synth += (X_ctrl / X_mean @ w_arr) * y_mean

        def human_fmt(x, pos):
            if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
            if abs(x) >= 1e3: return f"{x/1e3:.0f}k"
            return f"{x:,.0f}"

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            ax.grid(True, linestyle="--", alpha=0.15, color="white")
            for spine in ax.spines.values():
                spine.set_color("#1E1E1E")

            ax.hist(agg_treat, bins=40, density=True,
                    color="#06B6D4", alpha=0.20, label="_nolegend_")
            ax.hist(agg_synth, bins=40, density=True,
                    color="#10B981", alpha=0.15, label="_nolegend_")

            try:
                xmin = min(agg_treat.min(), agg_synth.min())
                xmax = max(agg_treat.max(), agg_synth.max())
                xr   = np.linspace(xmin, xmax, 300)
                kde_t = stats.gaussian_kde(agg_treat)
                kde_s = stats.gaussian_kde(agg_synth)
                ax.plot(xr, kde_t(xr), color="#06B6D4", linewidth=2.5,
                        label="Total Treatment")
                ax.plot(xr, kde_s(xr), color="#10B981", linewidth=2.5,
                        linestyle="--", label="Total Synthetic")
            except Exception:
                pass

            ax.axvline(agg_treat.mean(), color="#06B6D4", linestyle=":",
                       linewidth=1.5, alpha=0.8)
            ax.axvline(agg_synth.mean(), color="#10B981", linestyle=":",
                       linewidth=1.5, alpha=0.8)

            ax.xaxis.set_major_formatter(ticker.FuncFormatter(human_fmt))
            ax.tick_params(colors="#CBD5E1", labelsize=10)

            pct = scenarios[scenario].get("pct_treatment", 0)
            fig.text(0.5, 0.98,
                     "Consolidated Pre-Period KPI Distribution (Treatment vs Synthetic)",
                     ha="center", va="top", fontsize=15, fontweight="bold", color="white")
            fig.text(0.5, 0.93,
                     f"Scenario {scenario}  •  Aggregated across {len(clusters)} clusters ({pct:.0%} tx)  •  Daily values",
                     ha="center", va="top", fontsize=11, color="#94A3B8")

            try:
                from scipy.stats import wasserstein_distance
                wdist = wasserstein_distance(agg_treat, agg_synth)
                wdist_pct = wdist / (agg_treat.mean() or 1e-10) * 100
                wdist_str = f"{wdist_pct:.1f}%"
            except Exception:
                wdist_str = "N/A"

            bbox = dict(boxstyle="round,pad=0.4", facecolor="#111111",
                        edgecolor="#333333", alpha=0.8)
            info = (
                f"μ={agg_treat.mean():,.0f}\n"
                f"Treatment  σ={agg_treat.std():,.0f}\n"
                f"Synthetic  σ={agg_synth.std():,.0f}\n"
                f"Wasserstein={wdist_str}"
            )
            ax.text(0.97, 0.95, info, transform=ax.transAxes, fontsize=10,
                    va="top", ha="right", color="white",
                    bbox=bbox, family="monospace")

            ax.legend(loc="upper left", fontsize=11, framealpha=0.4,
                      facecolor="#111111", edgecolor="#333333", labelcolor="white")

            plt.tight_layout(rect=[0, 0, 1, 0.91])
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

                dates = plot_data.get("dates")
                if dates is None:
                    df = syn.get("df")
                    if df is not None:
                        d_col = date_col if date_col and date_col in df.columns else df.columns[0]
                        dates = df.sort_values(d_col)[d_col].tolist()
                    else:
                        dates = list(range(len(y_treat)))

                if post_only:
                    y_treat = y_treat[t_idx:]
                    y_synth = y_synth[t_idx:]
                    dates = dates.iloc[t_idx:] if hasattr(dates, 'iloc') else dates[t_idx:]

                ax.plot(dates, y_treat, color="#06B6D4", linewidth=2, label="Treatment")
                ax.plot(dates, y_synth, color="#10B981", linewidth=2, linestyle="--", label="Synthetic")

                if not post_only:
                    v_line_pos = dates.iloc[t_idx] if hasattr(dates, 'iloc') else dates[t_idx]
                    ax.axvline(v_line_pos, linestyle=":", color="#EF4444", linewidth=2, alpha=0.8, label="Intervention")

                ax.set_title(f"Cluster {i} ({geo_name})", color="white", fontsize=11, fontweight="bold")
                
                ax.grid(True, linestyle="--", alpha=0.15, color="white")
                ax.tick_params(colors="#CBD5E1", labelsize=9)
                ax.yaxis.set_major_formatter(formatter)
                
                for spine in ax.spines.values():
                    spine.set_color("#1E1E1E")
                
                if dates and not isinstance(dates[0], int):
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
        
        dates = None
        agg_treat = None
        agg_synth = None
        t_idx = None

        for res in results:
            plot_data = res.get("synthetic", {}).get("plotting_data", {})
            if not plot_data or "y" not in plot_data:
                continue
            y_arr = np.array(plot_data["y"])
            s_arr = np.array(plot_data["synthetic"])
            if agg_treat is None:
                agg_treat = y_arr.copy()
                agg_synth = s_arr.copy()
                dates = plot_data.get("dates")
            else:
                agg_treat += y_arr
                agg_synth += s_arr
            t_idx = plot_data["treatment_idx"]

        if agg_treat is None:
            print("  [plot_consolidated_effect] No plotting data found in results.")
            return

        if dates is None:
            dates = list(range(len(agg_treat)))

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
                v_line_pos = dates.iloc[t_idx] if hasattr(dates, 'iloc') else dates[t_idx]
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

    def plot_consolidated_lift_distributions(self, show_null=False, figsize=(15, 6)):
        """
        Plot the bootstrap distributions of the Absolute and Percentual Lifts.
        Shows histograms with KDE curves and 95% Confidence Intervals.
        If show_null=True, overlays the theoretical Null Hypothesis (H0) distribution.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import scipy.stats as stats
        from ._bootstrap import bootstrap_significance

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
            fig.text(0.5, 0.92, "Empirical distributions generated via Circular Moving Block Bootstrap (CMBB)", ha="center", va="top", fontsize=12, color="#94A3B8")

            plt.tight_layout(rect=[0, 0, 1, 0.90])
            plt.show()

    def plot_cluster_lift_distributions(self, show_null=False, figsize=None):
        """
        Plot bootstrap lift distributions for each cluster individually.

        One row per cluster, two columns (absolute on the left, percentual on
        the right).  Reuses the CMBB bootstrap arrays already computed during
        inference — no re-sampling needed.

        Parameters
        ----------
        show_null : bool
            If True, overlays the null hypothesis (H0) distribution centred at 0.
        figsize : tuple, optional
            Figure size.  Defaults to (14, 5 * n_clusters).
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import scipy.stats as stats

        PALETTE = ["#06B6D4", "#F59E0B", "#10B981", "#EF4444",
                   "#8B5CF6", "#3B82F6", "#F97316", "#EC4899"]

        def human_format(x, pos):
            if abs(x) >= 1e9: return f"{x/1e9:.1f}B"
            if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
            if abs(x) >= 1e3: return f"{x/1e3:.1f}k"
            return f"{x:,.0f}"

        results = self.get("results", [])
        if not results:
            print("  [plot_cluster_lift_distributions] No experiment results found.")
            return

        n_clusters = len(results)
        fig_w = figsize[0] if figsize else 14
        fig_h = figsize[1] if figsize else 5 * n_clusters

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                n_clusters, 2,
                figsize=(fig_w, fig_h),
                squeeze=False,
            )
            fig.patch.set_facecolor("black")

            for i, res in enumerate(results):
                boot      = res.get("synthetic", {}).get("bootstrap", {})
                plot_data = res.get("synthetic", {}).get("plotting_data", {})
                if not boot or not plot_data:
                    continue

                treat_geo = plot_data.get("treatment_geo", f"Cluster {i}")
                if isinstance(treat_geo, list):
                    treat_geo = ", ".join(treat_geo)
                color = PALETTE[i % len(PALETTE)]

                boot_abs = np.array(boot["boot_totals_abs"])
                boot_pct = np.array(boot["boot_totals_pct"]) * 100
                ci_l_abs = boot["ci_lower_total_abs"]
                ci_u_abs = boot["ci_upper_total_abs"]
                ci_l_pct = boot["ci_lower_total_pct"] * 100
                ci_u_pct = boot["ci_upper_total_pct"] * 100

                post_real  = np.array(plot_data["post_real"])
                post_synth = np.array(plot_data["post_synth"])
                tot_synth  = float(np.sum(post_synth)) or 1e-10
                obs_abs    = float(np.sum(post_real) - np.sum(post_synth))
                obs_pct    = obs_abs / tot_synth * 100

                bbox_props = dict(
                    boxstyle="round,pad=0.4", facecolor="#111111",
                    edgecolor="#333333", alpha=0.8,
                )

                for j, (data, ci_l, ci_u, obs, is_pct) in enumerate([
                    (boot_abs, ci_l_abs, ci_u_abs, obs_abs, False),
                    (boot_pct, ci_l_pct, ci_u_pct, obs_pct, True),
                ]):
                    ax = axes[i, j]
                    ax.set_facecolor("black")
                    ax.grid(True, linestyle="--", alpha=0.15, color="white")
                    for spine in ax.spines.values():
                        spine.set_color("#1E1E1E")

                    std_data = np.std(data)
                    ax.hist(data, bins=40, density=True,
                            color=color, alpha=0.25, edgecolor=None)

                    try:
                        if show_null:
                            null_data = data - np.mean(data)
                            null_kde  = stats.gaussian_kde(null_data)
                            x_null = np.linspace(
                                min(null_data) - std_data,
                                max(null_data) + std_data, 200,
                            )
                            ax.plot(x_null, null_kde(x_null),
                                    color="#EF4444", linewidth=2, label="Null (H0)")
                            ax.fill_between(x_null, 0, null_kde(x_null),
                                            color="#EF4444", alpha=0.12)
                            ax.axvline(0, color="#EF4444", linestyle="--",
                                       linewidth=1.5, alpha=0.8)
                        else:
                            ax.axvline(0, color="#EF4444", linestyle="-",
                                       linewidth=2, label="Null (0)")

                        kde    = stats.gaussian_kde(data)
                        x_eval = np.linspace(
                            min(data) - std_data, max(data) + std_data, 200
                        )
                        ax.plot(x_eval, kde(x_eval),
                                color=color, linewidth=2.5, label="Observed Lift (H1)")
                        x_fill = np.linspace(ci_l, ci_u, 100)
                        ax.fill_between(x_fill, 0, kde(x_fill),
                                        color=color, alpha=0.25)
                    except Exception:
                        if not show_null:
                            ax.axvline(0, color="#EF4444", linestyle="-",
                                       linewidth=2, label="Null (0)")

                    ax.axvline(obs,  color="#FBBF24", linestyle="--",
                               linewidth=2, label="Point Estimate")
                    ax.axvline(ci_l, color="#94A3B8", linestyle=":",
                               linewidth=2, label="95% CI")
                    ax.axvline(ci_u, color="#94A3B8", linestyle=":",
                               linewidth=2)

                    kind = "Percentual" if is_pct else "Absolute"
                    ax.set_title(
                        f"C{i} — {treat_geo}  |  {kind} Lift",
                        color="white", fontweight="bold", fontsize=11,
                    )
                    ax.tick_params(colors="#CBD5E1", labelsize=9)

                    if is_pct:
                        ax.xaxis.set_major_formatter(
                            ticker.PercentFormatter(xmax=100, decimals=1)
                        )
                        info = (
                            f"Lift: {obs_pct:.2f}%\n"
                            f"95% CI: [{ci_l_pct:.2f}%, {ci_u_pct:.2f}%]\n"
                            f"Std: {np.std(boot_pct):.2f}%"
                        )
                    else:
                        ax.xaxis.set_major_formatter(
                            ticker.FuncFormatter(human_format)
                        )
                        fmt = lambda v: f"{v:,.0f}"
                        info = (
                            f"Lift: {fmt(obs_abs)}\n"
                            f"95% CI: [{fmt(ci_l_abs)}, {fmt(ci_u_abs)}]\n"
                            f"Std: {fmt(np.std(boot_abs))}"
                        )

                    ax.text(
                        0.03, 0.95, info,
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment="top", color="white",
                        bbox=bbox_props, family="monospace",
                    )
                    ax.legend(
                        loc="upper right", framealpha=0.4,
                        facecolor="#111111", edgecolor="#333333",
                        labelcolor="white", fontsize=8,
                    )

            fig.suptitle(
                "Cluster-Level Treatment Effect (Bootstrap Distribution)",
                color="white", fontsize=15, fontweight="bold",
            )
            fig.text(
                0.5, 0.98,
                "Empirical distributions generated via Circular Moving Block Bootstrap (CMBB)",
                ha="center", va="top", fontsize=11, color="#94A3B8",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()
