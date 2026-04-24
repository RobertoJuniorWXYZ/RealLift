import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import hashlib
import os
import math


# ─── PAGE DIMENSIONS ─────────────────────────────────────────────────────────
A4_W = 8.27
A4_H = 11.69

MH = 1.0 / 21.0
MV = 1.0 / 29.7
LEFT   = MH
RIGHT  = 1 - MH
TOP    = 1 - MV
BOTTOM = MV
WIDTH  = RIGHT - LEFT

# ─── TYPOGRAPHY ───────────────────────────────────────────────────────────────
H1      = 18
H1_SUB  = 10
H2      = 12
H3      = 10
H4      = 9
BODY    = 8.5
SMALL   = 7.5
CAPTION = 6.5

# ─── COLORS ───────────────────────────────────────────────────────────────────
C_TITLE      = '#1a2744'
C_SECTION    = '#2c3e50'
C_SUBSECTION = '#34495e'
C_GREEN      = '#27ae60'
C_BLUE       = '#2980b9'
C_RED        = '#c0392b'
C_ROW_ALT    = '#f7f9fc'
C_BORDER     = '#dfe6e9'
C_GRAY       = '#95a5a6'
C_TEXT       = '#333333'
C_MUTED      = '#777777'
C_LIGHT      = '#aaaaaa'

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
LINE_H      = 0.023
BLOCK_GAP   = 0.018
SECTION_GAP = 0.038
ROW_H       = 0.018


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _human_fmt(num, pos):
    mag = 0
    while abs(num) >= 1000:
        mag += 1; num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'B', 'T'][mag])


def _report_id(ts_str):
    """Short deterministic hash from timestamp for auditability."""
    return hashlib.sha256(ts_str.encode()).hexdigest()[:8].upper()


def _style_table(tbl, header_color=C_SECTION, font_size=SMALL):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C_BORDER)
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=font_size)
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor('white' if r % 2 == 1 else C_ROW_ALT)


def _header(fig, ts, report_id, logo=None, title="DATA QUALITY ASSESSMENT REPORT",
            subtitle="Automated Dataset Evaluation & Validation"):
    """Standardized 3-line report header on page 1, with optional logo."""
    fig.text(0.5, TOP - 0.010, title,
             ha='center', fontsize=H1, fontweight='bold', color=C_TITLE)
    fig.text(0.5, TOP - 0.038, subtitle,
             ha='center', fontsize=H1_SUB, color=C_MUTED, style='italic')
    fig.text(0.5, TOP - 0.058, f"Generated: {ts}   |   Report ID: {report_id}",
             ha='center', fontsize=BODY, color=C_LIGHT)

    if logo and os.path.isfile(logo):
        try:
            img = plt.imread(logo)
            ax_logo = fig.add_axes([RIGHT - 0.14, TOP - 0.06, 0.14, 0.06])
            ax_logo.imshow(img, aspect='equal')
            ax_logo.axis('off')
        except Exception:
            pass


def _page_header(fig, ts=None, report_id=None, title="DATA QUALITY ASSESSMENT REPORT"):
    """Compact header for pages 2+."""
    fig.text(0.5, TOP - 0.005, title,
             ha='center', fontsize=H4, fontweight='bold', color=C_LIGHT)


def _footer(fig, report_id):
    """Standardized footer with pipeline identity."""
    fig.text(LEFT, BOTTOM * 0.35,
             f"DataOps Automated Pipeline  ·  Report {report_id}",
             fontsize=CAPTION, color=C_LIGHT)
    fig.text(RIGHT, BOTTOM * 0.35, "RealLift",
             ha='right', fontsize=CAPTION, color=C_LIGHT, fontweight='bold')
    line = plt.Line2D([LEFT, RIGHT], [BOTTOM * 0.6, BOTTOM * 0.6],
                      transform=fig.transFigure, color=C_BORDER, linewidth=0.5)
    fig.add_artist(line)


def _section(fig, y, text):
    """Section title with underline."""
    fig.text(LEFT, y, text, fontsize=H2, fontweight='bold', color=C_SECTION)
    line = plt.Line2D([LEFT, LEFT + 0.32], [y - 0.008, y - 0.008],
                      transform=fig.transFigure, color=C_BORDER, linewidth=1.0)
    fig.add_artist(line)
    return y - 0.028


def _block_title(fig, y, text):
    """Sub-block title (Dataset, Data Quality, Result)."""
    fig.text(LEFT + 0.02, y, text, fontsize=H3, fontweight='bold', color=C_SUBSECTION)
    return y - LINE_H


def _kv(fig, y, key, value):
    """Key-value pair with fixed alignment."""
    fig.text(LEFT + 0.04, y, key, fontsize=H4, color=C_MUTED)
    fig.text(LEFT + 0.30, y, str(value), fontsize=H4, color=C_TEXT, fontweight='bold')
    return y - LINE_H


def _bullet(fig, y, text):
    """Methodology bullet point."""
    fig.text(LEFT + 0.04, y, "•", fontsize=BODY, color=C_SUBSECTION)
    fig.text(LEFT + 0.06, y, text, fontsize=BODY, color=C_TEXT)
    return y - 0.017


def _draw_table_pages(pdf, title, subtitle, col_labels, rows,
                      header_color, report_id, font_size=SMALL, rows_per_page=46,
                      page_title="DATA QUALITY ASSESSMENT REPORT"):
    """Paginated table with consistent header/footer."""
    n_pages = max(1, math.ceil(len(rows) / rows_per_page))

    for p in range(n_pages):
        fig = plt.figure(figsize=(A4_W, A4_H))
        plt.axis('off')

        _page_header(fig, None, report_id, title=page_title)

        pg = f"  ({p+1}/{n_pages})" if n_pages > 1 else ""
        fig.text(0.5, TOP - 0.035, title + pg,
                 ha='center', fontsize=H2, fontweight='bold', color=C_TITLE)

        if subtitle:
            fig.text(0.5, TOP - 0.058, subtitle,
                     ha='center', fontsize=BODY, color=C_MUTED, style='italic')

        si = p * rows_per_page
        ei = min(si + rows_per_page, len(rows))
        page_rows = rows[si:ei]
        n = len(page_rows)

        t_top = TOP - 0.08
        t_h = ROW_H * (n + 1)
        t_bot = max(t_top - t_h, BOTTOM + 0.03)

        ax = fig.add_axes([LEFT, t_bot, WIDTH, t_top - t_bot])
        ax.axis('off')
        tbl = ax.table(cellText=page_rows, colLabels=col_labels,
                       loc='upper center', cellLoc='center')
        _style_table(tbl, header_color=header_color, font_size=font_size)
        tbl.auto_set_column_width(list(range(len(col_labels))))

        _footer(fig, report_id)
        pdf.savefig(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cleaning_report(
    pdf_name,
    meta_info,
    vol_df=None,
    final_score_df=None,
    df_raw=None,
    df_cleaned=None,
    date_col=None,
    imputation_method='interpolation',
    logo=None
):
    """
    Generates an auditable A4 PDF report for DataOps diagnostics.
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rid = _report_id(ts)
    m = meta_info

    with PdfPages(pdf_name) as pdf:

        # ══════════════════════════════════════════════════════════════════
        # PAGE 1: SUMMARY + METHODOLOGY
        # ══════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(A4_W, A4_H))
        plt.axis('off')

        _header(fig, ts, rid, logo=logo)

        # ── DATASET SUMMARY ──
        y = _section(fig, TOP - 0.09, "DATASET SUMMARY")

        # Block 1: Dataset
        y = _block_title(fig, y, "Dataset")
        retention = (m['final_geos'] / m['initial_geos'] * 100) if m['initial_geos'] > 0 else 0
        y = _kv(fig, y, "Timeline", f"{m['start']}  →  {m['end']}")
        y = _kv(fig, y, "Duration", f"{m['days']} days")
        y = _kv(fig, y, "Entities", f"{m['initial_geos']:,} initial  →  {m['final_geos']:,} selected  ({retention:.1f}%)")

        y -= BLOCK_GAP

        # Block 2: Data Quality
        y = _block_title(fig, y, "Data Quality")
        y = _kv(fig, y, "Original matrix", f"{m['total_cells_original']:,} cells  ·  {m['imputed_cells_original']:,} imputed  ({m['imputed_pct_original']:.1f}%)")
        y = _kv(fig, y, "Selected matrix", f"{m['total_cells']:,} cells  ·  {m['imputed_cells']:,} imputed  ({m['imputed_pct']:.1f}%)")
        y = _kv(fig, y, "Method", f"{m['method'].upper()}  (constant: {m['constant']})")

        y -= BLOCK_GAP

        # Block 3: Result
        y = _block_title(fig, y, "Result")
        y = _kv(fig, y, "Entities selected", str(m['n_sel']))
        y = _kv(fig, y, "Volume retained", m['pct_total_sel'])
        y = _kv(fig, y, "Avg vol / entity", m['avg_vol_sel'])

        y -= SECTION_GAP

        # ── METHODOLOGY ──
        y = _section(fig, y, "METHODOLOGY")

        y = _bullet(fig, y, "Exclude entities with excessive zero or missing values.")
        y = _bullet(fig, y, "Segment entities via quantile-based volume ranking.")
        y = _bullet(fig, y, "Retain high-signal entities that represent core market activity.")
        y = _bullet(fig, y, "Apply imputation to ensure numerical continuity.")
        y = _bullet(fig, y, "Track imputation impact per entity in the scorecard.")

        y -= BLOCK_GAP

        fig.text(LEFT + 0.04, y, "Key Metrics", fontsize=H4, fontweight='bold', color=C_SUBSECTION)
        y -= LINE_H
        y = _kv(fig, y, "% Zeros", "Share of original data points that were zero or missing.")
        y = _kv(fig, y, "% Impact", "Share of final volume generated by imputation.")
        y = _kv(fig, y, "% of Original", "Volume coverage of selected entities vs. total market.")

        _footer(fig, rid)
        pdf.savefig(fig)
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════
        # DISTRIBUTION ANALYSIS
        # ══════════════════════════════════════════════════════════════════
        if vol_df is not None:
            global_vol = vol_df["Sum_Original"].sum()
            unique_qs = vol_df['Quantile'].unique()
            nq = len(unique_qs)

            q_rows = []
            cum = 0.0
            for q in unique_qs:
                qd = vol_df[vol_df['Quantile'] == q]
                ng = len(qd)
                az = f"{qd['Zero_Rate'].mean():.1f}%"
                sv = qd['Sum_Original'].sum()
                pv = (sv / global_vol * 100) if global_vol > 0 else 0.0
                cum += pv
                av = f"{(sv / ng):,.0f}" if ng > 0 else "—"
                q_rows.append([q, ng, az, f"{sv:,.0f}", f"{pv:.1f}%", f"{cum:.1f}%", av])

            _draw_table_pages(
                pdf,
                "DISTRIBUTION ANALYSIS",
                f"Entities segmented into {nq} quantiles based on total volume.",
                ["Quantile", "Entities", "Avg Zeros", "Total Volume", "% Total", "Cumulative", "Avg Vol/Entity"],
                q_rows,
                header_color=C_SECTION,
                report_id=rid,
                font_size=SMALL,
                rows_per_page=46
            )

        # ══════════════════════════════════════════════════════════════════
        # ENTITY SCORECARD
        # ══════════════════════════════════════════════════════════════════
        if final_score_df is not None and not final_score_df.empty:
            s_rows = []
            for _, row in final_score_df.iterrows():
                gn = str(row['Geo'])
                if len(gn) > 30: gn = gn[:27] + "..."
                s_rows.append([
                    gn,
                    int(row['N_Imputed']),
                    f"{row['Zero_Rate']:.2f}%",
                    f"{row['Sum_Interpolated']:,.0f}",
                    f"{row['Sum_Original']:,.0f}",
                    f"{row['Pct_Interpolated']:.2f}%"
                ])

            n_ent = len(final_score_df)
            _draw_table_pages(
                pdf,
                f"ENTITY SCORECARD  —  {n_ent} ENTITIES",
                "Sorted by data quality (ascending zero-rate), then by volume (descending).",
                ["Entity", "Imputed", "% Zeros", "Σ Imputed Vol", "Σ Original Vol", "% Impact"],
                s_rows,
                header_color=C_BLUE,
                report_id=rid,
                font_size=SMALL,
                rows_per_page=46
            )

        # ══════════════════════════════════════════════════════════════════
        # TIME SERIES VALIDATION
        # ══════════════════════════════════════════════════════════════════
        if df_raw is not None and df_cleaned is not None and date_col is not None:
            geo_cols = [c for c in df_cleaned.columns if c != date_col]
            top_vols = df_cleaned[geo_cols].sum().sort_values(ascending=False)
            n_plot = min(12, len(top_vols))
            cols_plot = top_vols.head(n_plot).index.tolist()

            fig_v = plt.figure(figsize=(A4_W, A4_H))

            _page_header(fig_v, ts, rid)

            fig_v.text(0.5, TOP - 0.035, "TIME SERIES VALIDATION",
                       ha='center', fontsize=H2, fontweight='bold', color=C_TITLE)
            fig_v.text(0.5, TOP - 0.06,
                       f"Comparison before and after imputation for the top {n_plot} selected entities.",
                       ha='center', fontsize=BODY, color=C_MUTED, style='italic')

            px = LEFT + 0.04
            pw = WIDTH - 0.08

            # BEFORE
            ax1 = fig_v.add_axes([px, 0.53, pw, 0.34])
            for c in cols_plot:
                ax1.plot(df_raw[date_col], df_raw[c], alpha=0.5, linewidth=0.8, label=c)
            ax1.set_title("BEFORE IMPUTATION", fontsize=H3, fontweight='bold', color=C_RED, pad=10)
            ax1.grid(True, linestyle='--', alpha=0.12)
            ax1.yaxis.set_major_formatter(FuncFormatter(_human_fmt))
            ax1.legend(loc='upper left', fontsize=5, ncol=3, framealpha=0.7)
            ax1.tick_params(labelsize=CAPTION)

            # AFTER
            ax2 = fig_v.add_axes([px, 0.10, pw, 0.34])
            for c in cols_plot:
                ax2.plot(df_cleaned[date_col], df_cleaned[c], alpha=0.5, linewidth=0.8, label=c)
            ax2.set_title(f"AFTER IMPUTATION  ({imputation_method.upper()})", fontsize=H3, fontweight='bold', color=C_GREEN, pad=10)
            ax2.grid(True, linestyle='--', alpha=0.12)
            ax2.yaxis.set_major_formatter(FuncFormatter(_human_fmt))
            ax2.legend(loc='upper left', fontsize=5, ncol=3, framealpha=0.7)
            ax2.tick_params(labelsize=CAPTION)

            _footer(fig_v, rid)
            pdf.savefig(fig_v)
            plt.close(fig_v)


# ═══════════════════════════════════════════════════════════════════════════════
# DOE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_doe_report(
    pdf_name,
    doe_result,
    design_meta,
    logo=None
):
    """
    Generates an auditable A4 PDF report for Design of Experiments results.

    Parameters:
        pdf_name (str): Output PDF file path.
        doe_result (dict): Return value of design_of_experiments().
        design_meta (dict): Metadata dict with keys:
            n_geos, search_mode, experiment_type, experiment_days, mde,
            pre_start, end_date, n_folds
        logo (str): Optional path to logo image.
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rid = _report_id(ts)
    scenarios = doe_result["scenarios"]
    comparison = doe_result["comparison"]
    exp_type = doe_result["experiment_type"]
    is_did = exp_type == "matched_did"
    m = design_meta

    with PdfPages(pdf_name) as pdf:

        # ══════════════════════════════════════════════════════════════════
        # PAGE 1: DESIGN SUMMARY
        # ══════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(A4_W, A4_H))
        plt.axis('off')

        # Header
        fig.text(0.5, TOP - 0.010, "DESIGN OF EXPERIMENT REPORT",
                 ha='center', fontsize=H1, fontweight='bold', color=C_TITLE)
        fig.text(0.5, TOP - 0.038, "Automated Scenario Analysis & Power Estimation",
                 ha='center', fontsize=H1_SUB, color=C_MUTED, style='italic')
        fig.text(0.5, TOP - 0.058, f"Generated: {ts}   |   Report ID: {rid}",
                 ha='center', fontsize=BODY, color=C_LIGHT)

        if logo and os.path.isfile(logo):
            try:
                img = plt.imread(logo)
                ax_logo = fig.add_axes([RIGHT - 0.14, TOP - 0.06, 0.14, 0.06])
                ax_logo.imshow(img, aspect='equal')
                ax_logo.axis('off')
            except Exception:
                pass

        # ── DESIGN PARAMETERS ──
        y = _section(fig, TOP - 0.09, "DESIGN PARAMETERS")

        exp_days_str = ", ".join(str(d) for d in m.get("experiment_days", [])) if m.get("experiment_days") else "N/A"
        mde_str = f"{m['mde']*100:.1f}%" if m.get("mde") else "Auto (computed per duration)"

        y = _kv(fig, y, "Experiment type", exp_type.replace("_", " ").title())
        y = _kv(fig, y, "Search mode", m.get("search_mode", "N/A"))
        y = _kv(fig, y, "Total geos", str(m.get("n_geos", "N/A")))
        y = _kv(fig, y, "Pre-treatment", f"{m.get('pre_start', 'N/A')}  →  {m.get('end_date', 'N/A')}")
        y = _kv(fig, y, "OOF folds", str(m.get("n_folds", 5)))
        y = _kv(fig, y, "Target MDE", mde_str)
        y = _kv(fig, y, "Candidate days", exp_days_str)
        y = _kv(fig, y, "Scenarios", str(len(scenarios)))

        y -= SECTION_GAP

        # ── SCENARIO COMPARISON ──
        y = _section(fig, y, "SCENARIO COMPARISON")

        if comparison is not None and not comparison.empty:
            is_auto = m.get("mde") is None
            comp_rows = []

            if is_auto:
                days_to_eval = m.get("experiment_days", [21, 30, 60])
                col_labels = ["Scenario", "% Treated", "Clusters", "Distinct", "Controls"]
                col_labels += [f"MDE @{d}d" for d in days_to_eval]

                for _, row in comparison.iterrows():
                    r = [int(row["Scenario"]), row["% Treated"], int(row["Clusters"]),
                         int(row["Distinct"]), int(row["Controls"])]
                    for d in days_to_eval:
                        r.append(row.get(f"mde_{d}d", "N/A"))
                    comp_rows.append(r)
            else:
                col_labels = ["Scenario", "% Treated", "Clusters", "Distinct", "Controls", "Min Days", "Power"]
                for _, row in comparison.iterrows():
                    comp_rows.append([
                        int(row["Scenario"]), row["% Treated"], int(row["Clusters"]),
                        int(row["Distinct"]), int(row["Controls"]),
                        row.get("best_days", "N/A"), row.get("best_power", "N/A")
                    ])

            # Inline table
            ax_comp = fig.add_axes([LEFT, y - 0.06, WIDTH, 0.06])
            ax_comp.axis('off')
            tbl = ax_comp.table(cellText=comp_rows, colLabels=col_labels,
                                loc='upper center', cellLoc='center')
            _style_table(tbl, header_color=C_SECTION, font_size=SMALL)
            tbl.auto_set_column_width(list(range(len(col_labels))))

        _footer(fig, rid)
        pdf.savefig(fig)
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════
        # PER-SCENARIO PAGES
        # ══════════════════════════════════════════════════════════════════
        for s_idx, scenario in enumerate(scenarios):
            if scenario.get("clusters") is None:
                continue

            clusters = scenario["clusters"]
            duration = scenario["duration"]
            cv_summary = scenario.get("validation")
            n_treat = scenario["n_treatment"]
            pct = scenario["pct_treatment"]
            cluster_results = duration["cluster_results"]
            consolidated = duration["consolidated"]

            fig_s = plt.figure(figsize=(A4_W, A4_H))
            plt.axis('off')
            _page_header(fig_s, ts, rid, title="DESIGN OF EXPERIMENT REPORT")

            # Title
            fig_s.text(0.5, TOP - 0.035,
                       f"SCENARIO {s_idx + 1}  —  {pct:.0%} TREATMENT  ({n_treat} geo{'s' if n_treat > 1 else ''})",
                       ha='center', fontsize=H2, fontweight='bold', color=C_TITLE)

            y = TOP - 0.07

            # ── CLUSTER MDE / DURATION TABLE ──
            y = _section(fig_s, y, "CLUSTER ANALYSIS")
            is_auto = m.get("mde") is None

            if is_auto:
                days_list = m.get("experiment_days", [21, 30, 60])
                cl_labels = ["Cluster", "Treatment", "Controls"]
                cl_labels += [f"MDE @{d}d" for d in days_list]

                cl_rows = []
                for i, (cl, cr) in enumerate(zip(clusters, cluster_results)):
                    treat = ", ".join(cl["treatment"])
                    weights = cl.get("control_weights", [])
                    n_ctrl = sum(1 for w in weights if w > 0.001) if weights else len(cl["control"])

                    r = [i, treat, n_ctrl]
                    curve = cr.get("mde_curve")
                    for d in days_list:
                        if curve is not None:
                            val = curve.loc[curve["days"] == d, "mde"]
                            r.append(f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A")
                        else:
                            r.append("N/A")
                    cl_rows.append(r)

                # Consolidated row
                c_curve = consolidated.get("mde_curve")
                c_row = ["CONSOL.", "pooled", ""]
                for d in days_list:
                    if c_curve is not None:
                        val = c_curve.loc[c_curve["days"] == d, "mde"]
                        c_row.append(f"{val.values[0]*100:.2f}%" if len(val) > 0 else "N/A")
                    else:
                        c_row.append("N/A")
                cl_rows.append(c_row)
            else:
                cl_labels = ["Cluster", "Treatment", "Controls", "Min Days", "Power"]
                cl_rows = []
                for i, (cl, cr) in enumerate(zip(clusters, cluster_results)):
                    treat = ", ".join(cl["treatment"])
                    weights = cl.get("control_weights", [])
                    n_ctrl = sum(1 for w in weights if w > 0.001) if weights else len(cl["control"])
                    bd = cr["summary"].get("best_days")
                    bp = cr["summary"].get("best_power")
                    cl_rows.append([i, treat, n_ctrl,
                                    f"{int(bd)}d" if bd else "N/A",
                                    f"{bp:.1%}" if bp else "N/A"])
                c_sum = consolidated["summary"]
                cl_rows.append(["CONSOL.", "pooled", "",
                                f"{int(c_sum['best_days'])}d" if c_sum.get("best_days") else "N/A",
                                f"{c_sum['best_power']:.1%}" if c_sum.get("best_power") else "N/A"])

            n_cl_rows = len(cl_rows)
            tbl_h = ROW_H * (n_cl_rows + 1)
            ax_cl = fig_s.add_axes([LEFT, y - tbl_h - 0.01, WIDTH, tbl_h + 0.01])
            ax_cl.axis('off')
            tbl_cl = ax_cl.table(cellText=cl_rows, colLabels=cl_labels,
                                 loc='upper center', cellLoc='center')
            _style_table(tbl_cl, header_color=C_SECTION, font_size=SMALL)
            tbl_cl.auto_set_column_width(list(range(len(cl_labels))))

            y -= tbl_h + 0.04

            # ── CROSS-VALIDATION SUMMARY ──
            if cv_summary is not None and not cv_summary.empty:
                y = _section(fig_s, y, "CROSS-VALIDATION SUMMARY")

                if is_did:
                    cv_labels = ["Cluster", "Treatment", "R²"]
                    cv_rows = []
                    for i, row in cv_summary.iterrows():
                        treat = ", ".join(row["treatment"]) if isinstance(row["treatment"], list) else str(row["treatment"])
                        cv_rows.append([i, treat, f"{row['r2_test']*100:.2f}%"])
                else:
                    cv_labels = ["Cluster", "Treatment", "R² Train", "R² Test", "MAPE Tr", "MAPE Te", "WAPE Tr", "WAPE Te"]
                    cv_rows = []
                    for i, row in cv_summary.iterrows():
                        treat = ", ".join(row["treatment"]) if isinstance(row["treatment"], list) else str(row["treatment"])
                        cv_rows.append([
                            i, treat,
                            f"{row['r2_train']*100:.2f}%", f"{row['r2_test']*100:.2f}%",
                            f"{row['mape_train']*100:.2f}%", f"{row['mape_test']*100:.2f}%",
                            f"{row['wape_train']*100:.2f}%", f"{row['wape_test']*100:.2f}%",
                        ])

                n_cv = len(cv_rows)
                cv_h = ROW_H * (n_cv + 1)
                ax_cv = fig_s.add_axes([LEFT, y - cv_h - 0.01, WIDTH, cv_h + 0.01])
                ax_cv.axis('off')
                tbl_cv = ax_cv.table(cellText=cv_rows, colLabels=cv_labels,
                                     loc='upper center', cellLoc='center')
                _style_table(tbl_cv, header_color=C_BLUE, font_size=SMALL)
                tbl_cv.auto_set_column_width(list(range(len(cv_labels))))

                y -= cv_h + 0.04

            # ── EXPERIMENTAL SCOPE ──
            all_treat_set = set()
            all_ctrl_set = set()
            for cl in clusters:
                all_treat_set.update(cl["treatment"])
                all_ctrl_set.update(cl["control"])
            distinct = len(all_treat_set | all_ctrl_set)
            total_geos = m.get("n_geos", 0)
            coverage = f"{distinct / total_geos:.0%}" if total_geos > 0 else "N/A"

            y = _section(fig_s, y, "EXPERIMENTAL SCOPE")
            y = _kv(fig_s, y, "Distinct geos used", f"{distinct}  ({coverage} coverage)")

            # Treatment pool — wrap to avoid overflow
            sorted_treats = sorted(all_treat_set)
            chunk_size = 5
            y = _kv(fig_s, y, "Treatment pool", ", ".join(sorted_treats[:chunk_size]))
            for ci in range(chunk_size, len(sorted_treats), chunk_size):
                chunk = ", ".join(sorted_treats[ci:ci + chunk_size])
                fig_s.text(LEFT + 0.30, y, chunk, fontsize=H4, color=C_TEXT, fontweight='bold')
                y -= LINE_H

            _footer(fig_s, rid)
            pdf.savefig(fig_s)
            plt.close(fig_s)

            # ── DONOR POOLS (one table per cluster) ──
            for i, cl in enumerate(clusters):
                treat_name = ", ".join(cl["treatment"])
                controls = cl["control"]
                weights = cl.get("control_weights", [])

                donors = [(d, w) for d, w in zip(controls, weights) if w > 0.001]
                donors.sort(key=lambda x: x[1], reverse=True)

                if not donors:
                    continue

                dp_rows = [[d, f"{w*100:.2f}%"] for d, w in donors]

                _draw_table_pages(
                    pdf,
                    f"DONOR POOL  —  {treat_name.upper()}",
                    f"Cluster {i}: {len(donors)} donor{'s' if len(donors) != 1 else ''} with weight > 0.001, sorted by weight (descending).",
                    ["Donor", "Weight"],
                    dp_rows,
                    header_color=C_SECTION,
                    report_id=rid,
                    font_size=SMALL,
                    rows_per_page=46,
                    page_title="DESIGN OF EXPERIMENT REPORT"
                )

