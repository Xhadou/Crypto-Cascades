"""
Visualization Module

Creates publication-quality figures for FOMO contagion research:
1. SEIR state trajectories over time
2. Network visualization with state coloring
3. Fear & Greed Index correlation plots
4. R₀ sensitivity analysis
5. Community structure heatmaps
6. Parameter confidence intervals
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union
import logging

from src.state_engine.state_assigner import State
from src.estimation.estimator import EstimationResult
from src.hypothesis.hypothesis_tester import HypothesisResult
from src.utils.logger import get_logger


# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
})

# Color palette for states
STATE_COLORS = {
    'S': '#2E86AB',  # Blue - Susceptible
    'E': '#F6AE2D',  # Orange - Exposed
    'I': '#E94F37',  # Red - Infected
    'R': '#7FB069',  # Green - Recovered
}


class SEIRVisualizer:
    """Visualization tools for SEIR epidemic model results."""
    
    def __init__(self, output_dir: str = "figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        self.logger = get_logger(__name__)
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_seir_trajectory(
        self,
        results: pd.DataFrame,
        title: str = "SEIR Dynamics",
        fgi_values: Optional[np.ndarray] = None,
        show_uncertainty: bool = False,
        uncertainty_data: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot SEIR state trajectories over time.
        
        Args:
            results: DataFrame with columns [t, S_frac, E_frac, I_frac, R_frac]
            title: Plot title
            fgi_values: Optional FGI values for secondary axis
            show_uncertainty: Whether to show uncertainty bands
            uncertainty_data: Monte Carlo statistics (from NetworkSEIR.run_monte_carlo)
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        t = np.asarray(results['t'].values)
        
        # Plot SEIR curves - use np.asarray for type compatibility
        ax1.plot(t, np.asarray(results['S_frac'].values), color=STATE_COLORS['S'], label='Susceptible', linewidth=2)
        ax1.plot(t, np.asarray(results['E_frac'].values), color=STATE_COLORS['E'], label='Exposed', linewidth=2)
        ax1.plot(t, np.asarray(results['I_frac'].values), color=STATE_COLORS['I'], label='Infected', linewidth=2)
        ax1.plot(t, np.asarray(results['R_frac'].values), color=STATE_COLORS['R'], label='Recovered', linewidth=2)
        
        # Uncertainty bands
        if show_uncertainty and uncertainty_data is not None:
            for state in ['S_frac', 'E_frac', 'I_frac', 'R_frac']:
                color = STATE_COLORS[state[0]]
                ax1.fill_between(
                    uncertainty_data['t'],
                    uncertainty_data[state]['q05'],
                    uncertainty_data[state]['q95'],
                    color=color, alpha=0.2
                )
        
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Population Fraction')
        ax1.set_ylim(0, 1)
        ax1.legend(loc='center right')
        ax1.grid(True, alpha=0.3)
        
        # Secondary axis for FGI
        if fgi_values is not None and len(fgi_values) > 0:
            ax2 = ax1.twinx()
            fgi_t = np.linspace(0, t[-1], len(fgi_values))
            ax2.plot(fgi_t, fgi_values, color='purple', linestyle='--', 
                    label='Fear & Greed Index', alpha=0.7, linewidth=1.5)
            ax2.set_ylabel('Fear & Greed Index', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper right')
        
        ax1.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            self.logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_epidemic_curve(
        self,
        results: pd.DataFrame,
        title: str = "Epidemic Curve",
        show_r0_regions: bool = True,
        r0_value: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot the epidemic curve (infections over time).
        
        Args:
            results: DataFrame with SEIR data
            title: Plot title
            show_r0_regions: Whether to show R₀>1 and R₀<1 regions
            r0_value: Basic reproduction number
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t = np.asarray(results['t'].values)
        I = np.asarray(results['I_frac'].values if 'I_frac' in results.columns else results['I'].values)
        
        # Plot infected curve
        ax.fill_between(t, 0, I, color=STATE_COLORS['I'], alpha=0.3)
        ax.plot(t, I, color=STATE_COLORS['I'], linewidth=2.5, label='Infected')
        
        # Mark peak
        peak_idx = int(np.argmax(I))
        peak_t, peak_I = float(t[peak_idx]), float(I[peak_idx])
        ax.scatter([peak_t], [peak_I], color='darkred', s=100, zorder=5)
        ax.annotate(
            f'Peak: {peak_I:.3f}\nt={peak_t:.0f}',
            xy=(peak_t, peak_I),
            xytext=(peak_t + len(t)*0.1, peak_I),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray')
        )
        
        # R₀ annotation
        if r0_value is not None:
            ax.text(
                0.02, 0.98, f'R₀ = {r0_value:.2f}',
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Infected Fraction')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_network_states(
        self,
        G: nx.Graph,
        node_states: Dict[int, State],
        title: str = "Network FOMO State",
        pos: Optional[Dict] = None,
        node_size_attr: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Visualize network with nodes colored by SEIR state.
        
        Args:
            G: NetworkX graph
            node_states: Mapping of node -> State
            title: Plot title
            pos: Node positions (computed if None)
            node_size_attr: Node attribute to use for sizing
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Compute layout if not provided
        if pos is None:
            self.logger.info("Computing network layout...")
            if G.number_of_nodes() > 1000:
                # Use faster layout for large graphs
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
            else:
                pos = nx.spring_layout(G, seed=42)
        
        # Node colors based on state
        node_colors = []
        for node in G.nodes():
            state = node_states.get(node, State.SUSCEPTIBLE)
            node_colors.append(STATE_COLORS.get(state.value, '#999999'))
        
        # Node sizes
        if node_size_attr and node_size_attr in nx.get_node_attributes(G, node_size_attr):
            sizes = [G.nodes[n].get(node_size_attr, 20) * 50 for n in G.nodes()]
        else:
            degree_view = G.degree()  # type: ignore[operator]
            degrees = dict(degree_view)
            max_deg = max(degrees.values()) if degrees else 1
            sizes = [20 + 100 * degrees[n] / max_deg for n in G.nodes()]
        
        # Draw network
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=sizes, alpha=0.8, ax=ax
        )
        
        # Legend
        legend_patches = [
            mpatches.Patch(color=STATE_COLORS['S'], label='Susceptible'),
            mpatches.Patch(color=STATE_COLORS['E'], label='Exposed'),
            mpatches.Patch(color=STATE_COLORS['I'], label='Infected'),
            mpatches.Patch(color=STATE_COLORS['R'], label='Recovered'),
        ]
        ax.legend(handles=legend_patches, loc='upper left')
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_fgi_correlation(
        self,
        fgi_values: np.ndarray,
        infection_rate: np.ndarray,
        correlation: float,
        p_value: float,
        title: str = "Fear & Greed Index vs Infection Rate",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot correlation between FGI and infection rate.
        
        Args:
            fgi_values: Fear & Greed Index values
            infection_rate: Corresponding infection rates
            correlation: Correlation coefficient
            p_value: P-value of correlation
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(
            fgi_values, infection_rate,
            c=np.arange(len(fgi_values)),  # Color by time
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        
        # Trend line
        z = np.polyfit(fgi_values, infection_rate, 1)
        p = np.poly1d(z)
        fgi_sorted = np.sort(fgi_values)
        ax.plot(fgi_sorted, p(fgi_sorted), 'r--', linewidth=2, label='Trend')
        
        # Colorbar for time
        cbar = plt.colorbar(scatter, ax=ax, label='Time')
        
        # Annotations
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        ax.text(
            0.05, 0.95,
            f'ρ = {correlation:.3f}{significance}\np = {p_value:.4f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.set_xlabel('Fear & Greed Index')
        ax.set_ylabel('Infection Rate (ΔI/Δt)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # FGI interpretation zones
        ax.axvspan(0, 25, alpha=0.1, color='red', label='Extreme Fear')
        ax.axvspan(25, 45, alpha=0.1, color='orange', label='Fear')
        ax.axvspan(55, 75, alpha=0.1, color='lightgreen', label='Greed')
        ax.axvspan(75, 100, alpha=0.1, color='green', label='Extreme Greed')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_parameter_sensitivity(
        self,
        sensitivity_df: pd.DataFrame,
        title: str = "Parameter Sensitivity Analysis",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot parameter sensitivity tornado chart.
        
        Args:
            sensitivity_df: DataFrame with sensitivity analysis results
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = np.asarray(sensitivity_df['parameter'].values)
        elasticity = np.asarray(sensitivity_df['elasticity'].values)
        
        # Sort by absolute elasticity
        sort_idx = np.argsort(np.abs(elasticity))[::-1]
        params = params[sort_idx]
        elasticity = elasticity[sort_idx]
        
        # Colors based on sign
        colors = ['#E94F37' if e > 0 else '#2E86AB' for e in elasticity]
        
        y_pos = np.arange(len(params))
        bars = ax.barh(y_pos, elasticity, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'β' if p == 'beta' else f'σ' if p == 'sigma' else f'γ' for p in params])
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        ax.set_xlabel('Elasticity')
        ax.set_title(title)
        
        # Legend
        red_patch = mpatches.Patch(color='#E94F37', label='Positive effect')
        blue_patch = mpatches.Patch(color='#2E86AB', label='Negative effect')
        ax.legend(handles=[red_patch, blue_patch])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_hypothesis_results(
        self,
        results: Dict[str, HypothesisResult],
        title: str = "Hypothesis Test Results",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot summary of hypothesis test results.
        
        Args:
            results: Dict of hypothesis results
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: P-values with significance threshold
        ax1 = axes[0]
        hypotheses = sorted(results.keys())
        p_values = [results[h].p_value for h in hypotheses]
        colors = ['#7FB069' if p < 0.05 else '#E94F37' for p in p_values]
        
        bars = ax1.bar(hypotheses, p_values, color=colors, alpha=0.8)
        ax1.axhline(y=0.05, color='black', linestyle='--', label='α = 0.05')
        ax1.set_ylabel('P-value')
        ax1.set_title('P-values by Hypothesis')
        ax1.legend()
        
        # Add significance stars
        for i, (h, p) in enumerate(zip(hypotheses, p_values)):
            if p < 0.001:
                star = '***'
            elif p < 0.01:
                star = '**'
            elif p < 0.05:
                star = '*'
            else:
                star = ''
            ax1.text(i, p + 0.02, star, ha='center', fontsize=14)
        
        # Right: Effect sizes
        ax2 = axes[1]
        effect_sizes = [results[h].effect_size for h in hypotheses]
        ax2.bar(hypotheses, effect_sizes, color='steelblue', alpha=0.8)
        ax2.set_ylabel('Effect Size')
        ax2.set_title('Effect Sizes by Hypothesis')
        ax2.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Small')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
        ax2.axhline(y=0.8, color='gray', linestyle='-', alpha=0.5, label='Large')
        ax2.legend(title='Cohen\'s d')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def plot_community_infection_heatmap(
        self,
        communities: Dict[int, List[int]],
        infection_times: Dict[int, float],
        title: str = "Infection Timing by Community",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot heatmap of infection timing by community.
        
        Args:
            communities: Community membership dict
            infection_times: Node -> infection time
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Compute mean infection time per community
        comm_data = []
        for comm_id, nodes in sorted(communities.items()):
            times = [infection_times.get(n, np.nan) for n in nodes]
            times = [t for t in times if not np.isnan(t)]
            if times:
                comm_data.append({
                    'community': comm_id,
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'size': len(nodes),
                    'infected_frac': len(times) / len(nodes)
                })
        
        if not comm_data:
            self.logger.warning("No infection data available for communities")
            return fig
        
        df = pd.DataFrame(comm_data)
        df = df.sort_values('mean_time')
        
        # Bar plot
        cmap = plt.get_cmap('RdYlGn_r')
        colors = cmap(df['mean_time'] / df['mean_time'].max())
        bars = ax.barh(
            range(len(df)), df['mean_time'], 
            xerr=df['std_time'], color=colors, alpha=0.8
        )
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f"C{c}" for c in df['community']])
        ax.set_xlabel('Mean Infection Time')
        ax.set_ylabel('Community')
        ax.set_title(title)
        
        # Annotate with community size
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(
                row['mean_time'] + row['std_time'] + 1, i,
                f"n={row['size']:.0f}",
                va='center', fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def create_summary_dashboard(
        self,
        seir_results: pd.DataFrame,
        G: nx.Graph,
        node_states: Dict[int, State],
        hypothesis_results: Dict[str, HypothesisResult],
        estimated_params: EstimationResult,
        fgi_values: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a comprehensive dashboard with all key visualizations.
        
        Args:
            seir_results: SEIR simulation results
            G: Network graph
            node_states: Current node states
            hypothesis_results: Hypothesis test results
            estimated_params: Estimated parameters
            fgi_values: Fear & Greed Index values
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. SEIR Trajectory (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        t = np.asarray(seir_results['t'].values)
        ax1.plot(t, np.asarray(seir_results['S_frac'].values), color=STATE_COLORS['S'], label='S', linewidth=2)
        ax1.plot(t, np.asarray(seir_results['E_frac'].values), color=STATE_COLORS['E'], label='E', linewidth=2)
        ax1.plot(t, np.asarray(seir_results['I_frac'].values), color=STATE_COLORS['I'], label='I', linewidth=2)
        ax1.plot(t, np.asarray(seir_results['R_frac'].values), color=STATE_COLORS['R'], label='R', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Population Fraction')
        ax1.set_title('SEIR Dynamics')
        ax1.legend(loc='center right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Parameter estimates (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        params = ['β', 'σ', 'γ']
        values = [estimated_params.beta, estimated_params.sigma, estimated_params.gamma]
        cis = [estimated_params.beta_ci, estimated_params.sigma_ci, estimated_params.gamma_ci]
        
        ax2.barh(params, values, color='steelblue', alpha=0.8)
        for i, (val, ci) in enumerate(zip(values, cis)):
            ax2.errorbar(val, i, xerr=[[val - ci[0]], [ci[1] - val]], 
                        color='black', capsize=5)
        ax2.set_xlabel('Value')
        ax2.set_title(f'Estimated Parameters\nR₀ = {estimated_params.r0():.2f}')
        
        # 3. Hypothesis results (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        hypotheses = sorted(hypothesis_results.keys())
        p_values = [hypothesis_results[h].p_value for h in hypotheses]
        colors = ['#7FB069' if p < 0.05 else '#E94F37' for p in p_values]
        ax3.bar(hypotheses, p_values, color=colors, alpha=0.8)
        ax3.axhline(y=0.05, color='black', linestyle='--')
        ax3.set_ylabel('P-value')
        ax3.set_title('Hypothesis Test P-values')
        
        # 4. FGI correlation (middle center)
        if fgi_values is not None:
            ax4 = fig.add_subplot(gs[1, 1])
            I_values = np.asarray(seir_results['I_frac'].values)
            min_len = min(len(fgi_values), len(I_values) - 1)
            delta_I = np.diff(I_values)[:min_len]
            fgi_aligned = fgi_values[:min_len]
            ax4.scatter(fgi_aligned, delta_I, alpha=0.5, c=range(len(fgi_aligned)), cmap='viridis')
            ax4.set_xlabel('Fear & Greed Index')
            ax4.set_ylabel('ΔInfected')
            ax4.set_title('FGI vs Infection Rate')
            ax4.grid(True, alpha=0.3)
        
        # 5. Network degree distribution (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        degrees = [d for _, d in G.degree()]  # type: ignore[misc]
        ax5.hist(degrees, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Degree')
        ax5.set_ylabel('Frequency')
        ax5.set_title(f'Degree Distribution\n<k>={np.mean(degrees):.1f}')
        ax5.set_yscale('log')
        
        # 6. State distribution over time (bottom row spans 2)
        ax6 = fig.add_subplot(gs[2, :2])
        ax6.stackplot(
            t,
            np.asarray(seir_results['S_frac'].values),
            np.asarray(seir_results['E_frac'].values),
            np.asarray(seir_results['I_frac'].values),
            np.asarray(seir_results['R_frac'].values),
            colors=[STATE_COLORS['S'], STATE_COLORS['E'], STATE_COLORS['I'], STATE_COLORS['R']],
            labels=['S', 'E', 'I', 'R'],
            alpha=0.8
        )
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Population Fraction')
        ax6.set_title('State Distribution Over Time')
        ax6.legend(loc='center right')
        
        # 7. Summary statistics (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        summary_text = [
            "SUMMARY STATISTICS",
            "-" * 30,
            f"Network nodes: {G.number_of_nodes():,}",
            f"Network edges: {G.number_of_edges():,}",
            f"Mean degree: {np.mean(degrees):.2f}",
            "",
            f"Basic R₀: {estimated_params.r0():.3f}",
            f"β (transmission): {estimated_params.beta:.4f}",
            f"σ (incubation): {estimated_params.sigma:.4f}",
            f"γ (recovery): {estimated_params.gamma:.4f}",
            "",
            f"Peak infected: {seir_results['I_frac'].max():.3f}",
            f"Peak time: {seir_results['I_frac'].idxmax()}",
            "",
            "Hypothesis Support:",
        ]
        
        for h in sorted(hypothesis_results.keys()):
            r = hypothesis_results[h]
            status = "✓" if r.reject_null else "✗"
            summary_text.append(f"  {h}: {status} (p={r.p_value:.3f})")
        
        ax7.text(0.1, 0.95, '\n'.join(summary_text), transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        fig.suptitle('Crypto FOMO Contagion Analysis Dashboard', fontsize=18, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path)
            self.logger.info(f"Saved dashboard to {save_path}")
        
        return fig

    def plot_r0_comparison_by_period(
        self,
        period_results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create bar chart comparing R₀ across market periods.
        
        Visualizes the three-period research design: Training (2017 bull),
        Control (2018 bear), and Validation (2020-21 bull) with confidence
        intervals and epidemic threshold reference line.
        
        Args:
            period_results: Dict with period names as keys, each containing:
                - 'r0': Estimated R₀ value
                - 'r0_ci': Tuple of (lower, upper) 95% CI bounds
                - 'market_type': 'bull' or 'bear'
                - 'description': Period description for labels
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        periods = list(period_results.keys())
        r0_values = [period_results[p]['r0'] for p in periods]
        
        # Calculate error bars from confidence intervals
        errors = []
        for p in periods:
            ci = period_results[p].get('r0_ci', (r0_values[periods.index(p)], r0_values[periods.index(p)]))
            r0 = period_results[p]['r0']
            errors.append([r0 - ci[0], ci[1] - r0])
        errors = np.array(errors).T
        
        # Color by market type
        colors = []
        for p in periods:
            market_type = period_results[p].get('market_type', 'bull')
            if market_type == 'bull':
                colors.append('#E94F37')  # Red for bull markets
            else:
                colors.append('#2E86AB')  # Blue for bear markets
        
        # Create bar chart
        x_pos = np.arange(len(periods))
        bars = ax.bar(x_pos, r0_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add error bars
        ax.errorbar(x_pos, r0_values, yerr=errors, fmt='none', color='black', 
                    capsize=8, capthick=2, linewidth=2)
        
        # Add epidemic threshold line
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Epidemic threshold (R₀=1)')
        
        # Add value labels on bars
        for i, (bar, r0) in enumerate(zip(bars, r0_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[1][i] + 0.1,
                    f'{r0:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Create x-axis labels with descriptions
        labels = []
        for p in periods:
            desc = period_results[p].get('description', p.capitalize())
            market = period_results[p].get('market_type', '').upper()
            labels.append(f"{desc}\n({market})")
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Basic Reproduction Number (R₀)', fontsize=14)
        ax.set_title('FOMO Contagion R₀ Across Market Conditions', fontsize=16, fontweight='bold')
        
        # Add legend for market types
        bull_patch = mpatches.Patch(color='#E94F37', alpha=0.8, label='Bull Market')
        bear_patch = mpatches.Patch(color='#2E86AB', alpha=0.8, label='Bear Market')
        ax.legend(handles=[bull_patch, bear_patch, ax.get_lines()[0]], loc='upper right', fontsize=11)
        
        ax.set_ylim(0, max(r0_values) * 1.3)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            self.logger.info(f"Saved R₀ comparison plot to {save_path}")
        
        return fig


def main():
    """Test visualization module."""
    import networkx as nx
    
    print("Testing visualization module...")
    
    # Create test data
    t = np.arange(100)
    results = pd.DataFrame({
        't': t,
        'S_frac': 0.9 * np.exp(-0.03 * t),
        'E_frac': 0.1 * (1 - np.exp(-0.05 * t)) * np.exp(-0.02 * t),
        'I_frac': 0.2 * (1 - np.exp(-0.03 * t)) * np.exp(-0.01 * t),
        'R_frac': 0.0,
    })
    results['R_frac'] = 1 - results['S_frac'] - results['E_frac'] - results['I_frac']
    
    # Initialize visualizer
    viz = SEIRVisualizer(output_dir="figures")
    
    # Test SEIR trajectory plot
    fig = viz.plot_seir_trajectory(results, title="Test SEIR Dynamics")
    plt.close(fig)
    print("✓ SEIR trajectory plot created")
    
    # Test epidemic curve
    fig = viz.plot_epidemic_curve(results, r0_value=3.0)
    plt.close(fig)
    print("✓ Epidemic curve created")
    
    print("\nVisualization module tests passed!")


if __name__ == "__main__":
    main()
