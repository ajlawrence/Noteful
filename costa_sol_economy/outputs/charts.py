# =============================================================================
# Chart Generator
# =============================================================================
# Creates visualizations of economic data using Matplotlib.
#
# Chart types:
#   - Time series line charts
#   - Year-over-year comparison charts
#   - Bar charts for categorical data
#   - Dashboard-style multi-panel charts
# =============================================================================

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

# Matplotlib imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not installed. Install with: pip install matplotlib")

# Optional seaborn for prettier charts
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# CHART CONFIGURATION
# =============================================================================

# Default styling
DEFAULT_STYLE = {
    "figure_size": (10, 6),
    "dpi": 150,
    "line_width": 2,
    "title_size": 14,
    "label_size": 12,
    "colors": {
        "primary": "#2563EB",    # Blue
        "secondary": "#10B981",  # Green
        "warning": "#F59E0B",    # Orange
        "danger": "#EF4444",     # Red
        "neutral": "#6B7280",    # Gray
    },
    "grid_alpha": 0.3,
}

# Costa del Sol branding colors
COSTA_SOL_COLORS = ["#0066CC", "#FF6600", "#009933", "#CC0066", "#9933FF"]


# =============================================================================
# CHART GENERATOR CLASS
# =============================================================================

class ChartGenerator:
    """
    Generate charts and visualizations for economic data.

    Example usage:
        generator = ChartGenerator(output_dir="data/charts")

        # Line chart of tourism data
        generator.time_series_chart(
            tourism_df,
            title="Tourist Arrivals - Costa del Sol",
            filename="tourism_arrivals.png"
        )

        # YoY comparison
        generator.yoy_comparison_chart(
            df, title="YoY Change"
        )

        # Dashboard
        generator.create_dashboard(
            tourism_df, employment_df, gdp_df
        )
    """

    def __init__(
        self,
        output_dir: str = "data/charts",
        style: Optional[Dict] = None
    ):
        """
        Initialize the Chart Generator.

        Args:
            output_dir: Directory to save charts
            style: Custom style dictionary (overrides defaults)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for charts. "
                "Install with: pip install matplotlib"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.style = {**DEFAULT_STYLE, **(style or {})}

        # Set up matplotlib style
        if SEABORN_AVAILABLE:
            sns.set_theme(style="whitegrid")
        else:
            plt.style.use("seaborn-v0_8-whitegrid")

        logger.info(f"ChartGenerator initialized. Output dir: {output_dir}")

    # =========================================================================
    # TIME SERIES CHARTS
    # =========================================================================

    def time_series_chart(
        self,
        df: pd.DataFrame,
        title: str = "Time Series",
        date_column: str = "date",
        value_column: str = "value",
        filename: Optional[str] = None,
        show_trend: bool = True,
        highlight_anomalies: bool = False,
        y_label: Optional[str] = None
    ) -> str:
        """
        Create a time series line chart.

        Args:
            df: DataFrame with date and value columns
            title: Chart title
            date_column: Name of date column
            value_column: Name of value column
            filename: Output filename (auto-generated if not provided)
            show_trend: Show trend line (moving average)
            highlight_anomalies: Highlight statistical anomalies
            y_label: Y-axis label

        Returns:
            Path to saved chart file

        Example:
            path = generator.time_series_chart(
                tourism_df,
                title="Monthly Tourist Arrivals",
                y_label="Arrivals (thousands)"
            )
        """
        if df.empty:
            logger.warning("Empty DataFrame, cannot create chart")
            return ""

        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        fig, ax = plt.subplots(figsize=self.style["figure_size"])

        # Main line
        ax.plot(
            df[date_column],
            df[value_column],
            color=self.style["colors"]["primary"],
            linewidth=self.style["line_width"],
            label="Actual"
        )

        # Trend line (3-period moving average)
        if show_trend and len(df) >= 3:
            df["_trend"] = df[value_column].rolling(window=3, center=True).mean()
            ax.plot(
                df[date_column],
                df["_trend"],
                color=self.style["colors"]["secondary"],
                linewidth=1.5,
                linestyle="--",
                label="Trend (3-period MA)"
            )

        # Highlight anomalies
        if highlight_anomalies:
            mean = df[value_column].mean()
            std = df[value_column].std()
            threshold = 2.5

            anomalies = df[abs(df[value_column] - mean) > threshold * std]
            ax.scatter(
                anomalies[date_column],
                anomalies[value_column],
                color=self.style["colors"]["danger"],
                s=100,
                zorder=5,
                label="Anomalies"
            )

        # Formatting
        ax.set_title(title, fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("Date", fontsize=self.style["label_size"])
        ax.set_ylabel(y_label or value_column.title(), fontsize=self.style["label_size"])

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha="right")

        ax.legend(loc="upper left")
        ax.grid(True, alpha=self.style["grid_alpha"])

        plt.tight_layout()

        # Save
        filename = filename or f"timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.style["dpi"], bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Chart saved: {filepath}")
        return str(filepath)

    def yoy_comparison_chart(
        self,
        df: pd.DataFrame,
        title: str = "Year-over-Year Change",
        date_column: str = "date",
        yoy_column: str = "yoy_change",
        filename: Optional[str] = None
    ) -> str:
        """
        Create a bar chart showing YoY percentage changes.

        Positive changes are shown in green, negative in red.

        Args:
            df: DataFrame with YoY data
            title: Chart title
            date_column: Date column name
            yoy_column: YoY change column name
            filename: Output filename

        Returns:
            Path to saved chart
        """
        if df.empty or yoy_column not in df.columns:
            logger.warning("Cannot create YoY chart: missing data")
            return ""

        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.dropna(subset=[yoy_column])
        df = df.sort_values(date_column)

        fig, ax = plt.subplots(figsize=self.style["figure_size"])

        # Color based on positive/negative
        colors = [
            self.style["colors"]["secondary"] if v >= 0 else self.style["colors"]["danger"]
            for v in df[yoy_column]
        ]

        bars = ax.bar(
            df[date_column],
            df[yoy_column],
            color=colors,
            width=20  # Adjust for monthly data
        )

        # Add zero line
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Formatting
        ax.set_title(title, fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("Date", fontsize=self.style["label_size"])
        ax.set_ylabel("YoY Change (%)", fontsize=self.style["label_size"])

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45, ha="right")

        ax.grid(True, alpha=self.style["grid_alpha"], axis="y")

        plt.tight_layout()

        # Save
        filename = filename or f"yoy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.style["dpi"], bbox_inches="tight")
        plt.close(fig)

        logger.info(f"YoY chart saved: {filepath}")
        return str(filepath)

    def multi_series_chart(
        self,
        dataframes: List[pd.DataFrame],
        labels: List[str],
        title: str = "Comparison",
        date_column: str = "date",
        value_column: str = "value",
        filename: Optional[str] = None,
        normalize: bool = False
    ) -> str:
        """
        Create a chart with multiple time series.

        Args:
            dataframes: List of DataFrames to plot
            labels: Labels for each series
            title: Chart title
            date_column: Date column name
            value_column: Value column name
            filename: Output filename
            normalize: Normalize to 100 at start for comparison

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=self.style["figure_size"])

        colors = COSTA_SOL_COLORS

        for i, (df, label) in enumerate(zip(dataframes, labels)):
            if df.empty:
                continue

            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)

            values = df[value_column].values

            if normalize and len(values) > 0:
                first_val = values[0]
                if first_val != 0:
                    values = (values / first_val) * 100

            ax.plot(
                df[date_column],
                values,
                color=colors[i % len(colors)],
                linewidth=self.style["line_width"],
                label=label
            )

        ax.set_title(title, fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("Date", fontsize=self.style["label_size"])
        y_label = "Index (start = 100)" if normalize else value_column.title()
        ax.set_ylabel(y_label, fontsize=self.style["label_size"])

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45, ha="right")

        ax.legend(loc="best")
        ax.grid(True, alpha=self.style["grid_alpha"])

        plt.tight_layout()

        filename = filename or f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.style["dpi"], bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Multi-series chart saved: {filepath}")
        return str(filepath)

    # =========================================================================
    # DASHBOARD
    # =========================================================================

    def create_dashboard(
        self,
        tourism_df: Optional[pd.DataFrame] = None,
        employment_df: Optional[pd.DataFrame] = None,
        gdp_df: Optional[pd.DataFrame] = None,
        title: str = "Costa del Sol Economic Dashboard",
        filename: Optional[str] = None
    ) -> str:
        """
        Create a multi-panel dashboard with key metrics.

        Args:
            tourism_df: Tourism data
            employment_df: Employment data
            gdp_df: GDP data
            title: Dashboard title
            filename: Output filename

        Returns:
            Path to saved dashboard image
        """
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Create grid of subplots
        # 2 rows, 2 columns
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

        # Panel 1: Tourism (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if tourism_df is not None and not tourism_df.empty:
            self._plot_simple_series(ax1, tourism_df, "Tourist Arrivals")
        else:
            ax1.text(0.5, 0.5, "No tourism data", ha="center", va="center")
            ax1.set_title("Tourist Arrivals")

        # Panel 2: Employment (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        if employment_df is not None and not employment_df.empty:
            self._plot_simple_series(ax2, employment_df, "Employment Rate")
        else:
            ax2.text(0.5, 0.5, "No employment data", ha="center", va="center")
            ax2.set_title("Employment")

        # Panel 3: GDP (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        if gdp_df is not None and not gdp_df.empty:
            self._plot_simple_series(ax3, gdp_df, "Regional GDP")
        else:
            ax3.text(0.5, 0.5, "No GDP data", ha="center", va="center")
            ax3.set_title("Regional GDP")

        # Panel 4: Summary metrics (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_summary_panel(ax4, tourism_df, employment_df, gdp_df)

        # Save
        filename = filename or f"dashboard_{datetime.now().strftime('%Y%m%d')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.style["dpi"], bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Dashboard saved: {filepath}")
        return str(filepath)

    def _plot_simple_series(
        self,
        ax,
        df: pd.DataFrame,
        title: str,
        date_col: str = "date",
        value_col: str = "value"
    ):
        """Plot a simple time series on an axis."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        ax.plot(
            df[date_col],
            df[value_col],
            color=self.style["colors"]["primary"],
            linewidth=2
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

    def _plot_summary_panel(
        self,
        ax,
        tourism_df: Optional[pd.DataFrame],
        employment_df: Optional[pd.DataFrame],
        gdp_df: Optional[pd.DataFrame]
    ):
        """Create a text summary panel."""
        ax.axis("off")
        ax.set_title("Key Metrics", fontsize=12, fontweight="bold")

        metrics = []

        if tourism_df is not None and not tourism_df.empty:
            latest = tourism_df["value"].iloc[-1]
            if "yoy_change" in tourism_df.columns:
                yoy = tourism_df["yoy_change"].dropna()
                if len(yoy) > 0:
                    metrics.append(f"Tourism YoY: {yoy.iloc[-1]:+.1f}%")
            metrics.append(f"Latest arrivals: {latest:,.0f}")

        if employment_df is not None and not employment_df.empty:
            latest = employment_df["value"].iloc[-1]
            metrics.append(f"Employment rate: {latest:.1f}%")

        if gdp_df is not None and not gdp_df.empty:
            latest = gdp_df["value"].iloc[-1]
            metrics.append(f"GDP: €{latest:,.0f}M")

        if not metrics:
            metrics = ["No data available"]

        text = "\n\n".join([f"• {m}" for m in metrics])
        ax.text(
            0.1, 0.9, text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace"
        )

        # Add timestamp
        ax.text(
            0.1, 0.1,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            transform=ax.transAxes,
            fontsize=9,
            color="gray"
        )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Chart Generator...")

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=24, freq="M")
    base_values = [100, 95, 110, 140, 180, 200, 220, 210, 170, 130, 105, 100]
    values = base_values + [v * 1.05 for v in base_values]

    test_df = pd.DataFrame({
        "date": dates,
        "value": values,
        "yoy_change": [None] * 12 + [5.0] * 12
    })

    print("\nSample Data:")
    print(test_df.head())

    try:
        generator = ChartGenerator(output_dir="/tmp/test_charts")

        print("\n1. Time Series Chart:")
        path = generator.time_series_chart(
            test_df,
            title="Tourist Arrivals - Costa del Sol",
            y_label="Arrivals (thousands)"
        )
        print(f"   Saved to: {path}")

        print("\n2. YoY Comparison Chart:")
        path = generator.yoy_comparison_chart(
            test_df,
            title="Year-over-Year Change"
        )
        print(f"   Saved to: {path}")

        print("\n3. Dashboard:")
        path = generator.create_dashboard(
            tourism_df=test_df,
            title="Test Dashboard"
        )
        print(f"   Saved to: {path}")

        print("\n✅ All charts created successfully!")

    except ImportError as e:
        print(f"⚠️  {e}")
