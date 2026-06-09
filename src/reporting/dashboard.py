"""Streamlit dashboard for interactive bid review — stub (Phase 2).

Planned features:
  - Portfolio summary: dispatch timeline, reservoir levels, revenue breakdown
  - Bid curve visualisation: 10-step Elspot curves for Day 1
  - Sensitivity sliders: terminal water value, FCR-N cap, horizon
  - Side-by-side comparison: FI zone vs SE2 zone results
  - Export button: trigger CSV / bid-format download

Usage (once implemented):
    streamlit run src/reporting/dashboard.py -- --output-dir data/processed/bids/

Dependencies (not yet in pyproject.toml):
    streamlit >= 1.35
    plotly >= 5.0
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Dashboard not yet implemented. "
        "Run `python main.py` for CLI output."
    )


if __name__ == "__main__":
    main()
