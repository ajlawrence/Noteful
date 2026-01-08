# Costa del Sol Economic Monitor

A modular Python pipeline for tracking the economy of Costa del Sol, Spain (Málaga province, code ES617).

## 🎯 Overview

This system collects, processes, and analyzes economic data from multiple public sources:

- **Tourism**: Arrivals, overnight stays, expenditure (Dataestur, INE)
- **Employment**: Rates, unemployment (Eurostat)
- **GDP**: Regional contributions, growth (Eurostat)
- **Real Estate**: Housing price indices (INE)
- **News**: Economic events from Spanish news sources (RSS)

### Key Features

- ✅ **Modular architecture** - Easy to extend with new data sources
- ✅ **AI-powered insights** - Anomaly detection, trend analysis, report generation (Claude)
- ✅ **Automated scheduling** - Cron-ready, adaptive update frequencies
- ✅ **Data validation** - Quality checks before storage
- ✅ **Visualizations** - Matplotlib charts and dashboards
- ✅ **Email alerts** - Notifications for anomalies and failures
- ✅ **SQLite storage** - Simple, portable database

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### 2. Installation

```bash
# Clone or navigate to the repository
cd costa_sol_economy

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use any text editor
```

Required settings in `.env`:

```env
# For AI features (optional but recommended)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx

# For email alerts (optional)
SMTP_USERNAME=your.email@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx  # Gmail App Password
ALERT_RECIPIENT=your.email@gmail.com
```

### 4. Run the Pipeline

```bash
# Test connections first
python main.py --test

# Run the full pipeline
python main.py --mode full

# Or run specific schedules
python main.py --mode daily    # News only
python main.py --mode monthly  # Tourism, real estate
python main.py --mode quarterly # Employment, GDP
```

## 📁 Project Structure

```
costa_sol_economy/
├── main.py                 # Main entry point & orchestrator
├── config.yaml             # Data source configurations
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # This file
│
├── fetchers/               # Data fetching modules
│   ├── dataestur.py        # Spanish tourism data
│   ├── eurostat.py         # EU statistics
│   ├── ine.py              # Spanish national statistics
│   └── news.py             # RSS news feeds
│
├── processors/             # Data processing modules
│   ├── cleaner.py          # Data cleaning
│   ├── validator.py        # Quality validation
│   └── trends.py           # Trend calculations
│
├── storage/                # Database layer
│   ├── models.py           # Table definitions
│   └── database.py         # Database operations
│
├── ai/                     # AI/LLM features
│   ├── claude_client.py    # Anthropic API wrapper
│   ├── insights.py         # Anomaly detection
│   ├── summarizer.py       # News summarization
│   ├── quality.py          # AI data validation
│   └── reports.py          # Report generation
│
├── outputs/                # Output generation
│   ├── charts.py           # Matplotlib visualizations
│   └── alerts.py           # Email notifications
│
└── data/                   # Generated files (gitignored)
    ├── costa_econ.db       # SQLite database
    ├── charts/             # Generated PNG charts
    └── pipeline.log        # Log file
```

## 📊 Data Sources

| Source | Type | Frequency | Data |
|--------|------|-----------|------|
| Dataestur | API | Monthly | Tourist arrivals, expenditure |
| Eurostat | REST API | Quarterly | GDP, employment rates |
| INE | API/CSV | Monthly/Quarterly | Hotel occupancy, housing prices |
| RSS Feeds | Web | Daily | Economic news |

### Region Codes

- **ES617**: Málaga (NUTS3)
- **ES61**: Andalucía (NUTS2)

## 🤖 AI Features

With an Anthropic API key, you get:

1. **Anomaly Detection**: Identifies unusual data points and explains possible causes
2. **News Summarization**: Summarizes economic news with relevance scoring
3. **Data Quality Checks**: AI-powered validation of incoming data
4. **Report Generation**: Automated monthly economic briefings
5. **Trend Analysis**: Plain-language explanations of trends

### Cost Management

The system uses smart model selection:
- **claude-3-haiku**: Quick tasks (summaries, quality checks)
- **claude-sonnet-4-20250514**: Complex analysis (anomalies, reports)

Response caching reduces repeat API calls.

## ⚙️ Command Line Usage

```bash
# Full help
python main.py --help

# Run modes
python main.py --mode auto       # Determine what needs updating
python main.py --mode daily      # Daily tasks (news)
python main.py --mode monthly    # Monthly tasks (tourism)
python main.py --mode quarterly  # Quarterly tasks (GDP, employment)
python main.py --mode full       # Everything

# Other commands
python main.py --test            # Test API connections
python main.py --summary         # Show database summary
python main.py --report          # Generate report only

# Logging options
python main.py --log-level DEBUG
python main.py --log-file data/custom.log
```

## 📅 Scheduling (Automation)

### Option 1: Cron (Linux/Mac)

```bash
# Edit crontab
crontab -e

# Add these lines:
# Daily at 8:00 AM
0 8 * * * cd /path/to/costa_sol_economy && python main.py --mode daily

# Monthly on the 5th
0 9 5 * * cd /path/to/costa_sol_economy && python main.py --mode monthly

# Quarterly
0 9 15 1,4,7,10 * cd /path/to/costa_sol_economy && python main.py --mode quarterly
```

### Option 2: GitHub Actions

Create `.github/workflows/economy-pipeline.yml`:

```yaml
name: Costa Sol Economy Pipeline

on:
  schedule:
    - cron: '0 8 * * *'    # Daily at 8 AM UTC
    - cron: '0 9 5 * *'    # Monthly on 5th
  workflow_dispatch:        # Manual trigger

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python main.py --mode auto
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## 📧 Email Alerts Setup

1. Enable 2-Factor Authentication on your Google account
2. Go to https://myaccount.google.com/apppasswords
3. Generate an App Password for "Mail"
4. Add to your `.env`:

```env
SMTP_USERNAME=your.email@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx
ALERT_RECIPIENT=your.email@gmail.com
```

## 📈 Extending the System

### Adding a New Data Source

1. Create a new file in `fetchers/`:

```python
# fetchers/my_source.py
def fetch_my_data():
    # Fetch data
    df = pd.DataFrame(...)
    return df
```

2. Add to `main.py`:

```python
from fetchers.my_source import fetch_my_data

# In EconomyPipeline._fetch_xyz_data():
df = fetch_my_data()
self.db.save_xyz_data(df, source="my_source")
```

3. Add configuration in `config.yaml`

### Adding a New Visualization

```python
# In your code or extend outputs/charts.py
from outputs.charts import ChartGenerator

charts = ChartGenerator()
charts.time_series_chart(
    my_df,
    title="My Custom Chart",
    filename="custom_chart.png"
)
```

## 🔧 Troubleshooting

### APIs returning empty data

- Check your internet connection
- Run `python main.py --test` to verify connections
- Some APIs may be temporarily unavailable
- Check for API rate limiting

### AI features not working

- Verify `ANTHROPIC_API_KEY` is set in `.env`
- Check API key is valid: `python -c "from ai import ClaudeClient; c = ClaudeClient(); print(c.health_check())"`

### Email alerts not sending

- Verify Gmail App Password (not your regular password)
- Check 2FA is enabled on your Google account
- Run: `python -c "from outputs.alerts import EmailAlerts; e = EmailAlerts(); print(e.test_connection())"`

## 📝 License

This project is for educational and research purposes. Data from public sources should be used according to their respective terms of service.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Costa del Sol Economic Monitor** - Built with Python, powered by open data and AI.
