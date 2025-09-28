name: Fetch Stock Data and Update CSV

on:
  schedule:
    # Run daily at 6 PM IST (12:30 PM UTC) on Monday to Friday
    - cron: '30 12 * * 1-5'
  workflow_dispatch:  # Manual trigger

jobs:
  fetch-data:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run data fetch script
        run: python fetch_stock_data.py

      - name: Commit and push CSV
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add stock_data.csv
          git diff --staged --quiet || git commit -m "Update stock_data.csv - $(date)"
          git push
