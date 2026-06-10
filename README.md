# FinSight
ML-Powered Finance Dashboard
An interactive income, expense & savings analytics platform using Python Dash. Implements unsupervised learning (K-Means clustering) for user segmentation, supervised regression for 3-month savings forecasting, and statistical anomaly detection. Visualises 1,200 records across 100 users with 10 live-updating Plotly charts.

🚀 Getting Started
1. Clone the repository
bashgit clone https://github.com/your-username/finsight-analytics.git
cd finsight-analytics
2. Install dependencies
bashpip install dash dash-bootstrap-components plotly pandas numpy scikit-learn scipy
3. Add your dataset
Place your CSV file in the project folder and update the path in dashboard.py:
pythonCSV = r"C:\path\to\your\incomedatasetnew.csv"
4. Run the dashboard
bashpython dashboard.py
5. Open in your browser
http://127.0.0.1:8050

⚠️ Keep the terminal open while using the dashboard — closing it stops the server.


🎛 Using the Dashboard

City / Occupation / Segment dropdowns — filter all charts simultaneously
Month Range slider — scrub across Jan–Dec 2025
Hover over any chart — see detailed values per data point
Scatter plot — hover over individual users to see name, city, income & savings


📁 Project Structure
finsight-analytics/
├── dashboard.py              # Main app
├── incomedatasetnew.csv      # Dataset
