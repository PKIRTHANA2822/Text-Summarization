# Project Title
- Daily Traffic Volume Forecasting using Facebook Prophet
<img width="500" height="300" alt="image_96423651921642102015817" src="https://github.com/user-attachments/assets/9255b571-e9fe-486d-9b27-42bbee663e15" />
# Objective
- Forecast daily traffic counts for the next 60–200 days using historical timestamped traffic data, in order to support capacity planning, staffing, signal timing, maintenance scheduling, and incident preparedness.
- Primary goals
- Build a reliable time‑series model (Prophet) to predict daily totals.
- Quantify forecast uncertainty with prediction intervals.
- Evaluate accuracy on a held‑out test window (last 60 days).
- Package plots/tables for stakeholders.
# Why We Use This Project (Business Value)
- Operations: Optimize lane/booth staffing and shift rosters.
- Traffic Management: Improve signal timing, ramp metering, and diversion plans.
- Maintenance: Schedule road works during lower‑demand windows.
- Safety & Incident Response: Anticipate peak loads to stage resources.
- Planning & Budgeting: Demand projections for infrastructure upgrades.
- Prophet is chosen because it handles trend + multiple seasonalities, is robust to missing data/outliers, and provides interpretable components (trend/weekly/yearly effects) with credible intervals.
# Data Overview
- Source file: Traffic data.csv
- Columns (as used in the notebook):
- ID (identifier — dropped for modeling)
- Datetime (string; format %d-%m-%Y %H:%M, converted to pandas datetime)
- Count (traffic volume)
- Aggregation: Resampled to daily sums.
- Prophet schema:
- ds → datestamp (daily index)
- y → target (daily total Count)
- Assumptions/Notes
- Timezone and daylight‑saving treated implicitly; aggregation to daily reduces DST issues.
- Missing values should be imputed (forward‑fill) or excluded after aggregation if needed.
- Outliers (abnormal spikes/drops) can be capped or flagged for Prophet’s outlier robustness.
# Step‑by‑Step Approach
- Import & Setup: Install/import prophet, pandas, numpy, matplotlib.
- Load Data: pd.read_csv('Traffic data.csv').
- Parse Datetime: pd.to_datetime(..., format='%d-%m-%Y %H:%M').
- Initial EDA: Head, info, null checks; quick time‑series line plot.
- Aggregate: Set index to Datetime; create y=Count; resample to daily sums; add ds=index.
- Train/Test Split: Last 60 days as test; rest as train (no shuffle).
- Model Spec: Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9).
- Fit & Forecast (Test Window): Fit on train; make_future_dataframe(periods=60); predict; plot components.
- Evaluate: Compare test y vs. forecast yhat on the last 60 days; compute MAE/RMSE/MAPE; visualize actual vs. forecast with bands.
- Refit on Full Data: Retrain on all days; forecast 200‑day horizon for deployment.
- Package Outputs: Save plots, metrics table, and forecast CSV for stakeholders.
# Exploratory Data Analysis (EDA)
- Questions to answer
- Trend: Is traffic growing, flat, or declining over time?
- Seasonality: Weekly cycle (weekday vs. weekend)? Yearly effects (holidays)?
- Holidays/Events: Recurrent dips/spikes on festivals/public holidays?
- Data Quality: Missing timestamps after resampling? Outliers?
- Recommended EDA visuals
- Line plot of daily Count (after resample)
- Boxplots by day of week and month
- Rolling mean/STD (7‑day, 30‑day)
- Autocorrelation (ACF) to see persistence/weekly cycles
### Example snippets (optional)
# Daily series
df_daily = df.copy()  # after resampling
ax = df_daily['y'].plot(figsize=(10,4), title='Daily Traffic Volume')
ax.set(xlabel='', ylabel='Count')
# Add day-of-week / month for grouping
df_daily = df_daily.assign(
    dow=lambda d: d['ds'].dt.day_name(),
    month=lambda d: d['ds'].dt.month_name()
)

# Weekly boxplot
df_daily.boxplot(column='y', by='dow', rot=45, figsize=(10,4))
plt.suptitle(''); plt.title('Traffic by Day of Week'); plt.ylabel('Count')
# Feature Selection
- With Prophet, the core required features are:
- ds (date)
- y (target)
- Optional regressors can be added if available and causally meaningful:
- Calendars: public holidays/festivals, school terms
- Weather: rain, temperature, visibility
- Events: roadworks, strikes, sports matches, concerts
- Only include regressors that precede the target in time (no leakage) and are known for future dates (or can be forecasted themselves).
# Feature Engineering
- Resampling: Sum to daily (resample('D').sum())
- Missing Handling: Forward‑fill/linear interpolate after aggregation
- Outlier Treatment: Winsorize/cap extreme values; or let Prophet’s intervals absorb but still review
- Calendar Effects (recommended):
- Construct a holidays DataFrame and pass via Prophet(holidays=holidays_df)
- Add weekly_seasonality=True (Prophet has this by default); optionally daily_seasonality=False for daily totals
- Exogenous Regressors: model.add_regressor('rain'), etc., after aligning features to ds
- Holiday example
- from prophet.make_holidays import make_holidays_df
- holidays_df = make_holidays_df(year_list=sorted(df['ds'].dt.year.unique()), country='IN')
- model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
9) Model Training
- Baseline: Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
- Fit: model.fit(train)
- Horizon: make_future_dataframe(periods=60, freq='D') for test; later periods=200 for deployment
- Components: model.plot_components(forecast) to inspect trend/seasonality/holiday effects
- Cross‑validation (recommended)
- from prophet.diagnostics import cross_validation, performance_metrics
- cv = cross_validation(model, initial='365 days', period='30 days', horizon='60 days')
- perf = performance_metrics(cv, rolling_window=0.95)
- Tune seasonality_prior_scale, changepoint_prior_scale, and seasonality modes (additive vs. multiplicative) based on CV metrics.
# Model Testing & Evaluation
- Evaluate on the last 60 days (held out):
- Metrics
- MAE: mean absolute error
- RMSE: root mean squared error
- MAPE: mean absolute percentage error (handle zeros carefully)
- Coverage: % of actuals within [yhat_lower, yhat_upper]
- Example evaluation code
- import numpy as np
- from sklearn.metrics import mean_absolute_error, mean_squared_error
# Align test set and predictions (last 60 days)
- pred_60 = forecast.tail(60).set_index('ds')
- actual_60 = test.set_index('ds')
# Drop any dates that might be missing on either side
- aligned = actual_60.join(pred_60[['yhat','yhat_lower','yhat_upper']], how='inner')
- mae = mean_absolute_error(aligned['y'], aligned['yhat'])
- rmse = mean_squared_error(aligned['y'], aligned['yhat'], squared=False)
- mape = (np.abs(aligned['y'] - aligned['yhat']) / np.clip(aligned['y'], 1e-6, None)).mean() * 100
- coverage = ((aligned['y'] >= aligned['yhat_lower']) & (aligned['y'] <= aligned['yhat_upper'])).mean() * 100
- print({'MAE': mae, 'RMSE': rmse, 'MAPE_%': mape, 'PI_Coverage_%': coverage})
- Diagnostics
- Residual plots (actual − forecast) for autocorrelation
- Inspect component plots for reasonable weekly/yearly shapes
- Check for systematic under/over‑prediction on weekends/holidays
# Output & Deliverables
<img width="603" height="91" alt="Screenshot 2025-08-14 193908" src="https://github.com/user-attachments/assets/24da8827-a6d3-4cc0-a21b-3119c9524d3c" />

