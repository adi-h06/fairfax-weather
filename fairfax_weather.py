import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

weather = pd.read_csv("weather_data.csv")

weather["DATE"] = pd.to_datetime(weather["DATE"])
weather = weather.set_index("DATE")

weather["TAVG"] = (weather["TMAX"] + weather["TMIN"]) / 2

monthly = weather["TAVG"].resample("ME").mean()

trend = monthly.rolling(window=12).mean()
trend_clean = trend.dropna()

years_since_start = (trend_clean.index - trend_clean.index[0]).days / 365.25
result = stats.linregress(years_since_start, trend_clean.values)

print("\nlong term trend")
print("slope (F per decade):", round(result.slope * 10, 3))
print("p-value:", result.pvalue)
print("r-value:", round(result.rvalue, 3))

summer_months = [6, 7, 8]
summer_data = weather[weather.index.month.isin(summer_months)]

summer_yearly = summer_data.groupby(summer_data.index.year)["TAVG"].mean()

this_year = summer_yearly.index.max()
this_summer_avg = summer_yearly.loc[this_year]

historical_summers = summer_yearly[summer_yearly.index != this_year]

print(f"\ncomparing {this_year} summer to {len(historical_summers)} historical summers")
print("this summer average:", round(this_summer_avg, 2))
print("historical summer mean:", round(historical_summers.mean(), 2))
print("historical summer std dev:", round(historical_summers.std(), 2))

t_stat, p_value = stats.ttest_1samp(historical_summers, this_summer_avg)

print("\nt-statistic:", round(t_stat, 3))
print("p-value:", p_value)

if p_value < 0.05:
    print(f"{this_year} summer IS a statistically significant outlier")
else:
    print(f"{this_year} summer is NOT statistically significant compared to history")

print("\nvalidation: forecast using only early data")

cutoff_year = 2015

train_monthly = monthly[monthly.index.year <= cutoff_year]
train_trend = train_monthly.rolling(window=12).mean().dropna()

train_years_since_start = (train_trend.index - train_trend.index[0]).days / 365.25
train_result = stats.linregress(train_years_since_start, train_trend.values)

print(f"trained on data through {cutoff_year}")
print("trained slope (F per decade):", round(train_result.slope * 10, 3))

full_trend_clean = monthly.rolling(window=12).mean().dropna()
all_years_since_start = (full_trend_clean.index - train_trend.index[0]).days / 365.25

predicted_trend = train_result.intercept + train_result.slope * all_years_since_start

comparison = pd.DataFrame({
    "actual": full_trend_clean.values,
    "predicted": predicted_trend.values
}, index=full_trend_clean.index)

test_period = comparison[comparison.index.year > cutoff_year]

errors = test_period["actual"] - test_period["predicted"]
print("\nmean error (actual - predicted):", round(errors.mean(), 3))
print("mean absolute error:", round(errors.abs().mean(), 3))
print("rms error:", round((errors ** 2).mean() ** 0.5, 3))

fig, ax = plt.subplots()
ax.plot(monthly.index, monthly.values, color="lightgray", label="monthly average")
ax.plot(trend_clean.index, trend_clean.values, color="red", label="12-month rolling avg")
ax.set_title("Fairfax area temperature, 1990-2026")
ax.set_xlabel("year")
ax.set_ylabel("avg temperature (F)")
ax.legend()
plt.savefig("raw_and_trend.png")
plt.close()

fig, ax = plt.subplots()
ax.plot(trend_clean.index, trend_clean.values, color="red", label="rolling average")
fitted_line = result.intercept + result.slope * years_since_start
ax.plot(trend_clean.index, fitted_line, color="black", linestyle="--",
        label=f"fitted line ({result.slope * 10:.2f} F/decade)")
ax.set_title("warming trend, isolated")
ax.set_xlabel("year")
ax.set_ylabel("12-month rolling avg temp (F)")
ax.legend()
plt.savefig("trend_fit.png")
plt.close()

fig, ax = plt.subplots()
ax.hist(historical_summers.values, bins=12, color="lightgray", edgecolor="black", label="previous summers")
ax.axvline(this_summer_avg, color="red", label=f"{this_year} summer")
ax.set_title(f"{this_year} summer vs previous {len(historical_summers)} summers")
ax.set_xlabel("avg summer temperature (F)")
ax.set_ylabel("number of years")
ax.legend()
plt.savefig("summer_comparison.png")
plt.close()

print("\nsaved raw_and_trend.png, trend_fit.png, summer_comparison.png")