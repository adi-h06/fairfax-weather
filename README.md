# Fairfax Area Temperature Analysis
## Introduction

Analyzed 37 years (1990 - 2026) of temperature data in the Fairfax, Virginia area. Climate data was extracted from the NOAA National Centers for Environmental Information (NCEI).  

The program uses pandas to load and analyze the data by resampling the data into yearly rolling averages to identify a long term trend and ignore seasonal fluctuations. The program then fits a line to the trend and checks whether the summer of 2026 is an outlier compared to the 36 previous years using SciPy. 

## Results

### Long term trend

slope (F per decade): 0.9999857136682082
p-value: 2.5702817268009936e-51
r-value: 0.6425099408334637

### Comparing 2026 summer to 36 historical summers

this summer average: 77.03
historical summer mean: 74.85
historical summer std dev: 1.71

t-statistic: -7.684
p-value: 5.111476280292979e-09

According to the data, 2026 summer is a statistically significant outlier (p < 0.05)

### Testing the model's accuracy from 2015

trained slope (F per decade): 0.363

mean error (actual - predicted): 1.78
mean absolute error: 1.784
rms error: 1.941

A model trained on data through 2015 would have underestimated actual temperatures which suggests that temperature increase has accelerated in the past decade. 

## How to run

```
git clone https://github.com/adi-h06/fairfax-weather.git
pip install pandas scipy matplotlib
python fairfax_weather.py
```

Requires weather_data.csv which can be obtained at https://www.ncei.noaa.gov/cdo-web/search with the following search:

Weather Observation Type/Dataset: Daily Summaries
Date Range: 01/01/1990 - 08/12/2026
Search For: ZIP Codes
Search Term: 20166