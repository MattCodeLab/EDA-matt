# Import the relevant libraries
import pandas as pd
import requests as r
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Get the datasource
headline_inflation_data = r.get("https://api.data.gov.my/data-catalogue?id=cpi_headline_inflation").json()
core_inflation_data = r.get("https://api.data.gov.my/data-catalogue?id=cpi_core_inflation").json()

cinfdata = pd.DataFrame.from_records(core_inflation_data)
cinfdata['date'] = pd.to_datetime(cinfdata['date']).dt.strftime('%Y-%m')
cinfdata = cinfdata[cinfdata['division']== 'overall']
cinfdata = cinfdata[['date', 'inflation_yoy']]

# Manually imputing data, as the latest datapoint is not reflected in the API yet
cinfdata_new = pd.DataFrame({'date': ['2025-04'], 'inflation_yoy': [2.0]})
cinfdata = pd.concat([cinfdata, cinfdata_new], ignore_index=True)
cinfdata = cinfdata.sort_values(by='date', ascending=False)

# Renaming the column
cinfdata = cinfdata.rename(columns={'inflation_yoy': 'Core CPI Year-on-Year Growth %'})

infdata = pd.DataFrame.from_records(headline_inflation_data)
infdata['date'] = pd.to_datetime(infdata['date']).dt.strftime('%Y-%m')
infdata = infdata[infdata['division']== 'overall']
infdata = infdata[['date', 'inflation_yoy']]

# Manually imputing data, as the latest datapoint is not reflected in the API yet
infdata_new = pd.DataFrame({'date': ['2025-04'], 'inflation_yoy': [1.4]})
infdata = pd.concat([infdata, infdata_new], ignore_index=True)
infdata = infdata.sort_values(by='date', ascending=False)

# Renaming the column
infdata = infdata.rename(columns={'inflation_yoy': 'Headline CPI Year-on-Year Growth %'})

# Merging the DataFrames
inflation = pd.merge(infdata, cinfdata, on='date')
inflation = inflation.sort_values(by='date', ascending=False)

# Filter for datapoints across the past 12 months starting from May 2024
inflation = inflation[inflation['date']>'2024-04']
inflation = inflation.sort_values(by='date', ascending=True)

# Create a Combo Chart

fig, ax1 = plt.subplots(figsize=(12, 6))

# Create the bar chart on the first axis (left y-axis)
ax1.plot(inflation['date'], inflation['Headline CPI Year-on-Year Growth %'], color='grey', marker='o', linestyle='-', label='Headline CPI Year-on-Year Growth %')
ax1.set_ylabel('Headline CPI Year-on-Year Growth %', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xlabel('Date', color='black')

# Create the second axis (right y-axis)
ax2 = ax1.twinx()

# Create the line chart on the second axis (right y-axis)
ax2.plot(inflation['date'], inflation['Core CPI Year-on-Year Growth %'], color='black', marker='o', linestyle='-', label='Core CPI Year-on-Year Growth %')
ax2.set_ylabel('Core CPI Year-on-Year Growth %', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left')

plt.title('Headline vs Core CPI Year-on-Year Growth (As at April 2025)', color='black')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
