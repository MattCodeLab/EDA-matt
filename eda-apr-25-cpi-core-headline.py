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
cinfdata['date'] = pd.to_datetime(cinfdata['date'])
cinfdata = cinfdata[cinfdata['division']== 'overall']
cinfdata = cinfdata[['date', 'inflation_yoy']]

# Manually imputing data, as the latest datapoint is not reflected in the API yet
# cinfdata_new = pd.DataFrame({'date': ['2025-04'], 'inflation_yoy': [2.0]})
# cinfdata = pd.concat([cinfdata, cinfdata_new], ignore_index=True)
cinfdata = cinfdata.sort_values(by='date', ascending=False)

# Renaming the column
cinfdata = cinfdata.rename(columns={'inflation_yoy': 'Core CPI YoY %'})

infdata = pd.DataFrame.from_records(headline_inflation_data)
infdata['date'] = pd.to_datetime(infdata['date'])
infdata = infdata[infdata['division']== 'overall']
infdata = infdata[['date', 'inflation_yoy']]

# Renaming the column
infdata = infdata.rename(columns={'inflation_yoy': 'Headline CPI YoY %'})

# Merging the DataFrames
inflation = pd.merge(infdata, cinfdata, on='date')
inflation = inflation.sort_values(by='date', ascending=False)

# Filter for datapoints across the past 12 months starting from May 2024
inflation = inflation[inflation['date']>'2024-05-01']
inflation = inflation.sort_values(by='date', ascending=True)

fig, ax1 = plt.subplots(figsize=(6, 6))

# Plot the two lines
ax1.plot(inflation['date'], inflation['Headline CPI YoY %'], color='red', marker='o', label='Headline CPI YoY (%)')
ax2 = ax1.twinx()
ax2.plot(inflation['date'], inflation['Core CPI YoY %'], color='black', marker='o', linestyle = '--', label='Core CPI YoY (%)')

# Set y-label
ax1.set_ylabel('Headline CPI YoY (%)', color='red')
ax2.set_ylabel('Core CPI YoY (%)', color='black')

# Set title
ax1.set_title('YoY Inflation: May 2025', fontsize=14, color='black')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left')

# Set chart box color
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor('#cecece')

# Set top spine to white for both axes
for ax in [ax1, ax2]:
    ax.spines['top'].set_color('white')

# Set grid lines
ax1.yaxis.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()