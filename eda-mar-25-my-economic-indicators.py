import pandas as pd
import requests as r
from datetime import datetime
import matplotlib.pyplot as plt

# Get the datasource
mei_data = r.get("https://api.data.gov.my/data-catalogue?id=economic_indicators").json()

meidata = pd.DataFrame.from_records(mei_data)
meidata['date'] = pd.to_datetime(meidata['date'])

meidata = meidata.sort_values(by='date', ascending=False)

# Calculate YoY changes for lagging, leading, and coincident indexes
meidata_yoy = meidata.copy()
for idx in ['lagging', 'leading', 'coincident']:
    meidata_yoy[f'{idx}_yoy'] = meidata_yoy[idx].pct_change(periods=-12) * 100

# Drop the diffusion indexes and keep only date and YoY columns
cols_to_keep = ['date', 'leading_yoy', 'coincident_yoy']
meidata_yoy = meidata_yoy[cols_to_keep]
meidata_yoy = meidata_yoy.sort_values('date', ascending=True).reset_index(drop=True)

meidata_yoy_12m = meidata_yoy[meidata_yoy['date']>'2024-04-01']
meidata_yoy_12m = meidata_yoy_12m.sort_values(by='date', ascending=True).reset_index(drop=True)
meidata_cols_yoy_12m = meidata_yoy_12m.columns.tolist()
# Convert the data to 1 decimal place, except for the date
for col in meidata_cols_yoy_12m[1:]:
    meidata_yoy_12m[col] = meidata_yoy_12m[col].round(1)

meidata_12m = meidata[meidata['date']>'2024-04-01']
meidata_12m = meidata_12m.sort_values(by='date', ascending=True).reset_index(drop=True)
meidata_cols_12m = meidata_12m.columns.tolist()
# Convert the data to 1 decimal place, except for the date
for col in meidata_cols_12m[1:]:
    meidata_12m[col] = meidata_12m[col].round(1)

fig, ax = plt.subplots(figsize=(6, 6))

# Plot the two lines
ax.plot(meidata_yoy_12m['date'], meidata_yoy_12m['leading_yoy'], color='red', marker='o', label='Leading YoY (%)')
ax.plot(meidata_yoy_12m['date'], meidata_yoy_12m['coincident_yoy'], color='black', marker='o', linestyle = '--', label='Coincident YoY (%)')

# Set y-label
ax.set_ylabel('YoY Changes (%)', color='#5a5a5a')
ax.set_xlabel('', color='#5a5a5a')

# Set title
ax.set_title('Economic Indicators: May 2025', fontsize=14, color='black')

# Set legend at bottom left
ax.legend(loc='lower left', frameon=True)

# Set chart box color
for spine in ax.spines.values():
    spine.set_edgecolor('#cecece')

ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')

# Set grid lines
ax.yaxis.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()