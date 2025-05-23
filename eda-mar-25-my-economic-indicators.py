# %%
import pandas as pd
import requests as r
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Get the datasource
mei_data = r.get("https://api.data.gov.my/data-catalogue?id=economic_indicators").json()

meidata = pd.DataFrame.from_records(mei_data)
meidata['date'] = pd.to_datetime(meidata['date']).dt.strftime('%Y-%m')


# %%
# Manually imputing data, as the latest datapoint is not reflected in the API yet
meidata_new = pd.DataFrame({'date': ['2025-03'], 
                            'lagging': [145.4],
                            'leading': [112.5],
                            'coincident': [126.8],
                            'leading_diffusion': [71.400000],
                            'coincident_diffusion': [66.700000],})
meidata = pd.concat([meidata, meidata_new], ignore_index=True)
meidata = meidata.sort_values(by='date', ascending=False)

# %%
# Calculate YoY changes for lagging, leading, and coincident indexes
meidata_yoy = meidata.copy()
for idx in ['lagging', 'leading', 'coincident']:
    meidata_yoy[f'{idx}_yoy'] = meidata_yoy[idx].pct_change(periods=-12) * 100

# Drop the diffusion indexes and keep only date and YoY columns
cols_to_keep = ['date', 'leading_yoy', 'coincident_yoy']
meidata_yoy = meidata_yoy[cols_to_keep]
meidata_yoy = meidata_yoy.sort_values('date', ascending=True).reset_index(drop=True)

# %%
meidata_yoy_12m = meidata_yoy[meidata_yoy['date']>'2024-02']
meidata_yoy_12m = meidata_yoy_12m.sort_values(by='date', ascending=True).reset_index(drop=True)
meidata_cols_yoy_12m = meidata_yoy_12m.columns.tolist()
# Convert the data to 1 decimal place, except for the date
for col in meidata_cols_yoy_12m[1:]:
    meidata_yoy_12m[col] = meidata_yoy_12m[col].round(1)

# %%
meidata_12m = meidata[meidata['date']>'2024-02']
meidata_12m = meidata_12m.sort_values(by='date', ascending=True).reset_index(drop=True)
meidata_cols_12m = meidata_12m.columns.tolist()
# Convert the data to 1 decimal place, except for the date
for col in meidata_cols_12m[1:]:
    meidata_12m[col] = meidata_12m[col].round(1)

# %%
fig, ax = plt.subplots(figsize=(16, 9))

# Plot the two lines in greyscale
ax.plot(meidata_yoy_12m['date'], meidata_yoy_12m['leading_yoy'], color='red', marker='o', label='Leading YoY')
ax.plot(meidata_yoy_12m['date'], meidata_yoy_12m['coincident_yoy'], color='#868585', marker='o', linestyle = '--', label='Coincident YoY')

# Set y-label
ax.set_ylabel('YoY Changes %', color='#5a5a5a')

# Set title
ax.set_title('Malaysian Economic Indicators March 2025', fontsize=20, color='#5a5a5a')

# Set legend at bottom left
ax.legend(loc='lower left', frameon=True)

# Set chart box color
for spine in ax.spines.values():
    spine.set_edgecolor('#cecece')

# No x-axis title
ax.set_xlabel('')

# Improve x-tick labels
ax.set_xticklabels(meidata_yoy_12m['date'], rotation=45, ha='right', color="#5a5a5a")
ax.tick_params(axis='x', colors='#5a5a5a')
ax.tick_params(axis='y', colors='#5a5a5a')

# Set grid lines
ax.yaxis.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()

# %%
fig, ax1 = plt.subplots(figsize=(16, 9))

# Bar chart for YoY changes (primary y-axis)
bar_width = 0.25
x = range(len(meidata_yoy_12m['date']))

ax1.bar(x, meidata_yoy_12m['leading_yoy'], width=bar_width, color="#868585", label='Leading YoY')
ax1.bar([i + bar_width for i in x], meidata_yoy_12m['coincident_yoy'], width=bar_width, color="#5F5E5E", label='Coincident YoY')

ax1.set_ylabel('YoY Changes %', color='#5a5a5a')
ax1.tick_params(axis='y', colors='#5a5a5a')
ax1.set_xticks(x)
ax1.set_xticklabels(meidata_yoy_12m['date'], rotation=45, ha='right', color="#5a5a5a")
ax1.tick_params(axis='x', colors='#5a5a5a')

# # Secondary y-axis for diffusion index
# ax2 = ax1.twinx()
# ax2.set_ylabel('Diffusion Index %', color='#5a5a5a')
# ax2.plot(x, meidata_12m['leading_diffusion'], color="#4E4E4E", linestyle='--', linewidth=1, marker='.', label='Leading Diffusion')
# ax2.plot(x, meidata_12m['coincident_diffusion'], color="#000000", linestyle='--', linewidth=1, marker='.', label='Coincident Diffusion')

# Remove chart box
for spine in ax1.spines.values():
    spine.set_visible(False)
# for spine in ax2.spines.values():
#     spine.set_visible(False)

ax1.axvline(x=-1, color='#cecece', linewidth=1, linestyle='-')
# ax2.axvline(x=13, color='#cecece', linewidth=1, linestyle='-')
ax1.axhline(y=0, color='#cecece', linewidth=1, linestyle='-')

# Legends
bars_labels = ['Leading YoY', 'Coincident YoY']
# lines_labels = ['Leading Diffusion', 'Coincident Diffusion']
bars = [plt.Rectangle((0,0),1,1, color=c) for c in ['#868585', '#5F5E5E']]
# lines = [plt.Line2D([0], [0], color=c, linestyle='-', marker='o') for c in ["#4E4E4E", "#000000"]]
# ax1.legend(bars + lines, bars_labels + lines_labels, loc='upper right', frameon=True)
ax1.legend(bars, bars_labels, loc='upper right', frameon=True)

plt.title('Malaysian Economic Indicators March 2025', fontsize=20, color='#5a5a5a')
# plt.grid(True, linestyle='--', linewidth=0.5, color='#cecece')
plt.show()


