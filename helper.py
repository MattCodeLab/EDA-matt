import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow

def read_parquet_by_name(name, src_path='src_parquet.csv'):
    src_df = pd.read_csv(src_path)
    result = src_df.loc[src_df['name'] == name, 'links']
    if not result.empty:
        url = result.values[0]
        df = pd.read_parquet(url)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        raise ValueError(f"No URL found for name: {name}")

def read_url_by_name(name, src_path='src_url.csv'):
    # Reads the src file, finds the URL for the given name, and loads the JSON API response into a pandas DataFrame.
    src_df = pd.read_csv(src_path)
    result = src_df.loc[src_df['name'] == name, 'links']
    if not result.empty:
        url = result.values[0]
        df = pd.read_json(url)
        return df
    else:
        raise ValueError(f"No URL found for name: {name}")


# Plot a chart with 2 lines, one red and one black, one is MoM and one is YoY

def filter_tail(df, column, number_of_values):
    df = df.sort_values(by=column, ascending=True)
    df = df.tail(number_of_values)
    return df

    
def plot_two_lines(
    df1, x1, y1, label1,
    df2, x2, y2, label2,
    xlabel='Date', ylabel='Growth (%)',
    title='Industrial Production Index: May-25',
    legend_loc='lower left',
    color1='black', color2='red',
    linestyle1='--', linestyle2='-',
    marker1='o', marker2='o',
    linewidth1=2, linewidth2=2,
    figsize=(6, 6)
):
    fig, ax = plt.subplots(figsize=figsize)

    # Plot first line
    ax.plot(
        df1[x1],
        df1[y1],
        color=color1,
        marker=marker1,
        linestyle=linestyle1,
        linewidth=linewidth1,
        label=label1
    )

    # Plot second line
    ax.plot(
        df2[x2],
        df2[y2],
        color=color2,
        marker=marker2,
        linestyle=linestyle2,
        linewidth=linewidth2,
        label=label2
    )

    # Set axis labels
    ax.set_ylabel(ylabel, color='black')
    ax.set_xlabel(xlabel, color='black')

    # Set title
    ax.set_title(title, fontsize=14)

    # Set spines color
    for spine in ax.spines.values():
        spine.set_edgecolor('#cecece')
    ax.spines['top'].set_color('white')

    # Set tick params color
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Add grid
    ax.grid(True, which='major', axis='both', alpha=0.3)

    # Add legend
    ax.legend(loc=legend_loc)

    # Set x-axis major locator to MonthLocator and formatter to show abbreviated month and year
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

    # Ensure the x-axis ticks and labels are visible and not overlapping
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=0, ha='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()

# Filter columns
def filter_columns(df, column, values):
    return df[df[column] == values]