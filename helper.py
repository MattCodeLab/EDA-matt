def read_parquet_by_name(name, src_path='src.csv'):
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

