#%%
import pandas as pd

#%%
# reading in data from Excel file
df_old = pd.read_csv('C:/Users/sierr/Desktop/MSDS/Capstone/NBA_Dataset_old.csv')

# checking to see if any null values exist
df_old.isna().sum()

# checking null rows of fg_pct to see if null is caused by a 0 in either 'fg_per_g' or 'fga_per_g' row
test = df[df['fg_pct'].isnull()][['fg_per_g','fga_per_g']]
# sum of 0 shows that there are no non-0s in column
test['fga_per_g'].sum()
# fill NA values with 0
df['fg_pct'].fillna(0, inplace=True)

# repeating same process for fg2_pct, fg3_pct, ft_pct columns
test = df[df['fg2_pct'].isnull()][['fg2_per_g','fg2a_per_g']]
test['fg2_per_g'].sum()
df['fg2_pct'].fillna(0, inplace=True)

# fg3 null handling
test = df[df['fg3_pct'].isnull()][['fg3_per_g','fg3a_per_g']]
test['fg3_per_g'].sum()
df['fg3_pct'].fillna(0, inplace=True)

# ft null handling
test = df[df['ft_pct'].isnull()][['ft_per_g','fta_per_g']]
test['ft_per_g'].sum()
df['ft_pct'].fillna(0, inplace=True)

# checking to see if any null values remain
df.isna().sum()

# creating columns to show the difference between stats of the current year and the previous year
df = df.sort_values(by=['player', 'season'])
df['mpg_diff'] = df.groupby('player')['mp_per_g'].diff()
df['ppg_diff'] = df.groupby('player')['pts_per_g'].diff()
df['astpg_diff'] = df.groupby('player')['ast_per_g'].diff()
df['rebpg_diff'] = df.groupby('player')['trb_per_g'].diff()
df['stl_diff'] = df.groupby('player')['stl_per_g'].diff()
df['blk_diff'] = df.groupby('player')['blk_per_g'].diff()
df['tov_diff'] = df.groupby('player')['tov_per_g'].diff()
df['season_diff'] = df.groupby('player')['season'].diff()
df['age_diff'] = df.groupby('player')['age'].diff()

# checking for null values based on new columns, each players first available season should be all nulls
df.isna().sum()
# checking to see if number of nulls is equal to the number of unique players
len(df['player'].unique())

# checking counts of season_diff column to check for erroneous values
df['season_diff'].value_counts()

# checking to make sure there are no more erroneous values
test = df[(df['season_diff'] != df['age_diff']) & (df['season'] !=2023) & (df['season_diff'] > 0)]

# saving completed dataframe to csv file
df.to_csv('C:/Users/sierr/Desktop/MSDS/Capstone/NBA_Dataset_currentV3.csv')
