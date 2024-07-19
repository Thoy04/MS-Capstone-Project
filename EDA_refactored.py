import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from EDAtools import apply_smote, calc_3sigma_cutoffs, create_pie_chart, vote_diff_plot

# reading in data from excel file
df = pd.read_csv('C:/Users/sierr/Desktop/MSDS/Capstone/NBA_Dataset_currentV3.csv')

# creating columns to check if a player received votes and if they won the award for comparison
df['received_vote'] = df['award_share'].apply(lambda x: 1 if x > 0 else 0)
winners = df.groupby(by="season").max('award_share')
winners['won_mip'] = 1
df = df.merge(winners[["award_share", "won_mip"]], on=["season", "award_share"], how="left")
df["won_mip"] = df["won_mip"].fillna(value=0)

# updating counts so all are numeric, then updating type to float
df.loc[df['fg2_pct'] == '#DIV/0!', 'fg2_pct'] = 0
df['fg2_pct'] = df['fg2_pct'].astype('float64')

# removing unnecessary columns, trb_per_g included
df = df.drop(columns=["Unnamed: 0", "fga_per_g", "fg3a_per_g", "fg2a_per_g", "fta_per_g", "trb_per_g"], axis=1)

# checking correlations to see what we could filter off of - diff columns have highest correlations with receiving votes
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show(block=True)

# plotting difference for each stat between players who received votes and players who did not
plot_cols = ['g', 'gs', 'mp_per_g', 'pts_per_g', 'mpg_diff', 'ppg_diff', 'astpg_diff', 'rebpg_diff', 'tov_diff']
vote_diff_plot(plot_cols, df)

# calculating threshold for each stat where players can no longer win the award
sigma_cols = ['mp_per_g', 'pts_per_g', 'mpg_diff', 'ppg_diff', 'astpg_diff', 'rebpg_diff', 'tov_diff']
sigma_cutoffs = calc_3sigma_cutoffs(sigma_cols, df)

# creating one dictionary with all necessary cutoffs, 'g' column was found by manually exploring data
cutoffs = sigma_cutoffs
cutoffs['g'] = 29

# creating a copy of df to remove players based on cutoffs
filtered_df = df.copy()

# removing players using the cutoffs dictionary
for key, val in cutoffs.items():
    filtered_df = filtered_df[filtered_df[key] >= val]

# removing players who cannot win based on previous accomplishments
filtered_df = filtered_df[(filtered_df['previous_total'] == 0) & (filtered_df['mpg_diff'].notnull())]

# filtering results in more balanced data, however it is still quite imbalanced
create_pie_chart(filtered_df)

# Using SMOTE to balance
cols_to_drop = ['player', 'team_id', 'previous_mvp', 'previous_mip', 'previous_all_star', 'previous_all_nba',
                'previous_total', 'season_diff', 'age_diff', 'received_vote', 'won_mip']
class_val = .1
final_df = apply_smote(cols_to_drop, filtered_df, class_val)

# looking at results of smote process
create_pie_chart(final_df)
