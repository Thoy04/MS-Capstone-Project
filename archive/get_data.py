from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import boxscoreadvancedv2
import pandas as pd

active = players.get_active_players()
ids = []
for i in range(len(active)):
    id = active[i]['id']
    ids.append(id)

names = []
for id in ids:
    if id in exclusions:
        print('excluded')
    else:
        player = players.find_player_by_id(id)
        name = player['full_name']
        names.append(name)

df = pd.DataFrame()
exclusions = []
# for id in ids:
#     career = playercareerstats.PlayerCareerStats(per_mode36='PerGame', player_id=[id])
#     try:
#         last_row = career.get_data_frames()[0].iloc[-1]
#         df = pd.concat([df, last_row.to_frame().T])
#         df = df.reset_index(drop=True)
#     except IndexError:
#         print('stop')
#         exclusions.append(id)
rows = []
exclusions = []
for id in ids[0:100]:
    try:
        career = playercareerstats.PlayerCareerStats(per_mode36='PerGame', player_id=[id])
        last_row = career.get_dict()['resultSets'][0]['rowSet'][-1]
        rows.append(last_row)
    except IndexError:
        print('excluded')
        exclusions.append(id)
df = pd.DataFrame(rows, columns=career.get_dict()['resultSets'][0]['headers'])


df_2022 = pd.DataFrame()
exclusions = []
for id in ids:
    career = playercareerstats.PlayerCareerStats(per_mode36='PerGame', player_id=[id])
    try:
        last_row = career.get_data_frames()[0].iloc[-2]
        df_2022 = pd.concat([df_2022, last_row.to_frame().T])
        df_2022 = df_2022.reset_index(drop=True)
    except IndexError:
        print('stop')
        exclusions.append(id)

df_2022['player'] = names

df_2022_only = df_2022[df_2022['SEASON_ID']=='2022-23']
df_2022_only.to_csv('C:/Users/sierr/Desktop/MSDS/Capstone/NBA_Dataset_2022.csv')
