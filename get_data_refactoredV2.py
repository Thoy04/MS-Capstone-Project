from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
import pandas as pd


class GetActivePlayerIDs:
    """class to get data for active players from the NBA API"""
    def __init__(self):
        """initializing blank lists form ids, rows and exclusions"""
        self.ids = []

    def get_ids(self):
        active = players.get_active_players()
        for i in range(len(active)):
            id = active[i]['id']
            self.ids.append(id)
        print(f'{len(self.ids)} ids gathered')
        return self.ids


class GetPlayerStats:
    """class to get data for active players from the NBA API"""
    def __init__(self):
        """initializing blank lists form ids, rows and exclusions"""
        self.rows = []
        self.exclusions = []

    def get_stats(self, ids):
        """method to get the per game average stats for each active player using their id, excludes players who have
        no data for their latest season"""
        for id in ids:
            print(id)
            career = playercareerstats.PlayerCareerStats(per_mode36='PerGame', player_id=[id])
            try:
                last_row = career.get_data_frames()[0].iloc[-1]
                self.rows.append(last_row)
            except IndexError:
                self.exclusions.append(id)
        print(f"{len(self.rows)} players' stats gathered, {len(self.exclusions)} exclusions'")
        return self.rows, self.exclusions


class GetNames:
    def __init__(self):
        """initializing blank names list"""
        self.names = []

    def get_names(self, ids, exclusions):
        """method to get the name for each player from their id"""
        for id in ids:
            if id in exclusions:
                print('excluded')
            else:
                player = players.find_player_by_id(id)
                name = player['full_name']
                self.names.append(name)
        return self.names


if __name__ == '__main__':
    # instantiate class
    ids_retriever = GetActivePlayerIDs()
    # retrieve ids
    ids = ids_retriever.get_ids()
    # instantiate class
    stats_retriever = GetPlayerStats()
    # retrieve stats
    stats, exclusions = stats_retriever.get_stats(ids)
    # instantiate class
    names_retriever = GetPlayerStats()
    # retrieve names
    names = names_retriever.get_names(ids, exclusions)
    # transform stats and names lists to dataframe
    df = pd.DataFrame(stats)
    df['name'] = names
    # write dataframe to csv file
    df.to_csv('C:/Users/sierr/Desktop/MSDS/Capstone/NBA_Dataset_refactor.csv')

