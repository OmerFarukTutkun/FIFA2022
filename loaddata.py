import pandas as pd
from torch.utils.data import Dataset
import torch

MIN_YEAR = 1980
#read whole data to memory as it does not take much space
#number of teams = 316
def loadFifaData(filename):

    print("started reading data" )
    teams = {}
    x = []
    y = []

    df = pd.read_csv(filename)
    data_shape = df.shape

    for i in range(data_shape[0]):
        item = df.loc[i]
        if item['home_team'] not in teams:
            teams[ item['home_team'] ] = len(teams)

        if item['away_team'] not in teams:
            teams[ item['away_team'] ] = len(teams)
        date = item['date']
        year = date.split("-")[0]
        if int(year) >= MIN_YEAR:
            x.append( [ teams [ item['home_team' ]] , teams [ item['away_team' ]] ] )
            y.append( [item['home_score'] , item['away_score']])

    print(f"finished reading  data , Number of data {len(y)}" )
    return x , y, teams

class FIFADataset(Dataset):
    def __init__(self ,filename = 'results.csv'):
        self.x , self.y , self.teams = loadFifaData(filename)

    def __len__(self):
        return len(self.y)

    def get_numOfTeams(self):
        return len(self.teams)
    def getTensorFromTeams(self, team1, team2):
        tensor_x = torch.zeros(2*self.get_numOfTeams(), )
        tensor_x[self.teams[team1 ]] = 1.0
        tensor_x[self.teams[team2 ] + self.get_numOfTeams()] = 1.0
        return tensor_x
    def __getitem__(self, idx):

        teams = self.x[idx]
        scores = self.y[idx]

        tensor_x = torch.zeros(2*self.get_numOfTeams(), )
        tensor_x[teams[0]] = 1.0
        tensor_x[teams[1] + self.get_numOfTeams()] = 1.0

        if scores[0] > scores[1]:
            tensor_y= torch.tensor( [1.0])
        elif scores[0] == scores[1]:
            tensor_y= torch.tensor( [0.5])
        else:
            tensor_y= torch.tensor( [0.0])
        return tensor_x, tensor_y