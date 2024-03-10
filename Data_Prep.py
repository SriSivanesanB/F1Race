import pandas as pd

# DataFrame_1: Races
races_df = pd.read_csv('races.csv')

# DataFrame_2: Results
results_df = pd.read_csv('results.csv')

# DataFrame_3: Driver Standings
driver_standings_df = pd.read_csv('driver_standings.csv')

# DataFrame_4: Constructor Standings
constructor_standings_df = pd.read_csv('constructor_standings.csv')

# DataFrame_5: Qualifying
qualifying_df = pd.read_csv('qualifying.csv')

# DataFrame_6: Weather
weather_df = pd.read_csv('weather.csv')

#LookUp Fuction

def lookup (df, team, points):
    df['lookup1'] = df.season.astype(str) + df[team] + df['round'].astype(str)
    df['lookup2'] = df.season.astype(str) + df[team] + (df['round']-1).astype(str)
    new_df = df.merge(df[['lookup1', points]], how = 'left', left_on='lookup2',right_on='lookup1')
    new_df.drop(['lookup1_x', 'lookup2', 'lookup1_y'], axis = 1, inplace = True)
    new_df.rename(columns = {points+'_x': points+'_after_race', points+'_y': points}, inplace = True)
    new_df[points].fillna(0, inplace = True)
    return new_df

#PreProcessing

driver_standings = lookup(driver_standings, 'driver', 'driver_points')
driver_standings = lookup(driver_standings, 'driver', 'driver_wins')
driver_standings = lookup(driver_standings, 'driver', 'driver_standings_pos')

driver_standings.drop(['driver_points_after_race', 'driver_wins_after_race', 'driver_standings_pos_after_race'], 
                      axis = 1, inplace = True)

constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_points')
constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_wins')
constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_standings_pos')

constructor_standings.drop(['constructor_points_after_race', 'constructor_wins_after_race','constructor_standings_pos_after_race' ],
                           axis = 1, inplace = True)


