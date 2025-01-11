import pandas as pd
from mysql.connector import connect
from os import environ
from dotenv import load_dotenv
from pandas import DataFrame, set_option

set_option('display.max_columns', None)


def DBConnectionCursor():
    load_dotenv()

    connection = connect(
        host=environ["HOST"],
        user=environ["USER"],
        password=environ["PASSWORD"],
        database=environ["DATABASE"],
        port=3306,
        connection_timeout=800
    )

    cursor = connection.cursor()
    return connection, cursor


con, cur = DBConnectionCursor()

def checkTables():
    cur.execute("""
    SELECT TABLE_SCHEMA, TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE';
    """)
    tables = cur.fetchall()
    for table in tables:
        print(table)


def tableToDF(table: str):
    """
    IMPORTANT TABLES
    hltv_stats
    prizepicks_lines
    prizepicks_lines_proj
    """
    cur.execute("SELECT * FROM " + table)
    table = cur.fetchall()
    cols = [colName[0] for colName in cur.description]
    df = DataFrame(table, columns=cols)
    return df


def getMergedTable():

    hltv = tableToDF("hltv_stats")
    hltv["date"] = pd.to_datetime(hltv["date"]).dt.normalize()
    lines = tableToDF("prizepicks_lines")
    lines["game_date"] = pd.to_datetime(lines["game_date"]).dt.normalize()
    lines["line_score"] = lines["line_score"].astype(float)
    odds = tableToDF("bovado_odds")
    odds["date"] = pd.to_datetime(odds["date"]).dt.normalize()

    lineOdds1 = pd.merge(
        lines, odds,
        left_on=["game_date", "player_team"],
        right_on=["date", "team1"],
        how="inner"
    )

    lineOdds1["odds"] = lineOdds1["team1_odd"]

    lineOdds2 = pd.merge(
        lines, odds,
        left_on=["game_date", "player_team"],
        right_on=["date", "team2"],
        how="inner"
    )

    lineOdds2["odds"] = lineOdds2["team2_odd"]

    lineOdds = pd.concat([lineOdds1, lineOdds2], axis=0)
    lineOdds["odds"] = lineOdds["odds"].replace({"\+": ""}, regex=True).astype(float)

    lineOdds.loc[lineOdds["odds"] > 0, "odds"] = 100/((lineOdds[lineOdds["odds"] > 0]["odds"]) + 100)
    lineOdds.loc[lineOdds["odds"] < 0, "odds"] = -(lineOdds[lineOdds["odds"] < 0]["odds"])/(-(lineOdds[lineOdds["odds"] < 0]["odds"]) + 100)
    lineOdds["odds"] = lineOdds["odds"]

    lineOddsCols = ["player_name", "game_date", "stat_type", "line_score", "odds"]

    lineOdds = lineOdds[lineOddsCols]

    hltv = hltv[["name", "date", "map_number", "kills", "headshots", "assists", "deaths", "team_score", "opponent_score"]]
    map12Data = hltv[hltv["map_number"].isin(["1", "2"])].groupby(["name", "date"]).sum().reset_index()

    map12KillsLines = lineOdds[lineOdds["stat_type"] == "MAPS 1-2 Kills"]
    map12HSLines = lineOdds[lineOdds["stat_type"] == "MAPS 1-2 Headshots"]

    map12Lines = pd.merge(
        map12KillsLines,
        map12HSLines,
        on=["player_name", "game_date"],
        how="inner",
        suffixes=("", "_HS")
    )[["player_name", "game_date", "line_score", "line_score_HS", "odds"]]

    map12LineData = pd.merge(
        map12Lines,
        map12Data,
        left_on=["player_name", "game_date"],
        right_on=["name", "date"],
        how="inner"
    )[["name", "date", "map_number", "kills", "headshots", "assists", "deaths", "team_score", "opponent_score", "line_score", "line_score_HS", "odds"]]

    map12LineData["Rounds"] = map12LineData["team_score"] + map12LineData["opponent_score"]
    map12LineData["Round%"] = (map12LineData["Rounds"] - 26)/26
    map12LineData["KPR"] = map12LineData["kills"]/map12LineData["Rounds"]
    map12LineData["HSPR"] = map12LineData["headshots"]/map12LineData["Rounds"]
    map12LineData["APR"] = map12LineData["assists"] / map12LineData["Rounds"]
    map12LineData["DPR"] = map12LineData["deaths"] / map12LineData["Rounds"]
    map12LineData["HS%"] = map12LineData["headshots"] / map12LineData["kills"]

    map12LineData = map12LineData.sort_values(by="date").dropna()
    runningAverage = map12LineData.groupby("name")[["kills", "headshots", "HS%", "assists", "deaths", "Rounds", "Round%", "KPR", "HSPR", "APR", "DPR", "odds"]].shift(1).rolling(window=5, min_periods=1).mean()
    runningAverage = runningAverage.add_suffix("_avg")

    map12LineData = pd.concat([map12LineData, runningAverage], axis=1).dropna()

    # map3KillsLines = lineOdds[lineOdds["stat_type"] == "MAP 3 Kills"]
    # map3HSLines = lineOdds[lineOdds["stat_type"] == "MAP 3 Headshots"]
    # map3Data = hltv[hltv["map_number"] == "3"].groupby(["name", "date"]).sum().reset_index()

    return map12LineData.reset_index()