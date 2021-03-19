import numpy as np
import pandas as pd
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from multiprocessing import pool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def cleanTournamentData():
	t = pd.read_csv("MarchMadnessResults.csv")

	t.rename(columns={"Seed": "Seed1", "Score": "Score1", "Team": "Team1",
	                  "Seed.1": "Seed2", "Score.1": "Score2", "Team.1": "Team2",
	                  "Region Number": "RegionNumber", "Region Name": "RegionName"})

	t[["Team1", "Team2", "RegionName"]] = t[["Team1", "Team2", "RegionName"]].astype("category")

	# 1 if Team1 wins, 0 otherwise
	t["Result"] = t["Score1"] > t["Score2"]
	t["Result"] = t["Result"].astype("Int64")

	t.to_csv("MarchMadnessResultsClean.csv")


def buildDataSet(teamFile, tournamentFile):


def buildGame(team_a, team_b):
	team_a1 = team_a.add_suffix("1")
	team_b2 = team_b.add_suffix("2")
	game1 = team_a1.append(team_b2)
	team_a2 = team_a.add_suffix("2")
	team_b1 = team_b.add_suffix("1")
	game2 = team_b1.append(team_a2)

	return pd.concat([game1, game2], axis=1).T

class Bracket:

	def __init__(self):
		self.bracket = []

	def __hash__(self):
		return hash(
			tuple(tuple(x) for x in self.bracket)
		)

	def append(self, item):
		self.bracket.append(item)

	def trimObject(self):
		return tuple(tuple(x) for x in self.bracket)


def getName(a):
	"""
	Return the team name in this array
	:param a: The data array
	:return: The team name
	"""

	return "TODO"


class TournamentSimulation:

	def __init__(self, models, tournament_file):

		self.models = models

		self.orderedTeams = self.buildTeams(tournament_file)

	def buildTeams(self, tournament_file):
		# Get the tournament
		df = pd.read_csv(tournament_file)

		# Extract the names
		names = df["Name"].to_numpy()
		df.drop("Name", axis=1)

		# Extract the Stats
		stats = df.to_numpy()

		# Build a team object
		teams = []
		for i in range(len(names)):
			teams.append({"Name": names[i], "Stats": stats[i]})

		return teams

	def predictGameMaxAvg(self, team_a, team_b):
		"""
		Predicts the probability of a win between A and B selects the winner
		based on the most probable outcome

		Does not matter what team is team_a and what team is team_b

		:param team_a: The data series referring to team_a
		:param team_b: The data series referring to team_b
		:return: The winning team
		"""

		game = buildGame(team_a, team_b)

		predictions = []

		# Get the model predictions
		for model in self.models:
			a_versus_b, b_versus_a = model.predict_proba(game)
			predictions.extend([a_versus_b, b_versus_a.reverse()])

		# Get each teams win probability
		b_win_prob, a_win_prob = map(sum, zip(*predictions))

		# If b is more likely to win then let b move forward
		if b_win_prob > a_win_prob:
			return team_b
		else:
			return team_a

	def predictGameMaxMax(self, team_a, team_b):
		"""
		Predicts the probability of a win between A and B selects the winner
		based on the most probable outcome

		Does not matter what team is team_a and what team is team_b

		:param team_a: The data series referring to team_a
		:param team_b: The data series referring to team_b
		:return: The winning team
		"""

		game = buildGame(team_a, team_b)

		predictions = []

		# Get the model predictions
		for model in self.models:
			a_versus_b, b_versus_a = model.predict_proba(game)
			predictions.extend([a_versus_b, b_versus_a.reverse()])

		# Get each teams win probability
		highest_b_win_prob, highest_a_win_prob = map(max, zip(*predictions))

		# If b has the most likely chance of winning predict on it
		if highest_b_win_prob > highest_a_win_prob:
			return team_b

		# Else if A is more likely to win randomly predict on it
		else:
			return team_a

	def predictGameRandAvg(self, team_a, team_b):
		"""
		Predicts the probability of a win between A and B then randomly selects a winner based
		on that probability

		Does not matter what team is team_a and what team is team_b

		:param team_a: The data series referring to team_a
		:param team_b: The data series referring to team_b
		:return: The winning team
		"""

		game = buildGame(team_a, team_b)

		predictions = []

		# Predict the game probabilities for the model
		for model in self.models:
			a_versus_b, b_versus_a = model.predict_proba(game)
			predictions.extend([a_versus_b, b_versus_a.reverse()])

		# Get each teams win probability
		b_win_prob, a_win_prob = map(sum, zip(*predictions))

		# Choose a winning team based on the predicted probability
		if random.uniform(0, len(self.models)*2) < a_win_prob:
			return team_a

		return team_b

	def predictGameRandMax(self, team_a, team_b):
 		"""
		Finds the most confident models prediction and then randomly chooses a game on that prob

		Does not matter what team is team_a and what team is team_b

		:param team_a: The data series referring to team_a
		:param team_b: The data series referring to team_b
		:return: The winning team
		"""

		game = buildGame(team_a, team_b)

		predictions = []

		# Predict the game probabilities for the model
		for model in self.models:
			a_versus_b, b_versus_a = model.predict_proba(game)
			predictions.extend([a_versus_b, b_versus_a.reverse()])

		# Get each teams win probability
		highest_b_win_prob, highest_a_win_prob = map(max, zip(*predictions))

		# If b has the most likely chance of winning predict on it
		if highest_b_win_prob > highest_a_win_prob:

			if random.uniform(0, 1) < highest_b_win_prob:
				return team_b
			else:
				return team_a

		# Else if A is more likely to win randomly predict on it
		else:
			if random.uniform(0, 1) < highest_a_win_prob:
				return team_a
			else:
				return team_b

	def predictTournament(self, method):
		"""
		Predicts the bracket of the March Madness tournament

		:param method: The method used to predict the games
		:param tournament: Array of arrays where adjacent arrays are games to be played
		:return: Returns a resulting bracket object
		"""
		bracket = Bracket()

		# Choose the prediction technique
		predictGame = None
		if method == "Max":
			predictGame = self.predictGameMax

		elif method == "Rand":
			predictGame = self.predictGameRand



		current_round = self.tournament
		i = 0
		while len(current_round) > 1: # While there are games to predict

			bracket.append(tuple(current_round.map(getName)))

			next_round = []
			for j in range(len(current_round) / 2): # Predict all games and populate the next round

				winning_team = predictGame(current_round[2 * j], current_round[2 * j + 2])
				next_round.append(winning_team)

			current_round = next_round

			i += 1

		bracket.append(tuple(current_round.map(getName)))

		return bracket

	def prediction(self, iterations, method = "Rand"):
		"""
		Runs prediction multiple times on the tournament to get the most probable outcome

		:param method: The method which to predict the games
		:param iterations: Number of times to simulate the tournament
		:return: Bracket that occurred the most
		"""

		predictions = {}

		# Run the prediction and catalogue the most likely answers
		for i in range(iterations):
			b = self.predictTournament()

			if b in predictions:
				predictions[b] += 1

			else:
				predictions[b] = 1


		max_occurrence = 0
		most_probable_bracket = None
		for b in predictions:

			if predictions[b] > max_occurrence: # If this bracket occurred more often

				max_occurrence = predictions[b]
				most_probable_bracket = b

		return most_probable_bracket