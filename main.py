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


def padString(s, l):
	side = True
	while len(s) < l:

		if side:
			s = s + " "
		else:
			s = " " + s

		side = not side

	return s


class Bracket:

	def __init__(self):
		self.bracket = []

	def __hash__(self):
		return hash(
			tuple(tuple(x) for x in self.bracket)
		)

	def __str__(self):
		for i in range(4):
			print("Region", i + 1)

			print_bracket = [""] * 31

			# Add the first round in
			for j in range( i*16, (i+1)*16 ):

				if j % 2 == 0:
					print_bracket[j % 16] += padString(self.bracket[0][j], 25)
				else:
					print_bracket[j % 16] += " " * 25

			# Add arrows to second
			for j in range(8):
				print_bracket[j * 4] += "-\\ "
				print_bracket[j * 4 + 1] += "  -"
				print_bracket[j * 4 + 2] += "-/ "

			for j in range( i*8, (i+1)*8 ):

				if j % 2 == 0:
					print_bracket[j % 8] += padString(self.bracket[1][j], 25)
				else:
					print_bracket[j % 8] += " " * 25

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

	def __init__(self, model, tournament):

		self.model = model
		self.tournament = tournament

	def predictGame(self, team_a, team_b):
		"""
		Predicts the probability of a win between A and B then randomly selects a winner based
		on that probability

		Does not matter what team is team_a and what team is team_b

		:param team_a: The data series referring to team_a
		:param team_b: The data series referring to team_b
		:return: The winning team
		"""

		game = buildGame(team_a, team_b)

		game1, game2 = self.model.predict_proba(game)

		team_a_win_chance = game1[1] + game2[0]

		if random.uniform(0, 2) < team_a_win_chance:
			return team_a

		return team_b

	def predictTournament(self):
		"""
		Predicts the bracket of the March Madness tournament

		:param tournament: Array of arrays where adjacent arrays are games to be played
		:return: Returns a resulting bracket object
		"""
		bracket = Bracket()

		current_round = self.tournament
		i = 0
		while len(current_round) > 1: # While there are games to predict

			bracket.append(tuple(current_round.map(getName)))

			next_round = []
			for j in range(len(current_round) / 2): # Predict all games and populate the next round

				winning_team = self.predictGame(current_round[2 * j], current_round[2 * j + 2])
				next_round.append(winning_team)

			current_round = next_round

			i += 1

		bracket.append(tuple(current_round.map(getName)))

		return bracket

	def prediction(self, iterations):
		"""
		Runs prediction multiple times on the tournament to get the most probable outcome

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