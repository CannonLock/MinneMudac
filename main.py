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

def createModels():
	tourney = pd.read_csv('Final_Data.csv')
	prediction_data = pd.read_csv('Input_Data.csv')
	tourney.drop(['Unnamed: 0'], axis = 1, inplace = True)
	tourney['point_diff'] = (tourney['Tm.'] - tourney['Opp.'])
	tourney['ppg'] = tourney['Tm.'] / tourney['G']
	tourney['rpg'] = tourney['TRB'] / tourney['G']
	tourney['apg'] = tourney['AST'] / tourney['G']
	tourney['spg'] = tourney['STL'] / tourney['G']
	tourney['bpg'] = tourney['BLK'] / tourney['G']
	tourney['tpg'] = tourney['TOV'] / tourney['G']
	tourney['ftapg'] = tourney['FTA'] / tourney['G']
	tourney['papg'] = tourney['Opp.'] / tourney['G']
	tourney['opp_point_diff'] = tourney['Opp_Tm'] - tourney['Opp_Opp']
	tourney['opp_ppg'] = tourney['Opp_Tm'] / tourney['Opp_G']
	tourney['opp_rpg'] = tourney['Opp_TRB'] / tourney['Opp_G']
	tourney['opp_ftapg'] = tourney['Opp_FTA'] / tourney['G']
	tourney['opp_apg'] = tourney['Opp_AST'] / tourney['Opp_G']
	tourney['opp_spg'] = tourney['Opp_STL'] / tourney['Opp_G']
	tourney['opp_bpg'] = tourney['Opp_BLK'] / tourney['Opp_G']
	tourney['opp_tpg'] = tourney['Opp_TOV'] / tourney['Opp_G']
	tourney['Round'] = tourney['Round'].apply(lambda x: str(x) + "_round")
	tourney['Seed'] = tourney['Seed'].apply(lambda x: str(x) + "_seed")
	tourney['Opp_Seed'] = tourney['Opp_Seed'].apply(lambda x: str(x) + "_seedopp")
	#tourney['Seed'] = tourney['Seed'].astype('category')
	#tourney['Opp_Seed'] = tourney['Opp_Seed'].astype('category')
	prediction_data.drop(['Unnamed: 0'], axis = 1, inplace = True)
	prediction_data['point_diff'] = prediction_data['Tm.'] - prediction_data['Opp.']
	prediction_data['ppg'] = prediction_data['Tm.'] / prediction_data['G']
	prediction_data['rpg'] = prediction_data['TRB'] / prediction_data['G']
	prediction_data['apg'] = prediction_data['AST'] / prediction_data['G']
	prediction_data['spg'] = prediction_data['STL'] / prediction_data['G']
	prediction_data['bpg'] = prediction_data['BLK'] / prediction_data['G']
	prediction_data['tpg'] = prediction_data['TOV'] / prediction_data['G']
	prediction_data['papg'] = prediction_data['Opp.'] / prediction_data['G']
	prediction_data['opp_point_diff'] = prediction_data['Opp_Tm'] - prediction_data['Opp_Opp']
	prediction_data['opp_ppg'] = prediction_data['Opp_Tm'] / prediction_data['Opp_G']
	prediction_data['opp_rpg'] = prediction_data['Opp_TRB'] / prediction_data['Opp_G']
	prediction_data['opp_apg'] = prediction_data['Opp_AST'] / prediction_data['Opp_G']
	prediction_data['opp_spg'] = prediction_data['Opp_STL'] / prediction_data['Opp_G']
	prediction_data['opp_bpg'] = prediction_data['Opp_BLK'] / prediction_data['Opp_G']
	prediction_data['opp_tpg'] = prediction_data['Opp_TOV'] / prediction_data['Opp_G']
	prediction_data['Round'] = prediction_data['Round'].apply(lambda x: str(x) + "_round")
	prediction_data['Seed'] = prediction_data['Seed'].apply(lambda x: str(x) + "_seed")
	prediction_data['Opp_Seed'] = prediction_data['Opp_Seed'].apply(lambda x: str(x) + "_seedopp")

	y = tourney['Win']
	X = tourney[['W-L%','SRS','SOS','point_diff','FG%','Opp_WLRat','Opp_SRS','Opp_SOS','opp_point_diff','Opp_FGperc']]
	X = pd.concat([X,pd.get_dummies(tourney['Seed']),pd.get_dummies(tourney['Opp_Seed'])], axis = 1)
	X = X.drop(['16_seed','16_seedopp'], axis = 1)
	prediction_data = pd.concat([prediction_data,pd.get_dummies(prediction_data['Seed']),pd.get_dummies(prediction_data['Opp_Seed'])], axis = 1)
	prediction_data = prediction_data.drop(['16_seed','16_seedopp'], axis = 1)

	# Train the models
	lr = LogisticRegression(max_iter = 2000)
	lr.fit(X,y)
	rf = RandomForestClassifier(n_estimators = 1000)
	rf.fit(X,y)

	return [lr,rf]

class Bracket:

	def __init__(self):
		self.bracket = []

	def __hash__(self):
		return hash(
			tuple(tuple(x) for x in self.bracket)
		)

	def addRound(self, round, winArray):

		if len(winArray) != 64 / (2**round):
			print("Invalid Round added")

		self.bracket.append(winArray)

	def trimObject(self):
		return tuple(tuple(x) for x in self.bracket)

class TournamentSimulation:

	def __init__(self, models, tournament_file):

		self.models = models

		self.orderedTeams = self.buildTeams(tournament_file)

	def printMatchUp(self, team_a, team_b):
		print(team_a["Name"], ".vs.", team_b["Name"])

	def buildTeams(self, tournament_file):
		# Get the tournament
		df = pd.read_csv(tournament_file)

		print(df.head())

		# Extract the names
		names = df["Name"].to_numpy()
		df.drop("Name", axis=1)

		# Add the point diff
		df['point_diff'] = (df['Tm.'] - df['Opp.'])
		df.drop(['Tm.', "Opp."], axis=1)

		# Re order the cols
		final_df = df[['W-L%','SRS','SOS','point_diff','FG%']]

		# Encode the Seeds
		final_df = pd.concat([final_df, pd.get_dummies(df['Seed'])], axis=1)
		final_df = final_df.drop([16], axis=1)

		# Extract the Stats
		stats = final_df.to_numpy()

		# Build a team object
		teams = []
		for i in range(len(names)):
			teams.append({"Name": names[i], "Stats": stats[i]})

		return teams

	def buildGame(self, team_a, team_b):

		game_0 = np.append(team_a["Stats"], team_b["Stats"])
		game_1 = np.append(team_b["Stats"], team_a["Stats"])

		return np.array([game_0, game_1])

	def predictGameMaxAvg(self, team_a, team_b):
		"""
		Predicts the probability of a win between A and B selects the winner
		based on the most probable outcome

		Does not matter what team is team_a and what team is team_b

		:param team_a: The data series referring to team_a
		:param team_b: The data series referring to team_b
		:return: The winning team
		"""

		game = self.buildGame(team_a, team_b)

		predictions = []

		# Get the model predictions
		for model in self.models:
			a_versus_b, b_versus_a = model.predict_proba(game)
			predictions.extend([a_versus_b, np.flip(b_versus_a)])

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

		game = self.buildGame(team_a, team_b)

		predictions = []

		# Get the model predictions
		for model in self.models:
			a_versus_b, b_versus_a = model.predict_proba(game)
			predictions.extend([a_versus_b, np.flip(b_versus_a)])

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

		game = self.buildGame(team_a, team_b)

		predictions = []

		# Predict the game probabilities for the model
		for model in self.models:
			a_versus_b, b_versus_a = model.predict_proba(game)
			predictions.extend([a_versus_b, np.flip(b_versus_a)])

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

		game = self.buildGame(team_a, team_b)

		predictions = []

		# Predict the game probabilities for the model
		for model in self.models:
			a_versus_b, b_versus_a = model.predict_proba(game)
			predictions.extend([a_versus_b, np.flip(b_versus_a)])

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
		if method == "MaxAvg":
			predictGame = self.predictGameMaxAvg

		elif method == "MaxMax":
			predictGame = self.predictGameMaxMax

		elif method == "RandAvg":
			predictGame = self.predictGameRandAvg

		else :
			predictGame = self.predictGameRandMax


		current_round = self.orderedTeams
		i = 0
		while len(current_round) > 1: # While there are games to predict

			# print("Round:", i+1)
			bracket.addRound(i+1, tuple(map(lambda x: x["Name"], current_round)))

			next_round = []
			for j in range(int(len(current_round) / 2)): # Predict all games and populate the next round

				self.printMatchUp(current_round[2 * j], current_round[2 * j + 1])
				winning_team = predictGame(current_round[2 * j], current_round[2 * j + 1])

				next_round.append(winning_team)

			current_round = next_round

			i += 1

		bracket.addRound(i+1, tuple(map(lambda x: x["Name"], current_round)))

		return bracket

	def prediction(self, iterations, method=None):
		"""
		Runs prediction multiple times on the tournament to get the most probable outcome

		:param method: The method which to predict the games
		:param iterations: Number of times to simulate the tournament
		:return: Bracket that occurred the most
		"""

		predictions = {}

		# Run the prediction and catalogue the most likely answers
		for i in range(iterations):
			b = self.predictTournament(method)

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

if __name__ == "__main__":

	# Get the trained Models
	models = createModels()

	sim_0 = TournamentSimulation(models=[models[0]], tournament_file="Prediction_Feed.csv")

	a = sim_0.prediction(1, "MaxAvg")
	b = sim_0.prediction(1, "MaxMax")

	for row in a.trimObject():
		print(row)

	for row in b.trimObject():
		print(row)

	sim_1 = TournamentSimulation(models=[models[1]], tournament_file="Prediction_Feed.csv")

	a = sim_1.prediction(1, "MaxAvg")
	b = sim_1.prediction(1, "MaxMax")

	for row in a.trimObject():
		print(row)

	for row in b.trimObject():
		print(row)