import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from venantvr.business.bot_currency_pair import BotCurrencyPair
from venantvr.types.types_alias import GateioTimeFrame

from apprentissage.trainers.generic_trainer import GenericTrainer


# La classe RegressionTrainer hérite de GenericTrainer et implémente un modèle de régression
# linéaire pour prédire la variation du prix d'une paire de devises en fonction du volume et du RSI.
class RegressionTrainer(GenericTrainer):

    def __init__(self, currency_pair: BotCurrencyPair, dataframe, small_timeframe: GateioTimeFrame, middle_timeframe: GateioTimeFrame, model_file: str):
        # Appelle le constructeur de la classe parente GenericTrainer, en passant
        # la paire de devises, les timeframes et le fichier du modèle.
        super().__init__(currency_pair, dataframe, small_timeframe, middle_timeframe, model_file)

    # Méthode pour entraîner et prédire la variation du prix en fonction du volume et du RSI.
    def predict(self, close_column: str, volume_column: str, rsi_column: str):
        # Nom de la nouvelle colonne qui stockera la variation en pourcentage du prix de clôture
        percentage_change = 'percentage_change'

        # Crée une copie du DataFrame d'origine, ne gardant que les colonnes de prix de clôture,
        # de volume et de RSI spécifiés par l'utilisateur.
        model_dataframe = self.dataframe[[close_column, volume_column, rsi_column]].copy()

        # Calcule la variation en pourcentage du prix de clôture entre chaque ligne.
        # pct_change() calcule la variation relative, multipliée par 100 pour avoir le pourcentage.
        model_dataframe[percentage_change] = model_dataframe[close_column].pct_change() * 100.0

        # Supprime toutes les lignes contenant des valeurs NaN (après l'application de pct_change).
        # Cela évite les erreurs lors de l'entraînement du modèle avec des données manquantes.
        model_dataframe.dropna(inplace=True)

        # X représente les features (variables d'entrée) utilisées pour prédire la variation en pourcentage.
        # Ici, nous utilisons les colonnes de volume et de RSI comme features.
        x = model_dataframe[[volume_column, rsi_column]]

        # Y est la variable cible, qui représente la variation en pourcentage du prix de clôture.
        y = model_dataframe[percentage_change]

        # Sépare les données en un jeu d'entraînement (95%) et un jeu de test (5%).
        # Cela permet d'entraîner le modèle sur un sous-ensemble des données et de tester ses performances sur un autre.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

        # Vérifie si un modèle a déjà été chargé ou créé.
        # Si aucun modèle n'est présent, crée un modèle de régression linéaire.
        if self.model_blob is None:
            # Crée un nouveau modèle de régression linéaire.
            self.model_blob = LinearRegression()

            # Entraîne le modèle sur le jeu d'entraînement avec les variables d'entrée (volume et RSI)
            # et la cible (variation en pourcentage du prix).
            self.model_blob.fit(x_train, y_train)

            # Sauvegarde le modèle entraîné dans le fichier spécifié lors de l'initialisation.
            # Cela permet de réutiliser le modèle sans avoir à le réentraîner.
            self.dump_model(model=self.model_blob, file=self.model_file)

        # Récupère les dernières valeurs de volume et de RSI pour effectuer une prédiction.
        # Il s'agit des données les plus récentes, qui seront utilisées pour prédire la variation du prix.
        volume = self.dataframe[volume_column].iloc[-1]
        rsi = self.dataframe[rsi_column].iloc[-1]

        # Crée un DataFrame avec une seule ligne contenant les valeurs récentes de volume et de RSI.
        # C'est sur cette base que la prédiction sera faite.
        data_to_predict = pd.DataFrame({
            volume_column: [volume],
            rsi_column: [rsi]
        })

        # Utilise le modèle entraîné pour prédire la variation en pourcentage du prix de clôture
        # basée sur les dernières valeurs de volume et de RSI.
        output = self.model_blob.predict(data_to_predict)[0]

        # Retourne la prédiction (variation en pourcentage du prix de clôture).
        return output
