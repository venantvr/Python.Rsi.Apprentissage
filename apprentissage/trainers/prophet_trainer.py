import numpy as np
from prophet import Prophet
from venantvr.business.bot_currency_pair import BotCurrencyPair
from venantvr.dataframes.temporary_columns_manager import TemporaryColumnsManager
from venantvr.tooling.tooling_utils import list_diff
from venantvr.types.types_alias import GateioTimeFrame

from apprentissage.trainers.generic_trainer import GenericTrainer
from apprentissage.types import MlTranformLambda


# Classe ProphetTrainer pour entraîner un modèle Prophet et effectuer des prédictions basées sur le RSI d'une paire de devises.
class ProphetTrainer(GenericTrainer):

    def __init__(self, currency_pair: BotCurrencyPair, dataframe, small_timeframe: GateioTimeFrame, middle_timeframe: GateioTimeFrame, model_file: str):
        """
        Initialisation de ProphetTrainer.

        :param currency_pair: Paire de devises utilisée.
        :param dataframe: Données du marché sous forme de DataFrame.
        :param small_timeframe: Timeframe à court terme.
        :param middle_timeframe: Timeframe à moyen terme.
        :param model_file: Fichier pour sauvegarder le modèle Prophet.
        """
        # Appelle le constructeur de la classe parent GenericTrainer pour initialiser les attributs communs.
        super().__init__(currency_pair, dataframe, small_timeframe, middle_timeframe, model_file)

    def predict(self, transform: MlTranformLambda, features_columns: list[str], neighbors: int, small_timeframe: GateioTimeFrame,
                middle_timeframe: GateioTimeFrame):
        """
        Effectue des prédictions basées sur Prophet après transformation des données.

        :param transform: Fonction lambda qui applique des transformations au DataFrame.
        :param features_columns: Liste des colonnes de caractéristiques à utiliser.
        :param neighbors: Paramètre pour la compatibilité avec d'autres modèles (non utilisé ici).
        :param small_timeframe: Timeframe à court terme.
        :param middle_timeframe: Timeframe à moyen terme.
        :return: Tendance prédite et confiance de la prédiction.
        """
        # Définit les noms des colonnes pour les données temporelles (ds) et la cible (y).
        target_column = 'target'
        ds_column = 'ds'
        y_column = 'y'

        # Transformation du DataFrame initial avec la fonction `transform`, ajoutant les nouvelles colonnes nécessaires.
        transform_result = transform(self.dataframe)
        self.dataframe[target_column] = transform_result.target  # Ajout de la colonne 'target' au DataFrame.
        extra_features_columns = transform_result.extra_features_columns  # Colonnes supplémentaires créées par `transform`.
        initial_columns = transform_result.initial_columns  # Colonnes originales avant transformation.

        # Utilisation du gestionnaire de colonnes temporaires pour s'assurer que certaines colonnes sont nettoyées après usage.
        with TemporaryColumnsManager(dataframe=self.dataframe,
                                     drop=[target_column, ds_column, y_column] + list_diff(extra_features_columns, initial_columns)) as df:
            # Préparation des colonnes pour Prophet : 'ds' pour les dates, 'y' pour les valeurs cibles (ici, le RSI).
            df['ds'] = df.index  # La colonne 'ds' contient les timestamps (dates).
            df['y'] = df['rsi']  # La colonne 'y' contient les valeurs du RSI, utilisées comme cible.

            # Si le modèle n'a pas encore été entraîné, initialise et entraîne Prophet.
            if self.model_blob is None:
                # Initialisation du modèle Prophet avec les paramètres pour inclure les saisons quotidiennes et hebdomadaires.
                self.model_blob = Prophet(daily_seasonality=True, weekly_seasonality=True)
                self.model_blob.fit(df)  # Entraînement du modèle avec les données transformées.

                # Sauvegarde le modèle entraîné pour une utilisation future.
                self.dump_model(model=self.model_blob, file=self.model_file)

            # Génère un DataFrame futur pour les prédictions sur 10 périodes supplémentaires.
            future = self.model_blob.make_future_dataframe(periods=10)
            # Effectue des prédictions sur le futur DataFrame.
            forecast = self.model_blob.predict(future)

            # Détermine la tendance prédite en fonction de la valeur de 'yhat' (prédiction de Prophet).
            trend = forecast['yhat'].apply(lambda x: 'bullish' if x < 30 else ('bearish' if x > 70 else 'neutral'))

            # Calcule la confiance des prédictions en se basant sur l'intervalle de confiance de Prophet ('yhat_upper' et 'yhat_lower').
            confidence = forecast.apply(
                lambda row: (np.clip(row['yhat_upper'], 30, 70) - np.clip(row['yhat_lower'], 30, 70)) /
                            (row['yhat_upper'] - row['yhat_lower']), axis=1)

            # Renvoie la première tendance prédite et la confiance associée en pourcentage.
            return trend[0], 100.0 * confidence[0]
