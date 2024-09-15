import threading
from typing import Tuple

import numpy as np
from hyperopt import Trials, fmin, tpe
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from framework.business.bot_currency_pair import BotCurrencyPair
from framework.logs.logs_utils import logger
from framework.types.types_alias import GateioTimeFrame

from apprentissage.trainers.knn_trainer import KnnTrainer
from apprentissage.bootstrap.transform_result import TransformResult
from apprentissage.bootstrap.types import MlTranformLambda, HyperoptLambda


class MachineLearning:

    def __init__(self, small_timeframe: GateioTimeFrame, middle_timeframe: GateioTimeFrame):
        self.__lock = threading.Lock()
        self.small_timeframe = small_timeframe
        self.middle_timeframe = middle_timeframe

    def knn_transform_01(self, dataframe: DataFrame) -> TransformResult:
        """
        Transforme les données de trading en features numériques et catégorielles pour préparation au modèle KNN.

        Args:
            dataframe (DataFrame): Le DataFrame contenant les données de marché.

        Returns:
            Tuple[np.ndarray, list[str]]:
                - Un tableau numpy indiquant la cible pour chaque observation.
                - Une liste des noms de colonnes des features transformées.
        """

        # Pour ne pas supprimer les colonnes d'origine
        initial_columns = list(dataframe.columns)

        # Utilise un timeframe spécifique comme référence pour les noms des colonnes
        reference_timeframe = self.small_timeframe

        # Définition des noms des colonnes basées sur le timeframe de référence
        close_column = f'{reference_timeframe}_close'
        high_column = f'{reference_timeframe}_high'
        low_column = f'{reference_timeframe}_low'
        open_column = f'{reference_timeframe}_open'
        volume_column = f'{reference_timeframe}_volume'

        # Définition de nouvelles colonnes pour les caractéristiques calculées
        candle_size = f'{reference_timeframe}_candle_size'
        body_size = f'{reference_timeframe}_body_size'
        direction = f'{reference_timeframe}_direction'

        # Calcul de la taille de la bougie et du corps de la bougie
        dataframe[candle_size] = dataframe[high_column] - dataframe[low_column]
        dataframe[body_size] = abs(dataframe[close_column] - dataframe[open_column])

        # Détermination de la direction de la bougie (haussière ou baissière)
        # Conversion de la condition booléenne en entier (1 pour haussière, 0 pour baissière)
        # noinspection PyUnresolvedReferences
        dataframe[direction] = (dataframe[close_column] > dataframe[open_column]).astype(int)

        # Construction de la cible basée sur plusieurs conditions (retourne 'bullish' ou 'bearish')
        target = np.where(
            (dataframe['condition_01'] & dataframe['condition_02'] &
             dataframe['condition_03'] & dataframe['condition_04'] &
             dataframe['condition_05']),
            'bullish',
            'bearish'
        )

        # Remplacement des infinis générés par des opérations mathématiques précédentes par NaN
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Remplissage des valeurs manquantes pour les features numériques par la moyenne de chaque colonne
        features_numeriques = [candle_size, body_size, volume_column]
        for feature in features_numeriques:
            dataframe.fillna({feature: dataframe[feature].mean()}, inplace=True)
            # Alternative: utiliser .median() pour une robustesse accrue aux valeurs aberrantes

        # Remplissage des valeurs manquantes pour les features catégorielles par le mode
        features_categorielles = [direction]
        for feature in features_categorielles:
            dataframe.fillna({feature: dataframe[feature].mode()[0]}, inplace=True)

        # Liste combinée des noms de colonnes pour toutes les features transformées
        extra_features_columns = features_numeriques + features_categorielles

        # Normalisation des features avec StandardScaler pour préparer les données pour KNN
        scaler = StandardScaler()
        dataframe[extra_features_columns] = scaler.fit_transform(dataframe[extra_features_columns])

        return TransformResult(target, extra_features_columns, initial_columns)

    def knn_transform_02(self, dataframe: DataFrame) -> TransformResult:
        """
        Transforme les données de marché pour la préparation au modèle KNN en utilisant le timeframe moyen.

        Args:
            dataframe (DataFrame): DataFrame contenant les données de marché brutes.

        Returns:
            Tuple[np.ndarray, list[str]]:
                - Tableau numpy indiquant si chaque entrée est classée comme 'bullish' ou 'bearish'.
                - Liste des noms de colonnes des caractéristiques transformées et normalisées.
        """

        # Pour ne pas supprimer les colonnes d'origine
        initial_columns = list(dataframe.columns)

        if not list(dataframe.columns) == ['volume', 'close', 'high', 'low', 'open', 'amount', 'closed', 'rsi']:
            if not f'{self.small_timeframe}_volume' in list(dataframe.columns):
                logger.error(f'La colonne {self.small_timeframe}_volume est absente du DataFrame')
            if not f'{self.middle_timeframe}_volume' in list(dataframe.columns):
                logger.error(f'La colonne {self.middle_timeframe}_volume est absente du DataFrame')

        # Définition du timeframe de référence utilisé pour extraire les caractéristiques
        reference_timeframe = self.middle_timeframe

        # Noms des colonnes basés sur le timeframe de référence
        close_column = f'{reference_timeframe}_close'
        high_column = f'{reference_timeframe}_high'
        low_column = f'{reference_timeframe}_low'
        open_column = f'{reference_timeframe}_open'
        volume_column = f'{reference_timeframe}_volume'

        # Calcul de nouvelles caractéristiques basées sur les dimensions des bougies
        candle_size = f'{reference_timeframe}_candle_size'
        body_size = f'{reference_timeframe}_body_size'
        direction = f'{reference_timeframe}_direction'

        # Calcul de la taille totale et de la taille du corps de la bougie
        dataframe[candle_size] = dataframe[high_column] - dataframe[low_column]
        dataframe[body_size] = abs(dataframe[close_column] - dataframe[open_column])

        # Détermination de la direction de la bougie (1 pour haussière, 0 pour baissière)
        # noinspection PyUnresolvedReferences
        dataframe[direction] = (dataframe[close_column] > dataframe[open_column]).astype(int)

        # Construction de la cible en fonction de plusieurs conditions prédéfinies
        target = np.where(
            (dataframe['condition_01'] & dataframe['condition_02'] &
             dataframe['condition_03'] & dataframe['condition_04'] &
             dataframe['condition_05']),
            'bullish',
            'bearish'
        )

        # Remplacement des valeurs infinies résultant de divisions par zéro par NaN
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Remplissage des valeurs manquantes pour les caractéristiques numériques avec la moyenne
        features_numeriques = [candle_size, body_size, volume_column]
        for feature in features_numeriques:
            dataframe.fillna({feature: dataframe[feature].mean()}, inplace=True)

        # Remplissage des valeurs manquantes pour les caractéristiques catégorielles avec le mode
        features_categorielles = [direction]
        for feature in features_categorielles:
            dataframe.fillna({feature: dataframe[feature].mode()[0]}, inplace=True)

        # Liste de toutes les caractéristiques traitées à normaliser
        extra_features_columns = features_numeriques + features_categorielles

        # Normalisation des caractéristiques à l'aide d'un StandardScaler
        scaler = StandardScaler()
        dataframe[extra_features_columns] = scaler.fit_transform(dataframe[extra_features_columns])

        return TransformResult(target, extra_features_columns, initial_columns)

    def knn_transform_03(self, dataframe: DataFrame) -> TransformResult:
        """
        Transforme le DataFrame pour créer des features et cibler les observations selon des critères de mouvement des prix spécifiques.

        Cette fonction calcule si les prix atteindront une augmentation ou une diminution cible dans un nombre donné de futures observations.
        Elle génère des features basées sur les dimensions et la dynamique des bougies, et prépare les données pour un modèle prédictif.

        Args:
            dataframe (DataFrame): Le DataFrame contenant les données de marché.

        Returns:
            Tuple[np.ndarray, list[str]]:
                - Un tableau numpy indiquant la position du premier échantillon futur qui remplit les conditions de mouvement de prix.
                - Une liste de noms de colonnes pour les features numériques et catégorielles transformées.
        """

        # Pour ne pas supprimer les colonnes d'origine
        initial_columns = list(dataframe.columns)

        # Définition des seuils de mouvement des prix en pourcentage
        increase_percentage_target = 1.0  # Seuil d'augmentation des prix cible
        decrease_percentage_target = -2.0  # Seuil de diminution des prix cible
        look_ahead_samples = 10  # Nombre d'échantillons à examiner dans le futur

        # Utilisation du timeframe moyen comme référence pour les noms des colonnes
        reference_timeframe = self.middle_timeframe
        close_column = f'{reference_timeframe}_close'
        high_column = f'{reference_timeframe}_high'
        low_column = f'{reference_timeframe}_low'
        open_column = f'{reference_timeframe}_open'
        # volume_column = f'{reference_timeframe}_volume'

        # Calcul de la taille des bougies et du corps des bougies pour déterminer la volatilité
        candle_size = f'{reference_timeframe}_candle_size'
        body_size = f'{reference_timeframe}_body_size'
        direction = f'{reference_timeframe}_direction'

        # Calcul des variations de prix maximum et minimum sur les échantillons à venir
        max_price_change = (dataframe[high_column].rolling(window=look_ahead_samples, min_periods=1).max().shift(-look_ahead_samples) / dataframe[
            high_column] - 1.0) * 100.0
        min_price_change = (dataframe[low_column].rolling(window=look_ahead_samples, min_periods=1).min().shift(-look_ahead_samples) / dataframe[
            low_column] - 1.0) * 100.0

        # Définition de la cible basée sur si les conditions d'augmentation ou de diminution sont remplies
        conditions_met = (max_price_change >= increase_percentage_target) & (min_price_change <= decrease_percentage_target)
        target = np.where(conditions_met, 1, 0)  # 1 pour vrai, 0 pour faux

        # Préparation des features pour le modèle, y compris normalisation
        dataframe[candle_size] = dataframe[high_column] - dataframe[low_column]
        dataframe[body_size] = abs(dataframe[close_column] - dataframe[open_column])
        # noinspection PyUnresolvedReferences
        dataframe[direction] = (dataframe[close_column] > dataframe[open_column]).astype(int)

        # Remplissage des valeurs manquantes et normalisation des features
        features_numeriques = [candle_size, body_size]
        features_categorielles = [direction]
        extra_features_columns = features_numeriques + features_categorielles

        # Normalisation des données avec un StandardScaler pour garantir une mise à l'échelle appropriée pour le KNN
        scaler = StandardScaler()
        dataframe[extra_features_columns] = scaler.fit_transform(dataframe[extra_features_columns])

        return TransformResult(target, extra_features_columns, initial_columns)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def prophet_transform(self, dataframe: DataFrame) -> TransformResult:
        return TransformResult(np.array([], dtype=object), [], [])

    # noinspection PyMethodMayBeStatic
    def setup_and_predict(self, currency_pair: BotCurrencyPair, dataframe: DataFrame, knn_model_file: str,
                          knn_transform: MlTranformLambda, features_columns: list[str], neighbors: int, small_timeframe: GateioTimeFrame,
                          middle_timeframe: GateioTimeFrame):
        """
        Configure et exécute une prédiction en utilisant un modèle KNN pré-entrainé.

        Args:
            dataframe (DataFrame): Le DataFrame contenant les données sur lesquelles effectuer la prédiction.
            currency_pair (BotCurrencyPair): L'objet contenant les informations de la paire de devises à analyser.
            knn_model_file (str): Chemin vers le fichier du modèle KNN sauvegardé.
            knn_transform (Callable): Fonction qui transforme le DataFrame en un format adapté pour la prédiction du modèle KNN.
            features_columns (list[str]): Liste des noms des colonnes qui doivent être utilisées comme caractéristiques pour la prédiction.
            neighbors (int): Nombre de voisins à considérer dans l'algorithme KNN.
            small_timeframe (str): La granularité temporelle petite pour la prédiction, par exemple '1m' pour une minute.
            middle_timeframe (str): La granularité temporelle moyenne pour la prédiction, par exemple '5m' pour cinq minutes.

        Returns:
            dict: Un dictionnaire contenant la 'tendance' prédite et le 'niveau de confiance' associé.
        """

        if not list(dataframe.columns) == ['volume', 'close', 'high', 'low', 'open', 'amount', 'closed', 'rsi']:
            if not f'{self.small_timeframe}_volume' in list(dataframe.columns):
                logger.error(f'La colonne {self.small_timeframe}_volume est absente du DataFrame')
            if not f'{self.middle_timeframe}_volume' in list(dataframe.columns):
                logger.error(f'La colonne {self.middle_timeframe}_volume est absente du DataFrame')

        # Initialisation de l'objet KnnTrainer avec les données et configurations nécessaires
        knn_trainer = KnnTrainer(currency_pair=currency_pair, dataframe=dataframe, small_timeframe=self.small_timeframe, middle_timeframe=self.middle_timeframe,
                                 model_file=knn_model_file)

        if not list(dataframe.columns) == ['volume', 'close', 'high', 'low', 'open', 'amount', 'closed', 'rsi']:
            if not f'{self.small_timeframe}_volume' in list(dataframe.columns):
                logger.error(f'La colonne {self.small_timeframe}_volume est absente du DataFrame')
            if not f'{self.middle_timeframe}_volume' in list(dataframe.columns):
                logger.error(f'La colonne {self.middle_timeframe}_volume est absente du DataFrame')

        # Prédiction à l'aide du modèle KNN en utilisant la transformation des données, les colonnes de caractéristiques
        # spécifiées et les autres paramètres pertinents
        trend, confidence = knn_trainer.predict(
            transform=knn_transform,
            features_columns=features_columns,
            neighbors=neighbors,
            small_timeframe=small_timeframe,
            middle_timeframe=middle_timeframe
        )

        # Retourne un dictionnaire contenant les résultats de la prédiction
        return {'trend': trend, 'confidence': confidence}

    # noinspection PyMethodMayBeStatic
    def hyperopt_01(self, objective: HyperoptLambda, space: dict) -> dict:
        with self.__lock:
            # Exécution de l'optimisation
            trials = Trials()
            best = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=100,
                        trials=trials)
            logger.warning(f'Best parameters : {best}')
            return best
