from venantvr.business.bot_currency_pair import BotCurrencyPair
from venantvr.dataframes.temporary_columns_manager import TemporaryColumnsManager
from venantvr.logs.logs_utils import logger
from venantvr.tooling.tooling_utils import list_diff
from venantvr.types.types_alias import GateioTimeFrame

from apprentissage.trainers.generic_trainer import GenericTrainer
from apprentissage.bootstrap.types import MlTranformLambda


# La classe KnnTrainer hérite de GenericTrainer et est utilisée pour entraîner et prédire des modèles KNN (K-nearest neighbors).
class KnnTrainer(GenericTrainer):

    def __init__(self, currency_pair: BotCurrencyPair, dataframe, small_timeframe: GateioTimeFrame, middle_timeframe: GateioTimeFrame, model_file: str):
        # Appelle le constructeur de la classe parente GenericTrainer pour initialiser les attributs communs.
        super().__init__(currency_pair, dataframe, small_timeframe, middle_timeframe, model_file)

    # La méthode predict entraîne un modèle KNN pour prédire une tendance à partir des features.
    def predict(self, transform: MlTranformLambda,
                features_columns: list[str], neighbors: int, small_timeframe: GateioTimeFrame, middle_timeframe: GateioTimeFrame):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        """
        Utilise un modèle KNeighborsClassifier pour prédire une tendance et fournir une mesure de confiance.
        Sauvegarde également le modèle pour une réutilisation future.

        :param transform: Fonction lambda qui applique une transformation sur le dataframe.
        :param features_columns: Liste des colonnes qui seront utilisées comme caractéristiques.
        :param neighbors: Nombre de voisins à utiliser pour KNN.
        :param small_timeframe: Timeframe pour les données à court terme.
        :param middle_timeframe: Timeframe pour les données à moyen terme.
        :return: La tendance prédite pour le dernier point de données et la confiance associée.
        """

        # Définit la colonne cible utilisée pour l'entraînement (dans ce cas, un booléen indiquant une tendance).
        target_column = 'target'

        # Vérifie que les colonnes nécessaires pour les petits et moyens timeframes sont présentes dans le DataFrame.
        if not list(self.dataframe.columns) == ['volume', 'close', 'high', 'low', 'open', 'amount', 'closed', 'rsi']:
            if not f'{self.small_timeframe}_volume' in list(self.dataframe.columns):
                logger.error(f'La colonne {self.small_timeframe}_volume est absente du DataFrame')
            if not f'{self.middle_timeframe}_volume' in list(self.dataframe.columns):
                logger.error(f'La colonne {self.middle_timeframe}_volume est absente du DataFrame')

        # Applique la transformation lambda pour générer les données cibles et les caractéristiques supplémentaires.
        transform_result = transform(self.dataframe)
        self.dataframe[target_column] = transform_result.target  # Ajoute la colonne cible au DataFrame.
        extra_features_columns = transform_result.extra_features_columns  # Colonnes supplémentaires générées par la transformation.
        initial_columns = transform_result.initial_columns  # Colonnes initiales du DataFrame avant transformation.

        # Vérifie à nouveau que les colonnes spécifiques aux timeframes sont présentes après la transformation.
        if not list(self.dataframe.columns) == ['volume', 'close', 'high', 'low', 'open', 'amount', 'closed', 'rsi']:
            if not f'{self.small_timeframe}_volume' in list(self.dataframe.columns):
                logger.error(f'La colonne {self.small_timeframe}_volume est absente du DataFrame')
            if not f'{self.middle_timeframe}_volume' in list(self.dataframe.columns):
                logger.error(f'La colonne {self.middle_timeframe}_volume est absente du DataFrame')

        # Utilisation de TemporaryColumnsManager pour assurer que les colonnes temporaires ajoutées seront supprimées après l'opération.
        with (TemporaryColumnsManager(dataframe=self.dataframe,
                                      drop=[target_column] + list_diff(extra_features_columns, initial_columns)) as df):
            # Préparation des caractéristiques (X) et de la cible (Y) pour l'entraînement.
            x = df[features_columns + extra_features_columns].values  # Caractéristiques d'entrée (features).
            y = df[target_column].values  # Valeur cible à prédire (booléen).

            # Normalisation des caractéristiques pour améliorer la performance du KNN.
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)  # Mise à l'échelle des features.

            # Si le modèle n'existe pas encore, l'entraîne et le sauvegarde.
            if self.model_blob is None:
                # Division des données en jeu d'entraînement et jeu de test.
                x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.05, random_state=42)

                # Entraîne un modèle KNN avec le nombre de voisins spécifié.
                self.model_blob = KNeighborsClassifier(n_neighbors=neighbors)
                self.model_blob.fit(x_train, y_train)

                # Sauvegarde du modèle une fois entraîné.
                self.dump_model(model=self.model_blob, file=self.model_file)

            # Utilisation du modèle pour prédire la tendance sur le dernier point de données.
            predicted_probs = self.model_blob.predict_proba(x_scaled[-1].reshape(1, -1))[0]
            predicted_trend = self.model_blob.predict(x_scaled[-1].reshape(1, -1))[0]

            # La confiance est la probabilité associée à la classe prédite.
            confidence_measure = max(predicted_probs)
            return predicted_trend, 100.0 * confidence_measure  # Retourne la tendance et la confiance en pourcentage.
