from typing import Any
from joblib import load, dump
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from framework.business.bot_currency_pair import BotCurrencyPair
from framework.business.gateio_proxy import GateioProxy
from framework.logs.logs_utils import logger
from framework.tooling.tooling_utils import file_exists
from framework.types.types_alias import GateioTimeFrame


# La classe GenericTrainer fournit une structure générale pour entraîner différents modèles de machine learning
# sur des paires de devises et des données associées. Elle gère la sauvegarde et le chargement des modèles.
class GenericTrainer:

    def __init__(self, currency_pair: BotCurrencyPair, dataframe, small_timeframe: GateioTimeFrame, middle_timeframe: GateioTimeFrame, model_file: str):
        """
        Initialise un modèle générique pour entraîner et prédire en fonction des données d'une paire de devises.

        Paramètres :
        - currency_pair : Représente la paire de devises concernée par l'entraînement.
        - dataframe : DataFrame contenant les données sur lesquelles le modèle sera entraîné.
        - small_timeframe : Timeframe plus petit (ex : données horaires) pour les bougies.
        - middle_timeframe : Timeframe intermédiaire pour les bougies.
        - model_file : Le fichier où le modèle sera sauvegardé ou chargé.
        """
        self.dataframe = dataframe  # Le DataFrame des données à utiliser pour l'entraînement. Peut inclure des colonnes comme prix, volume, RSI, etc.

        # Timeframes associés aux données (ex : timeframe court pour des bougies d'une heure).
        self.small_timeframe = small_timeframe
        self.middle_timeframe = middle_timeframe

        # Vérification des colonnes dans le DataFrame pour s'assurer que les données nécessaires sont présentes.
        if not list(dataframe.columns) == ['volume', 'close', 'high', 'low', 'open', 'amount', 'closed', 'rsi']:
            # Si certaines colonnes nécessaires au modèle ne sont pas présentes, des erreurs sont enregistrées.
            if not f'{self.small_timeframe}_volume' in list(dataframe.columns):
                logger.error(f'La colonne {self.small_timeframe}_volume est absente du DataFrame')
            if not f'{self.middle_timeframe}_volume' in list(dataframe.columns):
                logger.error(f'La colonne {self.middle_timeframe}_volume est absente du DataFrame')

        self.currency_pair = currency_pair  # La paire de devises à laquelle le modèle est associé.
        self.model_file = model_file  # Le chemin du fichier où le modèle est ou sera sauvegardé.

        # Obtention d'une instance de GateioProxy pour interagir avec l'API du marché de Gate.io.
        self.gateio_proxy = GateioProxy.get()

        # Si le fichier du modèle existe déjà, charge le modèle depuis ce fichier.
        if file_exists(self.model_file):
            self.model_blob = load(self.model_file)
        else:
            # Sinon, initialise un modèle vide qui sera entraîné plus tard.
            self.model_blob = None

    # Méthode pour sauvegarder un modèle une fois qu'il a été entraîné.
    # Elle prend en charge différents types de modèles (régression, KNN, Prophet, etc.).
    def dump_model(self, model: LinearRegression | KNeighborsClassifier | Prophet | Any, file):
        """
        Sauvegarde un modèle entraîné dans un fichier.

        Paramètres :
        - model : Le modèle qui a été entraîné (peut être de type régression, KNN, Prophet, etc.).
        - file : Le fichier dans lequel le modèle sera sauvegardé.
        """
        # Si le fichier n'existe pas encore, on log une alerte et on sauvegarde le modèle dans le fichier spécifié.
        if not file_exists(file):
            logger.log_currency_warning(currency_pair=self.currency_pair,
                                        message=f'Sauvegarde de {file}')
            dump(model, file)  # Sauvegarde du modèle avec la bibliothèque joblib.
