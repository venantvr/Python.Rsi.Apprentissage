import json

from hyperopt import hp  # Importation de Hyperopt pour la définition des espaces de recherche d'hyperparamètres
from framework.business.bot_currency_pair import BotCurrencyPair
from framework.business.gateio_proxy import GateioProxy
from framework.dataframes.dataframes_utils import set_timestamp_as_index
from framework.logs.logs_utils import logger
from framework.tooling.tooling_utils import convert_gateio_timeframe_to_pandas, file_exists
from framework.types.types_alias import GateioTimeFrame, PandasTimeFrame

from apprentissage.events.hyperopt_event import HyperoptEvent
from apprentissage.bootstrap.machine_learning import MachineLearning


# La classe HyperoptTrainer permet d'optimiser un modèle de trading via la bibliothèque Hyperopt
# en ajustant les hyperparamètres des modèles de machine learning utilisés pour prédire les variations de prix.
class HyperoptTrainer:

    def __init__(self, currency_pair: BotCurrencyPair, gateio_proxy: GateioProxy, small_timeframe: GateioTimeFrame, middle_timeframe: GateioTimeFrame):
        """
        Initialise le formateur Hyperopt avec les paires de devises, le proxy de Gate.io pour récupérer les données de trading,
        et deux timeframes de bougies (small_timeframe et middle_timeframe).

        Paramètres :
        - currency_pair : La paire de devises sur laquelle travailler.
        - gateio_proxy : Le proxy utilisé pour interagir avec l'API de Gate.io.
        - small_timeframe : Timeframe pour les bougies à court terme (ex. : bougies horaires).
        - middle_timeframe : Timeframe pour les bougies à moyen terme (ex. : bougies de 4 heures).
        """
        self.currency_pair = currency_pair  # La paire de devises
        self.gateio_proxy = gateio_proxy  # Le proxy pour interagir avec l'API de Gate.io
        self.small_timeframe = small_timeframe  # Timeframe court (ex. : 1h)
        self.middle_timeframe = middle_timeframe  # Timeframe moyen (ex. : 4h)

        self.small_dataframe = None  # Stocke les données des petites bougies (small timeframe)
        self.middle_dataframe = None  # Stocke les données des bougies moyennes (middle timeframe)

        # Initialisation de la classe MachineLearning pour gérer l'entraînement et la prédiction avec les timeframes.
        self.machine_learning = MachineLearning(small_timeframe=self.small_timeframe,
                                                middle_timeframe=self.middle_timeframe)

    def predict(self):
        """
        Méthode principale qui effectue des prédictions en utilisant différents modèles de machine learning associés à la paire de devises.
        Elle utilise le cadre d'optimisation Hyperopt pour affiner les paramètres du modèle.

        Retourne :
        Un dictionnaire contenant les modèles après optimisation.
        """
        output = dict()  # Stocke les résultats des prédictions

        max_number_of_candles = self.gateio_proxy.get_max_number_of_candles()  # Récupère le nombre maximum de bougies à utiliser

        # Vérifie si les données des petites bougies ont déjà été récupérées, sinon les récupère via l'API Gate.io.
        if self.small_dataframe is None:
            small_dataframe = self.gateio_proxy.fetch_candles(currency_pair=self.currency_pair,
                                                              interval=self.small_timeframe,
                                                              number_of_candles=max_number_of_candles,
                                                              closed=False)
            # Définit l'index du DataFrame en fonction des timestamps (dates).
            self.small_dataframe = set_timestamp_as_index(small_dataframe)

        # Vérifie si les données des bougies moyennes ont déjà été récupérées.
        if self.middle_dataframe is None:
            middle_dataframe = self.gateio_proxy.fetch_candles(currency_pair=self.currency_pair,
                                                               interval=self.middle_timeframe,
                                                               number_of_candles=max_number_of_candles,
                                                               closed=False)
            self.middle_dataframe = set_timestamp_as_index(middle_dataframe)

        # Convertit le GateioTimeFrame en PandasTimeFrame pour pouvoir travailler avec les timeframes Pandas.
        pandas_timefame: PandasTimeFrame = convert_gateio_timeframe_to_pandas(self.small_timeframe)
        # Rééchantillonne les bougies moyennes pour qu'elles correspondent au timeframe plus petit (ex. : de 4h à 1h).
        self.middle_dataframe = self.middle_dataframe.resample(pandas_timefame).ffill()  # Remplit les valeurs manquantes

        # Boucle sur chaque modèle de machine learning associé à la paire de devises
        for transform_key, items in self.currency_pair.machine_learning_models.items():

            # Vérifie si un modèle sauvegardé existe déjà sur le disque, sinon le crée et le sauvegarde.
            if file_exists(items['path']):
                # Si le fichier existe, charge le modèle depuis le fichier JSON.
                with open(items['path'], 'r') as fichier_json:
                    model = json.load(fichier_json)
            else:
                # Si le modèle n'existe pas encore, récupère la méthode associée à transform_key et l'entraîne.
                transform = getattr(self, transform_key, None)
                if transform is not None:
                    model = transform()  # Crée le modèle
                    with open(items['path'], 'w') as fichier_json:
                        json.dump(model, fichier_json, indent=2, sort_keys=True)  # Sauvegarde le modèle dans un fichier JSON
            # Ajoute le modèle au dictionnaire de sortie.
            output[transform_key] = model

        return output  # Retourne les modèles prédits

    def hyperopt_01(self) -> dict:
        """
        Définit l'espace de recherche des hyperparamètres pour Hyperopt et exécute l'optimisation.
        Retourne le meilleur modèle trouvé.
        """
        # Définition de l'espace de recherche pour Hyperopt, spécifiant les hyperparamètres 'entry_mult' et 'exit_mult'.
        space = {
            'entry_mult': hp.uniform('entry_mult', 0.98, 0.995),  # Coefficient d'entrée entre 0.98 et 0.995
            'exit_mult': hp.uniform('exit_mult', 1.005, 1.02),  # Coefficient de sortie entre 1.005 et 1.02
        }
        # Exécute l'optimisation Hyperopt en utilisant la fonction objectif.
        result = self.machine_learning.hyperopt_01(objective=self.__objective_01,
                                                   space=space)

        # Crée un événement pour stocker les résultats de l'optimisation
        output = HyperoptEvent()
        output['entry_mult'] = result['entry_mult']
        output['exit_mult'] = result['exit_mult']

        # Log le résultat de l'optimisation
        logger.log_currency_warning(self.currency_pair, str(self.__objective_01(result, True)))

        return output  # Retourne l'événement contenant les hyperparamètres optimisés

    def __objective_01(self, params, verbose=False):
        """
        Fonction objectif utilisée par Hyperopt pour évaluer les performances des hyperparamètres.
        Elle simule des scénarios de trading basés sur des valeurs calculées pour 'entry_mult' et 'exit_mult'.

        Paramètres :
        - params : Les hyperparamètres d'entrée pour l'optimisation.
        - verbose : Si True, affiche des informations supplémentaires.

        Retourne :
        Le taux de succès négatif pour Hyperopt (car Hyperopt minimise la fonction).
        """
        entry_mult = params['entry_mult']  # Coefficient d'entrée
        exit_mult = params['exit_mult']  # Coefficient de sortie

        # Calcul des prix d'entrée et des cibles de sortie basés sur les coefficients multipliés aux prix d'ouverture.
        self.small_dataframe['entry_price'] = self.small_dataframe['open'] * entry_mult
        self.small_dataframe['exit_target'] = self.small_dataframe['entry_price'] * exit_mult

        successes = 0  # Nombre de transactions réussies
        total = 0  # Nombre total de transactions

        profit = 0.0  # Stocke les profits réalisés
        low_passed = False  # Indicateur pour vérifier si le prix bas est passé en dessous du prix d'entrée

        # Boucle à travers les bougies (lignes du DataFrame) pour évaluer les conditions de trading
        for index, row in self.small_dataframe.iterrows():

            # Si le prix le plus bas de la bougie est en dessous du prix d'entrée, active low_passed.
            if not low_passed and row['low'] < row['entry_price']:
                low_passed = True
                continue  # Continue à la prochaine bougie

            # Si low_passed est vrai et que le prix haut dépasse l'objectif de sortie, calcule le profit.
            if low_passed and row['high'] > row['exit_target'] and (exit_mult - entry_mult) > 0.02:
                profit = (row['exit_target'] - row['entry_price']) / row['entry_price']  # Calcul du profit en pourcentage
                successes += profit  # Ajoute le profit aux succès cumulés
                total += 1
                low_passed = False  # Réinitialise le flag pour la prochaine transaction

        if verbose:
            print(total)  # Affiche le nombre total de transactions (si verbose est activé)

        # Calcul du taux de succès ou de la rentabilité moyenne
        success_rate = successes  # (successes / total) * 100 si on voulait le taux de succès
        return -success_rate  # Retourne le taux de succès négatif pour Hyperopt (car Hyperopt minimise cette fonction)
