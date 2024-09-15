from typing import List
import numpy as np


class TransformResult:
    def __init__(self, target: np.ndarray, extra_features_columns: List[str], initial_columns: List[str]):
        """
        Classe pour encapsuler le résultat d'une transformation sur les données d'entraînement.

        :param target: La cible après transformation (sous forme d'un tableau NumPy).
        :param extra_features_columns: Les colonnes supplémentaires générées par la transformation.
        :param initial_columns: Les colonnes initiales avant transformation.
        """
        # Définition des attributs de classe avec leurs types respectifs
        self.target: np.ndarray = target
        self.extra_features_columns: List[str] = extra_features_columns
        self.initial_columns: List[str] = initial_columns
