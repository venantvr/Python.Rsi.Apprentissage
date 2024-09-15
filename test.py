from framework.types.types_alias import GateioTimeFrame

from apprentissage.bootstrap.machine_learning import MachineLearning

machine_learning = MachineLearning(GateioTimeFrame('1h'), GateioTimeFrame('12h'))
