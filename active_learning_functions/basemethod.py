from abc import ABC, abstractmethod


class BaseMethod(ABC):
    """
    Base model class providing a common interface for training and evaluation.
    """
    def __init__(self):
        self.name = str(self.__class__).split('.')[-1][:-2]

    @abstractmethod
    def sample(self, x_unlabeled, budget, classifier, x_labeled):
        raise NotImplementedError("Subclasses must implement the 'sample' method.")