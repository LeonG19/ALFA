from .least_confident import LeastConfident
from .margin import Margin
from .random import Random
from .coreset import Coreset
from .galaxy import Galaxy

METHOD_DICT = {
    'lc': LeastConfident,
    'random': Random,
    'margin': Margin,
    'coreset': Coreset,
    'galaxy': Galaxy
}
