from .least_confident import LeastConfident
from .entropy import Entropy
from .margin import Margin
from .random import Random
from .coreset import Coreset
from .galaxy import Galaxy
from .batchbald import BatchBALD
from .power_margin import PowerMargin
from .clue import CLUE
from .diana import DiaNA
from .eada import EADA
from .upper import UpperLimit
from .lower import LowerLimit
from .density_aware import Density

METHOD_DICT = {
    'random': Random,
    'entropy': Entropy,
    'lc': LeastConfident,
    'margin': Margin,
    'coreset': Coreset,
    'galaxy': Galaxy,
    'bald': BatchBALD,
    'powermargin': PowerMargin,
    'clue': CLUE,
    'diana': DiaNA,
    'eada': EADA,
    'upper': UpperLimit,
    'lower': LowerLimit,
    'density': Density
}
