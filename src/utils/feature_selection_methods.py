from enum import Enum


class FeatureSelectionMethod(str, Enum):
    ALL = "All Features"
    STRONG = "Strong Features"
    NON_CORRELATED = "Non - Correlated Features"
    STRONG_NON_CORRELATED = "Strong Non - Correlated Features"
    F_STAT_SELECTED = "F-stat Filtered Features"
    RFE = "RFE Selected Features"
    RANDOM_FOREST = "RFC Selected Important Features"
    L1 = "L1 Selected Important Features"
