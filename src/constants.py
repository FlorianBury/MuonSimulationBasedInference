MUON_MASS_GEV = 0.1056583755 # GeV

X0s = { # m
    "Air": 303.9,
    "H2" : 8.90,
    "H2O": 0.361,
    "C"  : 0.193,
    "Ca" : 0.1042,
    "Si" : 0.09370,
    "Al" : 0.08897,
    "Fe" : 0.01757,
    "Cu" : 0.01436,
    "Pb" : 0.005612,
    "W"  : 0.003504,
    "U"  : 0.003166,
}

LAMBDA_MIN = min([1/X0 for X0 in X0s.values()])
LAMBDA_MAX = max([1/X0 for X0 in X0s.values()])

