"""Configuration parameters for the IDX30 Portfolio Optimization project.

Defines constants, hyperparameters, temporal splits, and transaction fee assumptions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Global Random Seed
SEED: int = 29

# Financial and Trading Assumptions
RISK_FREE_RATE: float = 0.06  # 6% annual risk-free rate (Bank Indonesia BI-Rate / 10Y INDOGB)
TRADING_DAYS_PER_YEAR: int = 244  # Typical IDX trading days per year
FEE_BUY: float = 0.0015   # 0.15% broker/exchange fee on buy transactions
FEE_SELL: float = 0.0025  # 0.25% broker/exchange fee on sell transactions (incl. final tax)

# Portfolio Formulation Constraints
LAMBDA_DIV_DEFAULT: float = 0.05  # Diversification penalty weight (Herfindahl-Hirschman Index)
MAX_WEIGHT_DEFAULT: float = 0.20  # Maximum weight per individual asset (20% cap)

# Tuned Hyperparameters for Standard Genetic Algorithm (SGA)
SGA_BEST_PARAMS = {
    "pop_size": 440,
    "generations": 100,
    "pc": 0.5,
    "pm": 0.9,
    "max_weight": MAX_WEIGHT_DEFAULT,
}

# Tuned Hyperparameters for Adaptive Genetic Algorithm (AGA)
AGA_BEST_PARAMS = {
    "pop_size": 500,
    "generations": 88,
    "k1": 0.9,
    "k2": 0.9,
    "k3": 0.1,
    "k4": 0.9,
    "k5": 0.9,
    "k6": 0.3,
    "max_weight": MAX_WEIGHT_DEFAULT,
}

# Temporal Splits for Walk-Forward Optimization (WFO)
SPLIT_DATES = {
    "train_start": "2023-01-02",
    "train_end": "2024-07-31",
    "val_start": "2024-08-01",
    "val_end": "2025-01-24",
    "test_start": "2025-01-30",
    "test_end": "2026-01-26",
}

# Testing Walk-Forward 4-Phase Windows (In-Sample / Out-of-Sample)
WFO_TEST_PHASES: List[Dict[str, str]] = [
    {
        "phase": "Phase 1",
        "is_start": "2023-01-02",
        "is_end": "2025-01-24",
        "oos_start": "2025-01-30",
        "oos_end": "2025-04-30",
    },
    {
        "phase": "Phase 2",
        "is_start": "2023-01-02",
        "is_end": "2025-04-30",
        "oos_start": "2025-05-02",
        "oos_end": "2025-07-31",
    },
    {
        "phase": "Phase 3",
        "is_start": "2023-01-02",
        "is_end": "2025-07-31",
        "oos_start": "2025-08-01",
        "oos_end": "2025-10-31",
    },
    {
        "phase": "Phase 4",
        "is_start": "2023-01-02",
        "is_end": "2025-10-31",
        "oos_start": "2025-11-03",
        "oos_end": "2026-01-26",
    },
]

# Official IDX30 Index Constituent Weights per Phase (Market Benchmark)
IDX30_WEIGHT_MAPS: Dict[str, Dict[str, float]] = {
    "Fase 1": {
        'ADRO': 0.0167, 'AKRA': 0.0060, 'AMRT': 0.0354, 'ANTM': 0.0091, 'ARTO': 0.0069,
        'ASII': 0.0646, 'BBCA': 0.1500, 'BBNI': 0.0493, 'BBRI': 0.1500, 'BMRI': 0.1500,
        'BRPT': 0.0173, 'CPIN': 0.0187, 'EXCL': 0.0071, 'GOTO': 0.0552, 'ICBP': 0.0173,
        'INCO': 0.0056, 'INDF': 0.0224, 'INKP': 0.0102, 'ISAT': 0.0086, 'KLBF': 0.0164,
        'MAPI': 0.0079, 'MBMA': 0.0098, 'MDKA': 0.0143, 'MEDC': 0.0053, 'PGAS': 0.0121,
        'PTBA': 0.0073, 'SMGR': 0.0070, 'TLKM': 0.0880, 'UNTR': 0.0246, 'UNVR': 0.0070
    },
    "Fase 2": {
        'ADRO': 0.0144, 'AKRA': 0.0059, 'AMRT': 0.0272, 'ANTM': 0.0148, 'ASII': 0.0699,
        'BBCA': 0.1500, 'BBNI': 0.0476, 'BBRI': 0.1500, 'BBTN': 0.0044, 'BMRI': 0.1382,
        'BRPT': 0.0159, 'CPIN': 0.0184, 'EXCL': 0.0103, 'GOTO': 0.0565, 'ICBP': 0.0193,
        'INCO': 0.0045, 'INDF': 0.0246, 'INKP': 0.0085, 'ISAT': 0.0074, 'KLBF': 0.0177,
        'MAPI': 0.0091, 'MBMA': 0.0084, 'MDKA': 0.0154, 'MEDC': 0.0054, 'PGAS': 0.0141,
        'PTBA': 0.0084, 'SMGR': 0.0068, 'TLKM': 0.0964, 'UNTR': 0.0244, 'UNVR': 0.0062
    },
    "Fase 3": {
        'ADRO': 0.0130, 'AKRA': 0.0060, 'AMRT': 0.0291, 'ANTM': 0.0187, 'ASII': 0.0661,
        'BBCA': 0.1500, 'BBNI': 0.0439, 'BBRI': 0.1500, 'BMRI': 0.1231, 'BRPT': 0.0413,
        'CPIN': 0.0206, 'EXCL': 0.0091, 'GOTO': 0.0387, 'ICBP': 0.0174, 'INCO': 0.0058,
        'INDF': 0.0259, 'INKP': 0.0096, 'ISAT': 0.0089, 'ITMG': 0.0066, 'JPFA': 0.0059,
        'KLBF': 0.0197, 'MBMA': 0.0121, 'MDKA': 0.0204, 'MEDC': 0.0155, 'PGAS': 0.0123,
        'PTBA': 0.0068, 'SMGR': 0.0059, 'TLKM': 0.0978, 'UNTR': 0.0231, 'UNVR': 0.0064
    },
    "Fase 4": {
        'AADI': 0.0079, 'ADRO': 0.0103, 'AMRT': 0.0265, 'ANTM': 0.0182, 'ASII': 0.0784,
        'BBCA': 0.1500, 'BBNI': 0.0420, 'BBRI': 0.1500, 'BMRI': 0.1068, 'BRPT': 0.0649,
        'CPIN': 0.0199, 'GOTO': 0.0335, 'ICBP': 0.0136, 'INCO': 0.0062, 'INDF': 0.0213,
        'INKP': 0.0109, 'ISAT': 0.0071, 'ITMG': 0.0061, 'JPFA': 0.0086, 'KLBF': 0.0145,
        'MBMA': 0.0116, 'MDKA': 0.0176, 'MEDC': 0.0058, 'PGAS': 0.0126, 'PGEO': 0.0042,
        'PTBA': 0.0060, 'SMGR': 0.0059, 'TLKM': 0.1070, 'UNTR': 0.0244, 'UNVR': 0.0082
    }
}


