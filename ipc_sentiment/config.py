"""Subjects (topics) that move the Mexican IPC index, and the search queries used for each.

Each subject gets one column in the weekly sentiment matrix. Edit this list freely:
add, remove or refine queries. Spanish queries are searched in Google News Mexico,
English queries in Google News US.
"""

SUBJECTS = {
    "banxico_rates": {
        "es": "Banxico tasa de interés",
        "en": "Banxico interest rate",
    },
    "fed_us_rates": {
        "es": "Reserva Federal tasas",
        "en": "Federal Reserve rates",
    },
    "peso_fx": {
        "es": "peso mexicano dólar tipo de cambio",
        "en": "Mexican peso dollar",
    },
    "mx_inflation": {
        "es": "inflación México INEGI",
        "en": "Mexico inflation",
    },
    "us_mx_trade": {
        "es": "aranceles México Estados Unidos T-MEC",
        "en": "Mexico tariffs USMCA",
    },
    "mx_politics": {
        "es": "Sheinbaum reforma judicial inversión",
        "en": "Sheinbaum Mexico reform investors",
    },
    "oil_pemex": {
        "es": "Pemex petróleo",
        "en": "Pemex oil",
    },
    "nearshoring": {
        "es": "nearshoring México inversión extranjera",
        "en": "Mexico nearshoring investment",
    },
    "mx_economy": {
        "es": "economía mexicana PIB crecimiento",
        "en": "Mexico economy GDP",
    },
    "bmv_market": {
        "es": "Bolsa Mexicana de Valores IPC",
        "en": "Mexican stocks IPC index",
    },
}

IPC_TICKER = "^MXX"  # S&P/BMV IPC on Yahoo Finance
