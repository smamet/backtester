# 📊 Projet de Backtest et d’Analyse d’une Stratégie d’Arbitrage Binance (Perpetual vs Spot)

## 🎯 Objectif

Mettre en place un projet Python pour :

- Télécharger automatiquement les données historiques (bougies OHLCV) sur Binance, pour une paire donnée, sur les marchés **Spot** et **Perpetual Futures**.
- Télécharger également :
  - Les **funding rates** (taux de financement) des contrats Perpetual Futures (disponibles toutes les 8h sur Binance).
  - Les **frais d’emprunt margin** sur le marché Spot/Margin (fournis à l’heure sur Binance ou ajoutés via CSV si nécessaire).
- Stocker toutes les données localement au format **Feather** pour un accès rapide et des backtests reproductibles.
- Simuler une stratégie d’arbitrage basée sur le spread, tout en prenant en compte :
  - Les frais de transactions.
  - Les funding rates (mis à l’échelle selon le timeframe : 8h → par heure).
  - Les frais margin (également mis à l’échelle à l’heure).
- Analyser les performances de la stratégie (PnL net, incluant tous les frais).

---

## ⚙️ Fonctionnalités attendues

### 1. Téléchargement Automatique et Séquentiel des Données Binance
- **OHLCV candles** sur les marchés Spot et Futures.
- **Funding rates** historiques des Futures.
- **Frais margin** (taux d’emprunt) → soit via API, soit via un fichier CSV à intégrer.
- Prise en compte de la limite API Binance (~1000 bougies par requête) → téléchargement en séquences automatiques.
- Stockage des données dans le dossier `data/` en **fichiers Feather**.

### 2. Backtest Automatisé et Intelligent
- Vérification de l’existence des fichiers Feather avant d’exécuter le backtest.
- Simulation complète prenant en compte :
  - **Entrée uniquement si spread ≤ 2%** pour la toute première position.
  - Intégration automatique des frais :
    - **Funding rate** → ajustement automatique en fonction du timeframe (ex : 1h → funding / 8).
    - **Margin rate** → ajustement automatique en fonction du timeframe.
    - Frais de transaction également inclus.
  - Calcul PnL brut et net.
- Résultat clair : PnL total et aperçu des derniers trades.

### 3. Paramétrage Facile via un Fichier `config.py`
- Paire de trading.
- Timeframe.
- Capital initial.
- Seuils de spread.

---

## 📋 Règles de la Stratégie (Arbitrage Funding + Spread)

- **Ouverture :**
  - Première position ouverte uniquement lorsque le **spread ≤ 2%**.
- **Maintien / Sortie :**
  - (À définir dans les étapes futures).
- Tous les frais (funding, margin, transaction) doivent être pris en compte.

---

## ✅ Librairies et Contraintes
- `ccxt` pour l’accès API Binance (candles + funding rate).
- `pandas`, `numpy` pour les manipulations de données et calculs.
- `pyarrow` pour la gestion des fichiers Feather.
- Structure pensée pour pouvoir ajouter plus tard :
  - Analyse avancée (Sharpe, Win Rate, Drawdown).
  - Visualisation graphique des entrées/sorties.

---

## ✅ Résultat Attendu
Un projet Python **clé en main**, permettant de :
- Télécharger, stocker et analyser les données Binance Spot & Perpetual Futures.
- Simuler une stratégie réaliste prenant en compte :
  - Spread,
  - Funding,
  - Frais margin,
  - Frais de transactions.
- Être facilement extensible et prêt pour des analyses plus poussées.

---

> Ce projet sert de base solide pour tester des stratégies d’arbitrage crypto avancées en toute flexibilité.