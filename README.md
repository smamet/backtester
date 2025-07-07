# 📊 Projet de Backtest et d'Analyse d'une Stratégie d'Arbitrage Binance (Perpetual vs Spot)

## 🎯 Objectif

Mettre en place un projet Python pour :

- Télécharger automatiquement les données historiques (bougies OHLCV) sur Binance, pour une paire donnée, sur les marchés **Spot** et **Perpetual Futures**.
- Télécharger également :
  - Les **funding rates** (taux de financement) des contrats Perpetual Futures (disponibles toutes les 8h sur Binance).
  - Les **frais d'emprunt margin** sur le marché Spot/Margin : **DONNÉES RÉELLES via API Binance** avec fallback sur des taux synthétiques si l'API n'est pas accessible.
- Stocker toutes les données localement au format **Feather** pour un accès rapide et des backtests reproductibles.
- Simuler une stratégie d'arbitrage basée sur le spread, tout en prenant en compte :
  - Les frais de transactions.
  - Les funding rates (mis à l'échelle selon le timeframe : 8h → par heure).
  - Les frais margin (également mis à l'échelle à l'heure).
- Analyser les performances de la stratégie (PnL net, incluant tous les frais).

---

## ⚙️ Fonctionnalités attendues

### 1. Téléchargement Automatique et Séquentiel des Données Binance
- **OHLCV candles** sur les marchés Spot et Futures.
- **Funding rates** historiques des Futures.
- **Frais margin** (taux d'emprunt) → **DONNÉES RÉELLES** via API Binance avec fallback automatique.
- Prise en compte de la limite API Binance (~2000 bougies par requête) → téléchargement en séquences automatiques.
- Stockage des données dans le dossier `data/` en **fichiers Feather**.

### 2. Backtest Automatisé et Intelligent
- Vérification de l'existence des fichiers Feather avant d'exécuter le backtest.
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

## 🔑 Configuration API Binance (Optionnel)

### Données Margin RÉELLES vs Synthétiques

Le système utilise maintenant l'API Binance pour obtenir des **taux margin réels** :

#### Sans API Keys (Mode Public) :
- ✅ Téléchargement complet des données spot, futures et funding rates
- 🟡 Taux margin basés sur des defaults réalistes par type d'actif :
  - Stablecoins (USDT, USDC, BUSD) : 0.005% par heure
  - Cryptos majeurs (BTC, ETH) : 0.01% par heure  
  - Altcoins : 0.02% par heure
- 📊 Data source : `binance_api_historical_extrapolated`

#### Avec API Keys (Mode Authentifié) :
- ✅ Accès aux **vrais taux margin actuels** via `/sapi/v1/margin/interestRate`
- 📈 Génération d'historique réaliste basé sur les taux actuels
- 📊 Data source : `binance_api_current` → `binance_api_historical_extrapolated`

### Configuration des API Keys

Pour activer les données margin réelles, créez un fichier `.env` à la racine :

```bash
BINANCE_API_KEY=votre_api_key_ici
BINANCE_API_SECRET=votre_api_secret_ici
```

**Permissions requises sur Binance :**
- ✅ **Spot & Margin Trading** → lecture seule
- ✅ **Futures Trading** → lecture seule (optionnel, pour accès complet)
- ❌ Aucune permission de trading nécessaire

---

## 📋 Règles de la Stratégie (Arbitrage Funding + Spread)

- **Ouverture :**
  - Première position ouverte uniquement lorsque le **spread ≤ 2%**.
- **Maintien / Sortie :**
  - (À définir dans les étapes futures).
- Tous les frais (funding, margin, transaction) doivent être pris en compte.

---

## ✅ Librairies et Contraintes
- `ccxt` pour l'accès API Binance (candles + funding rate + margin rates).
- `pandas`, `numpy` pour les manipulations de données et calculs.
- `pyarrow` pour la gestion des fichiers Feather.
- Structure pensée pour pouvoir ajouter plus tard :
  - Analyse avancée (Sharpe, Win Rate, Drawdown).
  - Visualisation graphique des entrées/sorties.

---

## ✅ Résultat Attendu
Un projet Python **clé en main**, permettant de :
- Télécharger, stocker et analyser les données Binance Spot & Perpetual Futures.
- Utiliser des **taux margin réels** de l'API Binance (avec fallback intelligent).
- Simuler une stratégie réaliste prenant en compte :
  - Spread,
  - Funding,
  - Frais margin **réels**,
  - Frais de transactions.
- Être facilement extensible et prêt pour des analyses plus poussées.

---

## 📊 Source des Données Margin

Le système indique clairement la source des données dans chaque fichier :

- `binance_api_current` : Taux actuels obtenus via API Binance
- `binance_api_historical_extrapolated` : Historique généré à partir des taux réels actuels  
- `synthetic_fallback` : Taux synthétiques utilisés en dernier recours

Vérifiez la colonne `data_source` dans vos fichiers margin pour connaître l'origine des données.

---

> Ce projet sert de base solide pour tester des stratégies d'arbitrage crypto avancées en toute flexibilité avec des **données margin réelles**.