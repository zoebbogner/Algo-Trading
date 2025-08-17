"""
Comprehensive Cryptocurrency Symbols Configuration

This module provides an extensive list of cryptocurrency symbols organized by category,
along with metadata for correlation analysis and trading strategies.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CryptoCategory(Enum):
    """Cryptocurrency categories for analysis"""
    MAJOR = "major"
    DEFI = "defi"
    LAYER1 = "layer1"
    GAMING = "gaming"
    AI_BIGDATA = "ai_bigdata"
    PRIVACY = "privacy"
    EXCHANGE = "exchange"
    STABLECOIN = "stablecoin"
    MEME = "meme"
    EMERGING = "emerging"
    YIELD_FARMING = "yield_farming"
    METAVERSE = "metaverse"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class CryptoSymbol:
    """Cryptocurrency symbol with metadata"""
    symbol: str
    name: str
    category: CryptoCategory
    market_cap_rank: int
    bitcoin_correlation: float  # Historical correlation with Bitcoin
    volatility_profile: str     # High, Medium, Low
    trading_volume: str         # High, Medium, Low
    launch_date: str            # Approximate launch date
    description: str            # Brief description
    use_case: str              # Primary use case


# Comprehensive cryptocurrency symbols with metadata
CRYPTO_SYMBOLS = {
    # Major cryptocurrencies (Top 20 by market cap)
    'BTC/USDT': CryptoSymbol(
        symbol='BTC/USDT',
        name='Bitcoin',
        category=CryptoCategory.MAJOR,
        market_cap_rank=1,
        bitcoin_correlation=1.0,
        volatility_profile='High',
        trading_volume='Very High',
        launch_date='2009-01-03',
        description='First and largest cryptocurrency, digital gold',
        use_case='Store of value, payments, reserve asset'
    ),
    
    'ETH/USDT': CryptoSymbol(
        symbol='ETH/USDT',
        name='Ethereum',
        category=CryptoCategory.MAJOR,
        market_cap_rank=2,
        bitcoin_correlation=0.85,
        volatility_profile='High',
        trading_volume='Very High',
        launch_date='2015-07-30',
        description='Smart contract platform, decentralized applications',
        use_case='Smart contracts, DeFi, NFTs, dApps'
    ),
    
    'BNB/USDT': CryptoSymbol(
        symbol='BNB/USDT',
        name='Binance Coin',
        category=CryptoCategory.EXCHANGE,
        market_cap_rank=3,
        bitcoin_correlation=0.75,
        volatility_profile='Medium',
        trading_volume='High',
        launch_date='2017-07-25',
        description='Binance exchange utility token',
        use_case='Exchange fees, staking, DeFi'
    ),
    
    'ADA/USDT': CryptoSymbol(
        symbol='ADA/USDT',
        name='Cardano',
        category=CryptoCategory.LAYER1,
        market_cap_rank=4,
        bitcoin_correlation=0.70,
        volatility_profile='High',
        trading_volume='High',
        launch_date='2017-09-29',
        description='Proof-of-stake blockchain platform',
        use_case='Smart contracts, DeFi, identity management'
    ),
    
    'SOL/USDT': CryptoSymbol(
        symbol='SOL/USDT',
        name='Solana',
        category=CryptoCategory.LAYER1,
        market_cap_rank=5,
        bitcoin_correlation=0.65,
        volatility_profile='Very High',
        trading_volume='High',
        launch_date='2020-03-16',
        description='High-performance blockchain platform',
        use_case='DeFi, NFTs, gaming, high-speed transactions'
    ),
    
    'XRP/USDT': CryptoSymbol(
        symbol='XRP/USDT',
        name='Ripple',
        category=CryptoCategory.INFRASTRUCTURE,
        market_cap_rank=6,
        bitcoin_correlation=0.60,
        volatility_profile='Medium',
        trading_volume='High',
        launch_date='2012-09-24',
        description='Cross-border payment protocol',
        use_case='International money transfers, banking'
    ),
    
    'DOT/USDT': CryptoSymbol(
        symbol='DOT/USDT',
        name='Polkadot',
        category=CryptoCategory.INFRASTRUCTURE,
        market_cap_rank=7,
        bitcoin_correlation=0.75,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2020-05-26',
        description='Multi-chain network protocol',
        use_case='Interoperability, parachains, cross-chain'
    ),
    
    'DOGE/USDT': CryptoSymbol(
        symbol='DOGE/USDT',
        name='Dogecoin',
        category=CryptoCategory.MEME,
        market_cap_rank=8,
        bitcoin_correlation=0.55,
        volatility_profile='Very High',
        trading_volume='High',
        launch_date='2013-12-06',
        description='Meme-based cryptocurrency',
        use_case='Tipping, community, social media'
    ),
    
    'AVAX/USDT': CryptoSymbol(
        symbol='AVAX/USDT',
        name='Avalanche',
        category=CryptoCategory.LAYER1,
        market_cap_rank=9,
        bitcoin_correlation=0.70,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2020-07-22',
        description='High-performance blockchain platform',
        use_case='DeFi, smart contracts, subnets'
    ),
    
    'MATIC/USDT': CryptoSymbol(
        symbol='MATIC/USDT',
        name='Polygon',
        category=CryptoCategory.INFRASTRUCTURE,
        market_cap_rank=10,
        bitcoin_correlation=0.80,
        volatility_profile='Medium',
        trading_volume='High',
        launch_date='2019-04-26',
        description='Ethereum scaling solution',
        use_case='Layer 2 scaling, DeFi, low fees'
    ),
    
    # DeFi tokens
    'UNI/USDT': CryptoSymbol(
        symbol='UNI/USDT',
        name='Uniswap',
        category=CryptoCategory.DEFI,
        market_cap_rank=11,
        bitcoin_correlation=0.75,
        volatility_profile='High',
        trading_volume='High',
        launch_date='2020-09-17',
        description='Decentralized exchange protocol',
        use_case='DEX governance, liquidity provision'
    ),
    
    'LINK/USDT': CryptoSymbol(
        symbol='LINK/USDT',
        name='Chainlink',
        category=CryptoCategory.INFRASTRUCTURE,
        market_cap_rank=12,
        bitcoin_correlation=0.70,
        volatility_profile='High',
        trading_volume='High',
        launch_date='2017-09-19',
        description='Decentralized oracle network',
        use_case='Price feeds, smart contract data'
    ),
    
    'AAVE/USDT': CryptoSymbol(
        symbol='AAVE/USDT',
        name='Aave',
        category=CryptoCategory.DEFI,
        market_cap_rank=13,
        bitcoin_correlation=0.80,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2017-11-07',
        description='Decentralized lending protocol',
        use_case='Lending, borrowing, yield farming'
    ),
    
    'COMP/USDT': CryptoSymbol(
        symbol='COMP/USDT',
        name='Compound',
        category=CryptoCategory.DEFI,
        market_cap_rank=14,
        bitcoin_correlation=0.75,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2018-09-24',
        description='Decentralized lending protocol',
        use_case='Lending, borrowing, governance'
    ),
    
    'SUSHI/USDT': CryptoSymbol(
        symbol='SUSHI/USDT',
        name='SushiSwap',
        category=CryptoCategory.DEFI,
        market_cap_rank=15,
        bitcoin_correlation=0.80,
        volatility_profile='Very High',
        trading_volume='Medium',
        launch_date='2020-08-28',
        description='Decentralized exchange protocol',
        use_case='DEX, yield farming, governance'
    ),
    
    # Layer 1 alternatives
    'ATOM/USDT': CryptoSymbol(
        symbol='ATOM/USDT',
        name='Cosmos',
        category=CryptoCategory.LAYER1,
        market_cap_rank=16,
        bitcoin_correlation=0.65,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2019-03-14',
        description='Interoperable blockchain network',
        use_case='Cross-chain communication, DeFi'
    ),
    
    'NEAR/USDT': CryptoSymbol(
        symbol='NEAR/USDT',
        name='NEAR Protocol',
        category=CryptoCategory.LAYER1,
        market_cap_rank=17,
        bitcoin_correlation=0.60,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2020-04-22',
        description='Sharded proof-of-stake blockchain',
        use_case='DeFi, NFTs, developer platform'
    ),
    
    'FTM/USDT': CryptoSymbol(
        symbol='FTM/USDT',
        name='Fantom',
        category=CryptoCategory.LAYER1,
        market_cap_rank=18,
        bitcoin_correlation=0.70,
        volatility_profile='Very High',
        trading_volume='Medium',
        launch_date='2018-12-20',
        description='High-performance smart contract platform',
        use_case='DeFi, smart contracts, gaming'
    ),
    
    'ALGO/USDT': CryptoSymbol(
        symbol='ALGO/USDT',
        name='Algorand',
        category=CryptoCategory.LAYER1,
        market_cap_rank=19,
        bitcoin_correlation=0.55,
        volatility_profile='Medium',
        trading_volume='Medium',
        launch_date='2019-06-19',
        description='Pure proof-of-stake blockchain',
        use_case='DeFi, smart contracts, CBDCs'
    ),
    
    'ICP/USDT': CryptoSymbol(
        symbol='ICP/USDT',
        name='Internet Computer',
        category=CryptoCategory.INFRASTRUCTURE,
        market_cap_rank=20,
        bitcoin_correlation=0.50,
        volatility_profile='Very High',
        trading_volume='Medium',
        launch_date='2021-05-10',
        description='Decentralized internet infrastructure',
        use_case='Web services, cloud computing, DeFi'
    ),
    
    # Gaming and Metaverse
    'AXS/USDT': CryptoSymbol(
        symbol='AXS/USDT',
        name='Axie Infinity',
        category=CryptoCategory.GAMING,
        market_cap_rank=21,
        bitcoin_correlation=0.65,
        volatility_profile='Very High',
        trading_volume='Medium',
        launch_date='2020-11-04',
        description='Play-to-earn gaming platform',
        use_case='Gaming, NFTs, yield farming'
    ),
    
    'MANA/USDT': CryptoSymbol(
        symbol='MANA/USDT',
        name='Decentraland',
        category=CryptoCategory.METAVERSE,
        market_cap_rank=22,
        bitcoin_correlation=0.70,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2017-12-20',
        description='Virtual reality platform',
        use_case='Virtual real estate, gaming, social'
    ),
    
    'SAND/USDT': CryptoSymbol(
        symbol='SAND/USDT',
        name='The Sandbox',
        category=CryptoCategory.METAVERSE,
        market_cap_rank=23,
        bitcoin_correlation=0.75,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2011-08-15',
        description='Virtual gaming world',
        use_case='Gaming, virtual real estate, NFTs'
    ),
    
    'ENJ/USDT': CryptoSymbol(
        symbol='ENJ/USDT',
        name='Enjin Coin',
        category=CryptoCategory.GAMING,
        market_cap_rank=24,
        bitcoin_correlation=0.70,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2017-11-01',
        description='Gaming cryptocurrency platform',
        use_case='Gaming, NFTs, virtual items'
    ),
    
    'CHZ/USDT': CryptoSymbol(
        symbol='CHZ/USDT',
        name='Chiliz',
        category=CryptoCategory.GAMING,
        market_cap_rank=25,
        bitcoin_correlation=0.60,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2018-07-01',
        description='Sports and entertainment platform',
        use_case='Fan tokens, sports, entertainment'
    ),
    
    # AI and Big Data
    'OCEAN/USDT': CryptoSymbol(
        symbol='OCEAN/USDT',
        name='Ocean Protocol',
        category=CryptoCategory.AI_BIGDATA,
        market_cap_rank=26,
        bitcoin_correlation=0.65,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2017-11-30',
        description='Data exchange protocol',
        use_case='Data sharing, AI training, privacy'
    ),
    
    'FET/USDT': CryptoSymbol(
        symbol='FET/USDT',
        name='Fetch.ai',
        category=CryptoCategory.AI_BIGDATA,
        market_cap_rank=27,
        bitcoin_correlation=0.60,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2019-03-01',
        description='AI-powered blockchain platform',
        use_case='AI agents, automation, DeFi'
    ),
    
    'AGIX/USDT': CryptoSymbol(
        symbol='AGIX/USDT',
        name='SingularityNET',
        category=CryptoCategory.AI_BIGDATA,
        market_cap_rank=28,
        bitcoin_correlation=0.55,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2017-12-19',
        description='AI marketplace platform',
        use_case='AI services, machine learning, DeFi'
    ),
    
    'RLC/USDT': CryptoSymbol(
        symbol='RLC/USDT',
        name='iExec RLC',
        category=CryptoCategory.AI_BIGDATA,
        market_cap_rank=29,
        bitcoin_correlation=0.65,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2017-04-19',
        description='Cloud computing marketplace',
        use_case='Computing resources, AI, big data'
    ),
    
    'GRT/USDT': CryptoSymbol(
        symbol='GRT/USDT',
        name='The Graph',
        category=CryptoCategory.INFRASTRUCTURE,
        market_cap_rank=30,
        bitcoin_correlation=0.70,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2020-12-17',
        description='Blockchain data indexing protocol',
        use_case='Data querying, DeFi, analytics'
    ),
    
    # Privacy and Security
    'XMR/USDT': CryptoSymbol(
        symbol='XMR/USDT',
        name='Monero',
        category=CryptoCategory.PRIVACY,
        market_cap_rank=31,
        bitcoin_correlation=0.45,
        volatility_profile='Medium',
        trading_volume='Medium',
        launch_date='2014-04-18',
        description='Privacy-focused cryptocurrency',
        use_case='Private transactions, fungibility'
    ),
    
    'ZEC/USDT': CryptoSymbol(
        symbol='ZEC/USDT',
        name='Zcash',
        category=CryptoCategory.PRIVACY,
        market_cap_rank=32,
        bitcoin_correlation=0.50,
        volatility_profile='Medium',
        trading_volume='Low',
        launch_date='2016-10-28',
        description='Privacy-preserving cryptocurrency',
        use_case='Private transactions, selective disclosure'
    ),
    
    'DASH/USDT': CryptoSymbol(
        symbol='DASH/USDT',
        name='Dash',
        category=CryptoCategory.PRIVACY,
        market_cap_rank=33,
        bitcoin_correlation=0.55,
        volatility_profile='Medium',
        trading_volume='Low',
        launch_date='2014-01-18',
        description='Digital cash with privacy features',
        use_case='Payments, privacy, instant transactions'
    ),
    
    'LTC/USDT': CryptoSymbol(
        symbol='LTC/USDT',
        name='Litecoin',
        category=CryptoCategory.MAJOR,
        market_cap_rank=34,
        bitcoin_correlation=0.80,
        volatility_profile='Medium',
        trading_volume='High',
        launch_date='2011-10-07',
        description='Digital silver to Bitcoin\'s gold',
        use_case='Payments, faster transactions, testing'
    ),
    
    'BCH/USDT': CryptoSymbol(
        symbol='BCH/USDT',
        name='Bitcoin Cash',
        category=CryptoCategory.MAJOR,
        market_cap_rank=35,
        bitcoin_correlation=0.75,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2017-08-01',
        description='Bitcoin fork for larger blocks',
        use_case='Payments, merchant adoption, scaling'
    ),
    
    # Exchange tokens
    'OKB/USDT': CryptoSymbol(
        symbol='OKB/USDT',
        name='OKB',
        category=CryptoCategory.EXCHANGE,
        market_cap_rank=36,
        bitcoin_correlation=0.70,
        volatility_profile='Medium',
        trading_volume='Medium',
        launch_date='2019-01-31',
        description='OKX exchange utility token',
        use_case='Exchange fees, staking, DeFi'
    ),
    
    'HT/USDT': CryptoSymbol(
        symbol='HT/USDT',
        name='Huobi Token',
        category=CryptoCategory.EXCHANGE,
        market_cap_rank=37,
        bitcoin_correlation=0.65,
        volatility_profile='Medium',
        trading_volume='Medium',
        launch_date='2018-01-24',
        description='Huobi exchange utility token',
        use_case='Exchange fees, staking, DeFi'
    ),
    
    'KCS/USDT': CryptoSymbol(
        symbol='KCS/USDT',
        name='KuCoin Token',
        category=CryptoCategory.EXCHANGE,
        market_cap_rank=38,
        bitcoin_correlation=0.60,
        volatility_profile='Medium',
        trading_volume='Medium',
        launch_date='2017-09-26',
        description='KuCoin exchange utility token',
        use_case='Exchange fees, staking, DeFi'
    ),
    
    'CRO/USDT': CryptoSymbol(
        symbol='CRO/USDT',
        name='Cronos',
        category=CryptoCategory.EXCHANGE,
        market_cap_rank=39,
        bitcoin_correlation=0.65,
        volatility_profile='Medium',
        trading_volume='Medium',
        launch_date='2018-12-14',
        description='Crypto.com exchange utility token',
        use_case='Exchange fees, staking, DeFi'
    ),
    
    'BTT/USDT': CryptoSymbol(
        symbol='BTT/USDT',
        name='BitTorrent',
        category=CryptoCategory.INFRASTRUCTURE,
        market_cap_rank=40,
        bitcoin_correlation=0.55,
        volatility_profile='High',
        trading_volume='Medium',
        launch_date='2019-01-28',
        description='Decentralized file sharing protocol',
        use_case='File sharing, content distribution'
    ),
    
    # Stablecoins and yield farming
    'USDC/USDT': CryptoSymbol(
        symbol='USDC/USDT',
        name='USD Coin',
        category=CryptoCategory.STABLECOIN,
        market_cap_rank=41,
        bitcoin_correlation=0.05,
        volatility_profile='Very Low',
        trading_volume='Very High',
        launch_date='2018-09-26',
        description='USD-backed stablecoin',
        use_case='Stable value, DeFi, payments'
    ),
    
    'DAI/USDT': CryptoSymbol(
        symbol='DAI/USDT',
        name='Dai',
        category=CryptoCategory.STABLECOIN,
        market_cap_rank=42,
        bitcoin_correlation=0.10,
        volatility_profile='Low',
        trading_volume='High',
        launch_date='2017-12-18',
        description='Decentralized stablecoin',
        use_case='Stable value, DeFi, collateral'
    ),
    
    'BUSD/USDT': CryptoSymbol(
        symbol='BUSD/USDT',
        name='Binance USD',
        category=CryptoCategory.STABLECOIN,
        market_cap_rank=43,
        bitcoin_correlation=0.05,
        volatility_profile='Very Low',
        trading_volume='High',
        launch_date='2019-09-05',
        description='Binance USD-backed stablecoin',
        use_case='Stable value, trading, DeFi'
    ),
    
    'TUSD/USDT': CryptoSymbol(
        symbol='TUSD/USDT',
        name='TrueUSD',
        category=CryptoCategory.STABLECOIN,
        market_cap_rank=44,
        bitcoin_correlation=0.05,
        volatility_profile='Very Low',
        trading_volume='Medium',
        launch_date='2018-03-06',
        description='USD-backed stablecoin',
        use_case='Stable value, trading, DeFi'
    ),
    
    'FRAX/USDT': CryptoSymbol(
        symbol='FRAX/USDT',
        name='Frax',
        category=CryptoCategory.STABLECOIN,
        market_cap_rank=45,
        bitcoin_correlation=0.10,
        volatility_profile='Low',
        trading_volume='Medium',
        launch_date='2020-12-21',
        description='Fractional-algorithmic stablecoin',
        use_case='Stable value, DeFi, yield farming'
    ),
    
    # Meme coins and trending
    'SHIB/USDT': CryptoSymbol(
        symbol='SHIB/USDT',
        name='Shiba Inu',
        category=CryptoCategory.MEME,
        market_cap_rank=46,
        bitcoin_correlation=0.60,
        volatility_profile='Very High',
        trading_volume='High',
        launch_date='2020-08-01',
        description='Dogecoin-inspired meme token',
        use_case='Community, meme culture, DeFi'
    ),
    
    'PEPE/USDT': CryptoSymbol(
        symbol='PEPE/USDT',
        name='Pepe',
        category=CryptoCategory.MEME,
        market_cap_rank=47,
        bitcoin_correlation=0.55,
        volatility_profile='Very High',
        trading_volume='Medium',
        launch_date='2023-04-14',
        description='Pepe the Frog meme token',
        use_case='Meme culture, community, speculation'
    ),
    
    'FLOKI/USDT': CryptoSymbol(
        symbol='FLOKI/USDT',
        name='Floki Inu',
        category=CryptoCategory.MEME,
        market_cap_rank=48,
        bitcoin_correlation=0.50,
        volatility_profile='Very High',
        trading_volume='Medium',
        launch_date='2021-06-25',
        description='Viking-themed meme token',
        use_case='Meme culture, gaming, DeFi'
    ),
    
    'BONK/USDT': CryptoSymbol(
        symbol='BONK/USDT',
        name='Bonk',
        category=CryptoCategory.MEME,
        market_cap_rank=49,
        bitcoin_correlation=0.45,
        volatility_profile='Very High',
        trading_volume='Medium',
        launch_date='2022-12-25',
        description='Solana-based meme token',
        use_case='Meme culture, Solana ecosystem'
    ),
    
    'WIF/USDT': CryptoSymbol(
        symbol='WIF/USDT',
        name='dogwifhat',
        category=CryptoCategory.MEME,
        market_cap_rank=50,
        bitcoin_correlation=0.40,
        volatility_profile='Very High',
        trading_volume='Medium',
        launch_date='2023-12-19',
        description='Solana-based meme token',
        use_case='Meme culture, Solana ecosystem'
    ),
    
    # Emerging sectors
    'RNDR/USDT': CryptoSymbol(
        symbol='RNDR/USDT',
        name='Render Token',
        category=CryptoCategory.EMERGING,
        market_cap_rank=51,
        bitcoin_correlation=0.65,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2017-10-17',
        description='Decentralized GPU rendering network',
        use_case='3D rendering, AI, computing resources'
    ),
    
    'HIVE/USDT': CryptoSymbol(
        symbol='HIVE/USDT',
        name='Hive',
        category=CryptoCategory.EMERGING,
        market_cap_rank=52,
        bitcoin_correlation=0.55,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2020-03-20',
        description='Social media blockchain',
        use_case='Social media, content creation, DeFi'
    ),
    
    'STEEM/USDT': CryptoSymbol(
        symbol='STEEM/USDT',
        name='Steem',
        category=CryptoCategory.EMERGING,
        market_cap_rank=53,
        bitcoin_correlation=0.50,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2016-03-24',
        description='Social media blockchain platform',
        use_case='Content creation, social media, rewards'
    ),
    
    'WAXP/USDT': CryptoSymbol(
        symbol='WAXP/USDT',
        name='WAX',
        category=CryptoCategory.GAMING,
        market_cap_rank=54,
        bitcoin_correlation=0.60,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2017-12-21',
        description='Gaming and NFT platform',
        use_case='Gaming, NFTs, virtual items'
    ),
    
    'TLM/USDT': CryptoSymbol(
        symbol='TLM/USDT',
        name='Alien Worlds',
        category=CryptoCategory.GAMING,
        market_cap_rank=55,
        bitcoin_correlation=0.55,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2021-04-01',
        description='Play-to-earn gaming platform',
        use_case='Gaming, NFTs, yield farming'
    ),
    
    # Additional high-potential coins
    'ALICE/USDT': CryptoSymbol(
        symbol='ALICE/USDT',
        name='My Neighbor Alice',
        category=CryptoCategory.GAMING,
        market_cap_rank=56,
        bitcoin_correlation=0.65,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2021-03-15',
        description='Blockchain-based multiplayer game',
        use_case='Gaming, NFTs, virtual world'
    ),
    
    'ALPHA/USDT': CryptoSymbol(
        symbol='ALPHA/USDT',
        name='Alpha Finance Lab',
        category=CryptoCategory.DEFI,
        market_cap_rank=57,
        bitcoin_correlation=0.70,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2020-09-01',
        description='Cross-chain DeFi platform',
        use_case='DeFi, yield farming, cross-chain'
    ),
    
    'AUDIO/USDT': CryptoSymbol(
        symbol='AUDIO/USDT',
        name='Audius',
        category=CryptoCategory.EMERGING,
        market_cap_rank=58,
        bitcoin_correlation=0.60,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2019-09-15',
        description='Decentralized music platform',
        use_case='Music streaming, artist rewards, NFTs'
    ),
    
    'CLV/USDT': CryptoSymbol(
        symbol='CLV/USDT',
        name='Clover Finance',
        category=CryptoCategory.INFRASTRUCTURE,
        market_cap_rank=59,
        bitcoin_correlation=0.65,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2021-06-21',
        description='Substrate-based DeFi platform',
        use_case='DeFi, cross-chain, infrastructure'
    ),
    
    'CTSI/USDT': CryptoSymbol(
        symbol='CTSI/USDT',
        name='Cartesi',
        category=CryptoCategory.INFRASTRUCTURE,
        market_cap_rank=60,
        bitcoin_correlation=0.60,
        volatility_profile='High',
        trading_volume='Low',
        launch_date='2019-04-18',
        description='Layer-2 platform for DApps',
        use_case='Smart contracts, scalability, DeFi'
    )
}


def get_symbols_by_category(category: CryptoCategory) -> List[str]:
    """Get all symbols for a specific category"""
    return [symbol for symbol, data in CRYPTO_SYMBOLS.items() if data.category == category]


def get_symbols_by_correlation_range(min_corr: float, max_corr: float) -> List[str]:
    """Get symbols within a specific Bitcoin correlation range"""
    return [symbol for symbol, data in CRYPTO_SYMBOLS.items() 
            if min_corr <= data.bitcoin_correlation <= max_corr]


def get_high_volatility_symbols() -> List[str]:
    """Get symbols with high volatility profile"""
    return [symbol for symbol, data in CRYPTO_SYMBOLS.items() 
            if data.volatility_profile in ['High', 'Very High']]


def get_low_correlation_symbols() -> List[str]:
    """Get symbols with low Bitcoin correlation (diversification candidates)"""
    return [symbol for symbol, data in CRYPTO_SYMBOLS.items() 
            if abs(data.bitcoin_correlation) < 0.4]


def get_symbol_info(symbol: str) -> Optional[CryptoSymbol]:
    """Get detailed information for a specific symbol"""
    return CRYPTO_SYMBOLS.get(symbol)


def get_all_symbols() -> List[str]:
    """Get all available symbols"""
    return list(CRYPTO_SYMBOLS.keys())


def get_correlation_matrix_data() -> Dict[str, Dict]:
    """Get correlation data for analysis"""
    return {
        symbol: {
            'bitcoin_correlation': data.bitcoin_correlation,
            'category': data.category.value,
            'volatility': data.volatility_profile,
            'market_cap_rank': data.market_cap_rank
        }
        for symbol, data in CRYPTO_SYMBOLS.items()
    }


# Correlation analysis helper functions
def calculate_expected_correlation(symbol1: str, symbol2: str) -> float:
    """Calculate expected correlation between two symbols based on their Bitcoin correlations"""
    if symbol1 not in CRYPTO_SYMBOLS or symbol2 not in CRYPTO_SYMBOLS:
        return 0.0
    
    btc_corr1 = CRYPTO_SYMBOLS[symbol1].bitcoin_correlation
    btc_corr2 = CRYPTO_SYMBOLS[symbol2].bitcoin_correlation
    
    # Simple correlation estimation based on Bitcoin correlation
    # This is a heuristic and should be validated with real data
    if btc_corr1 > 0.7 and btc_corr2 > 0.7:
        return 0.8  # High correlation
    elif btc_corr1 < 0.3 and btc_corr2 < 0.3:
        return 0.6  # Moderate correlation
    elif (btc_corr1 > 0.7 and btc_corr2 < 0.3) or (btc_corr1 < 0.3 and btc_corr2 > 0.7):
        return 0.2  # Low correlation
    else:
        return 0.5  # Medium correlation


def get_diversification_recommendations() -> List[str]:
    """Get symbols recommended for portfolio diversification"""
    # Look for symbols with low Bitcoin correlation and different categories
    diversification_candidates = []
    
    # Get one symbol from each major category with low Bitcoin correlation
    categories_covered = set()
    
    for symbol, data in sorted(CRYPTO_SYMBOLS.items(), key=lambda x: x[1].bitcoin_correlation):
        if (data.category not in categories_covered and 
            abs(data.bitcoin_correlation) < 0.5 and
            data.volatility_profile != 'Very High'):
            
            diversification_candidates.append(symbol)
            categories_covered.add(data.category)
            
            if len(diversification_candidates) >= 10:  # Limit to top 10
                break
    
    return diversification_candidates


def get_correlation_trading_pairs() -> List[Tuple[str, str, str]]:
    """Get recommended trading pairs based on correlation analysis"""
    pairs = []
    
    # High correlation pairs (for pairs trading)
    high_corr_symbols = [s for s, d in CRYPTO_SYMBOLS.items() if d.bitcoin_correlation > 0.7]
    
    for i, symbol1 in enumerate(high_corr_symbols[:5]):
        for symbol2 in high_corr_symbols[i+1:6]:
            expected_corr = calculate_expected_correlation(symbol1, symbol2)
            if expected_corr > 0.7:
                pairs.append((symbol1, symbol2, "high_correlation"))
    
    # Low correlation pairs (for diversification)
    low_corr_symbols = [s for s, d in CRYPTO_SYMBOLS.items() if abs(d.bitcoin_correlation) < 0.4]
    
    for i, symbol1 in enumerate(low_corr_symbols[:5]):
        for symbol2 in low_corr_symbols[i+1:6]:
            expected_corr = calculate_expected_correlation(symbol1, symbol2)
            if expected_corr < 0.4:
                pairs.append((symbol1, symbol2, "low_correlation"))
    
    return pairs
