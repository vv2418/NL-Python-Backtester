PROMPTS = [

  
    "Backtest AAPL from 2018-01-01 to 2024-01-01 using a long-only strategy that buys when the 10-day moving average crosses above the 50-day moving average and sells on the opposite crossover.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying when the 20-day MA crosses above the 100-day MA and exiting on the reverse crossover.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying when the 5-day MA crosses above the 20-day MA and selling when the 5-day MA crosses back below the 20-day MA.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 with a rule to enter long when the 30-day MA crosses above the 200-day MA and exit when it crosses back below.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 using a strategy that enters long when the 50-day MA crosses above the 100-day MA and exits on the opposite crossover.",

    
    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying on a 15-day MA cross above a 60-day MA and exiting when it crosses back below.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 entering long when the 8-day MA crosses above the 40-day MA and exiting on the reverse.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 entering long when the 12-day MA crosses above the 26-day MA and exiting on the opposite crossover.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying when the 18-day MA crosses above the 90-day MA and selling when it crosses back below.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 entering long when the 7-day MA crosses above the 30-day MA and exiting when it crosses below.",


  
    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying when the 10-day MA crosses above the 50-day MA only if 20-day realized volatility is below its 1-year median. Exit on opposite crossover.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying only when the 20-day MA crosses above the 100-day MA AND 20-day RV is below the trailing 252-day median. Exit on opposite crossover.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying when the 5-day MA crosses above the 20-day MA only if 20-day realized volatility is unusually low (below its 1-year median). Sell on opposite cross.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 where entries require both a 30/200-day MA cross AND 20-day volatility below the 1-year median. Exit on MA cross down.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 with an entry rule: 50-day MA crosses above 100-day MA AND 20-day RV < 1-year median. Exit when 50-day crosses below 100-day.",

  
    "Backtest AAPL from 2018-01-01 to 2024-01-01 using a rule that buys on a 12/26 MA crossover only during low-volatility periods defined by 20-day RV below the past-year median. Exit on crossover down.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 entering on 18/90 MA cross up IF 20-day RV is below its 252-day trailing median. Exit on cross down.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying on 7/30 MA crossover only in low volatility environments (20-day RV < 1y median). Exit when MA reverses.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying on a 15/60 MA crossover IF volatility is below the 1-year median. Exit on crossover down.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying on a 10/40 MA cross only under low volatility (20-day RV < median of the past year). Exit on the opposite cross.",


   
    "Backtest AAPL from 2018-01-01 to 2024-01-01. Go long when the shorter moving average (10-day) rises above the longer one (50-day). Close the position when the shorter MA falls back underneath.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 and only take long trades when the 20-day moving average breaks above the 100-day moving average and volatility over the past 20 days is below its 1-year midpoint. Exit when the MA trend reverses.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 opening a position when the 5-day trend overtakes the 20-day trend and volatility is calm (20-day RV below the median of the last 252 days). Close when the fast MA loses momentum.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 using an MA rule: buy when the 30-day average crosses above the 200-day average during low-vol conditions (20-day RV < 1-year median). Exit on the MA cross down.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 only entering when the 50-day MA overtakes the 100-day MA while volatility remains subdued (20-day RV below trailing median). Exit on reversal.",


  
    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying whenever the fast MA (10-day) crosses above the slow MA (50-day) and selling when the reverse happens.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 taking a long position when the 5-day MA jumps above the 20-day MA. Exit when it falls back under.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 entering long after a 12-day MA crossing above a 26-day MA, and exiting after the inverse crossover.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 entering when the 18-day MA crosses above the 90-day MA and leaving when it goes below.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 long-only: buy on 7/30-day MA cross up, sell on cross down.",


    
    "Backtest AAPL from 2018-01-01 to 2024-01-01 adding a volatility filter: only enter on a 10/50 MA cross up if 20-day RV is below its 1-year median. Exit on opposite cross.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 with entries when the 20-day MA overtakes the 100-day MA AND volatility is low (20-day RV < median). Exit on bearish crossover.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 using a 5/20 MA crossover strategy conditioned on low volatility measured by 20-day RV under the 252-day median. Exit on reverse crossover.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying during calm volatility periods (20-day RV < 1-year median) when the 30-day MA crosses above 200-day MA and selling on the opposite cross.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01 buying with a 50/100-day MA crossover rule only when volatility is quiet based on 20-day RV. Exit on cross down."

    "Backtest SPY from 2020-01-01 to 2023-12-31. Go long when the 20-day moving average crosses above the 50-day moving average. Exit when the 20-day moving average crosses below the 50-day moving average. Show CAGR, max drawdown, and Sharpe ratio.",

    "Backtest TSLA from 2019-01-01 to 2024-01-01. Enter long positions when 30-day realized volatility is below its 1-year median. Exit when 30-day realized volatility exceeds its 1-year median. Calculate CAGR, max drawdown, and Sharpe ratio.",

    "Backtest AAPL from 2018-01-01 to 2024-01-01. Go long when the 10-day moving average crosses above the 50-day moving average. Only enter new positions when 20-day realized volatility is below its 1-year median. Exit when the 10-day moving average crosses back below the 50-day. Show CAGR, max drawdown, and Sharpe ratio.",

    "Backtest MSFT from 2017-01-01 to 2023-12-31. Enter long when the 5-day moving average crosses above the 20-day moving average. Exit when the 5-day moving average crosses below the 20-day moving average. Also exit when the 20-day moving average crosses below the 50-day moving average. Display CAGR, max drawdown, and Sharpe ratio.",

    "Backtest GOOGL from 2018-06-01 to 2024-06-01. Go long when 15-day realized volatility is below its 1-year median. Exit when the 10-day moving average crosses below the 30-day moving average. Show CAGR, max drawdown, and Sharpe ratio.",

    "Backtest NVDA from 2019-01-01 to 2024-01-01. Enter long positions when the 12-day moving average crosses above the 26-day moving average and when 25-day realized volatility is below its 1-year median. Exit when the 12-day moving average crosses below the 26-day moving average. Show CAGR, max drawdown, and Sharpe ratio.",

    "Backtest QQQ from 2018-01-01 to 2024-01-01. Go long when the 8-day moving average crosses above the 21-day moving average and 18-day realized volatility is below its 1-year median. Exit when the 8-day moving average crosses below the 21-day moving average or when 18-day realized volatility rises above its 1-year median. Calculate CAGR, max drawdown, and Sharpe ratio.",

    "Backtest AMZN from 2020-01-01 to 2023-12-31. Enter long when the 3-day moving average crosses above the 7-day moving average. Exit when the 3-day moving average crosses below the 7-day moving average. Show CAGR, max drawdown, and Sharpe ratio.",

    "Backtest JPM from 2015-01-01 to 2024-01-01. Go long when the 50-day moving average crosses above the 200-day moving average. Exit when the 50-day moving average crosses below the 200-day moving average. Display CAGR, max drawdown, and Sharpe ratio.",

    "Backtest VOO from 2016-06-15 to 2023-11-30. Enter long positions when the 14-day moving average crosses above the 40-day moving average and 22-day realized volatility is below its 1-year median. Exit when the 14-day moving average crosses below the 40-day moving average. Show CAGR, max drawdown, and Sharpe ratio.",

    "Backtest META from 2018-01-01 to 2024-01-01. Go long when the 10-day moving average crosses above the 30-day moving average. Exit when the 10-day moving average crosses below the 30-day moving average. Show CAGR, max drawdown, and Sharpe ratio.",

    "Backtest NFLX from 2017-01-01 to 2023-12-31. Enter long when the 15-day moving average crosses above the 40-day moving average. Exit when the 15-day moving average crosses below the 40-day moving average. Report CAGR, max drawdown, and Sharpe ratio.",

    "Backtest AMD from 2019-01-01 to 2024-01-01. Buy when the 5-day moving average crosses above the 15-day moving average. Sell when the 5-day moving average crosses below the 15-day moving average. Show CAGR, max drawdown, and Sharpe ratio.",

    "Backtest ORCL from 2016-01-01 to 2023-12-31. Enter long when the 20-day moving average crosses above the 60-day moving average. Exit when the 20-day moving average crosses below the 60-day moving average. Show performance metrics.",

    "Backtest INTC from 2018-01-01 to 2024-01-01. Go long when the 12-day moving average crosses above the 30-day moving average. Exit when it crosses below. Show CAGR and max drawdown.",

    "Backtest XOM from 2017-01-01 to 2024-01-01. Enter long when the 8-day moving average crosses above the 20-day. Exit when it crosses below. Display Sharpe, CAGR, and drawdown.",

    "Backtest KO from 2015-01-01 to 2023-12-31. Enter long when the 30-day moving average crosses above the 100-day moving average. Exit when the 30-day moving average crosses below the 100-day. Show metrics.",

    "Backtest PEP from 2018-01-01 to 2024-01-01. Buy when the 18-day MA crosses above the 45-day MA. Exit when the reverse happens. Show CAGR, max drawdown, and Sharpe ratio.",

    "Backtest DIS from 2016-01-01 to 2023-12-31. Enter long when the 5-day MA crosses above the 25-day MA. Exit on opposite crossover. Show performance.",

    "Backtest BA from 2019-01-01 to 2024-01-01. Go long when the 10-day MA crosses above the 60-day MA. Exit when it crosses below. Show CAGR and Sharpe ratio.",


    
    "Backtest IBM from 2015-01-01 to 2024-01-01. Go long when 20-day realized volatility is below its 1-year median. Exit when volatility rises above its 1-year median. Show CAGR and Sharpe ratio.",

    "Backtest MRK from 2017-01-01 to 2024-01-01. Enter long when 30-day realized volatility is below its 1-year median. Exit when it exceeds the median. Report performance metrics.",

    "Backtest ABNB from 2020-12-15 to 2024-01-01. Buy when 15-day realized volatility is below the 1-year median. Sell when it is above the median. Show CAGR and drawdown.",

    "Backtest SHOP from 2018-01-01 to 2024-01-01. Go long when 25-day realized volatility is below its 1-year median. Exit when volatility rises above the median. Show metrics.",

    "Backtest UBER from 2019-05-10 to 2024-01-01. Enter long when 12-day realized volatility is below its 1-year median. Exit when above. Show Sharpe ratio.",

    "Backtest SQ from 2016-01-01 to 2023-12-31. Buy when 18-day realized volatility is below the 1-year median. Sell when it crosses above. Report performance.",

    "Backtest PYPL from 2015-01-01 to 2024-01-01. Enter long when 10-day realized volatility is below the 1-year median. Exit when above median. Show metrics.",

    "Backtest WMT from 2017-01-01 to 2023-12-31. Buy when 22-day realized volatility is below its 1-year median. Exit when above. Show CAGR and drawdown.",

    "Backtest HD from 2016-01-01 to 2024-01-01. Go long when 14-day realized volatility is below the 1-year median. Exit when above. Show performance metrics.",

    "Backtest LOW from 2018-01-01 to 2024-01-01. Enter long when 30-day realized volatility is below the 1-year median. Sell when above. Calculate Sharpe ratio and drawdown.",


  
    "Backtest COST from 2017-01-01 to 2024-01-01. Buy when the 10-day MA crosses above the 30-day MA AND 20-day realized volatility is below its 1-year median. Exit when the 10-day MA crosses below the 30-day MA. Show full metrics.",

    "Backtest TGT from 2018-01-01 to 2024-01-01. Enter long when the 6-day MA crosses above the 20-day MA and 15-day RV is below its 1-year median. Exit when the 6-day MA crosses below. Report metrics.",

    "Backtest MCD from 2015-01-01 to 2024-01-01. Buy when 12-day MA crosses above 24-day MA and 20-day RV is below its 1-year median. Exit on opposite crossover. Show metrics.",

    "Backtest SBUX from 2016-01-01 to 2024-01-01. Go long when 8-day MA crosses above 18-day MA and 14-day RV is below its median. Exit on opposite crossover. Show performance.",

    "Backtest NKE from 2017-01-01 to 2024-01-01. Buy when 10-day MA crosses above 25-day MA AND 25-day volatility is below its median. Exit on the reverse crossover. Show Sharpe ratio.",

    "Backtest LULU from 2018-01-01 to 2024-01-01. Enter when 7-day MA crosses above 20-day MA and 10-day RV is below 1-year median. Exit on crossover down. Show metrics.",

    "Backtest CRM from 2015-01-01 to 2024-01-01. Buy when 20-day MA crosses above 50-day MA and 18-day realized volatility is below its median. Exit on opposite crossover. Show metrics.",

    "Backtest ADBE from 2016-01-01 to 2024-01-01. Enter long when 8-day MA crosses above 30-day MA and 12-day RV is below median. Exit when 8-day MA crosses below. Show performance.",

    "Backtest TSM from 2018-01-01 to 2024-01-01. Go long when 15-day MA crosses above 40-day MA AND 25-day RV is below its 1-year median. Exit on crossing down. Report metrics.",

    "Backtest AVGO from 2017-01-01 to 2024-01-01. Enter long when 10-day MA crosses above 35-day MA and 20-day RV is below median. Exit when 10-day MA crosses below. Show metrics.",

    "Backtest BABA from 2016-01-01 to 2024-01-01. Buy when 5-day MA crosses above 15-day MA AND 12-day RV is below median. Exit when 5-day MA crosses below. Show Sharpe and drawdown.",

    "Backtest PFE from 2015-01-01 to 2023-12-31. Enter long when 15-day MA crosses above 45-day MA and 20-day RV is below median. Exit on the opposite crossover. Show performance.",

    "Backtest CAT from 2017-01-01 to 2024-01-01. Buy when 10-day MA crosses above 30-day MA and 25-day RV is below the median. Exit when below. Show metrics.",

    "Backtest MMM from 2015-01-01 to 2024-01-01. Enter long when 8-day MA crosses above 22-day MA and 14-day RV is below median. Exit when 8-day MA crosses below. Show results.",

    "Backtest GS from 2016-01-01 to 2024-01-01. Go long when 12-day MA crosses above 28-day MA AND 18-day RV < median. Exit on opposite crossover. Show metrics.",


    
    "Backtest PLTR from 2020-01-01 to 2024-01-01. Go long when the 10-day MA crosses above the 20-day MA. Exit when the 10-day MA crosses below the 20-day MA OR when 15-day RV exceeds its 1-year median. Show metrics.",

    "Backtest ROKU from 2018-01-01 to 2024-01-01. Enter long when 8-day MA crosses above 25-day MA. Exit when 8-day MA crosses below OR 25-day RV rises above median. Show results.",

    "Backtest SNAP from 2017-01-01 to 2023-12-31. Buy when 5-day MA crosses above 12-day MA. Exit when 5-day MA crosses below OR 10-day RV exceeds median. Show metrics.",

    "Backtest ETSY from 2018-01-01 to 2024-01-01. Long when 15-day MA > 40-day MA. Exit if 15-day MA < 40-day MA or 20-day RV > median. Show metrics.",

    "Backtest RBLX from 2021-01-01 to 2024-01-01. Enter long when 10-day MA crosses above 30-day MA. Exit on crossover down OR RV above median. Show results.",

    "Backtest COIN from 2021-01-01 to 2024-01-01. Buy when 12-day MA crosses above 26-day MA. Exit when the 12-day MA crosses below OR 14-day RV exceeds median. Show metrics.",

    "Backtest HOOD from 2021-07-01 to 2024-01-01. Enter long when 8-day MA crosses above 20-day MA. Exit if 8-day MA crosses below or 12-day RV > median. Show metrics.",

    "Backtest AFRM from 2021-01-01 to 2024-01-01. Buy when 10-day MA crosses above 25-day MA. Exit when 10-day MA crosses below OR RV > median. Show metrics.",

    "Backtest DKNG from 2020-01-01 to 2024-01-01. Go long when 7-day MA crosses above 18-day MA. Exit when 7-day MA crosses below or RV rises above its median. Show performance.",

    "Backtest SOFI from 2021-01-01 to 2024-01-01. Enter long when 6-day MA crosses above 16-day MA. Exit when the 6-day MA crosses below OR 12-day RV exceeds median. Show metrics.",
]

