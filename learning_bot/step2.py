import pandas as pd
import matplotlib.pyplot as plt
import datetime
from PIL import Image

def start(bitcoin_im, market_info):
    
    split_date = '2018-01-01'
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
    ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
    ax1.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] < split_date]['bt_Close'], 
            color='#B08FC7', label='Training')
    ax1.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] >= split_date]['bt_Close'], 
            color='#8FBAC8', label='Test')
    ax2.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] < split_date]['eth_Close'], 
            color='#B08FC7')
    ax2.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
            market_info[market_info['Date'] >= split_date]['eth_Close'], color='#8FBAC8')
    ax1.set_xticklabels('')
    ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
    ax2.set_ylabel('Ethereum Price ($)',fontsize=12)
    plt.tight_layout()
    ax1.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0., prop={'size': 14})
    # fig.figimage(bitcoin_im.resize((int(bitcoin_im.size[0]*0.65), int(bitcoin_im.size[1]*0.65)), Image.ANTIALIAS), 
    #             200, 260, zorder=3,alpha=.5)
    # fig.figimage(eth_im.resize((int(eth_im.size[0]*0.65), int(eth_im.size[1]*0.65)), Image.ANTIALIAS), 
    #             350, 40, zorder=3,alpha=.5)
    return fig
#     plt.show()
