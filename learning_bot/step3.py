import pandas as pd
import matplotlib.pyplot as plt
import datetime
from PIL import Image

def start(bitcoin_im, eth_im, model_data, eth_model):
    split_date = '2018-01-01'
    # we don't need the date columns anymore
    training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
    training_set = training_set.drop('Date', 1)
    test_set = test_set.drop('Date', 1)
    window_len = 10
    norm_cols = [coin+metric for coin in ['bt_', 'eth_'] for metric in ['Close','Volume']]
    LSTM_training_inputs = []
    for i in range(len(training_set)-window_len):
        temp_set = training_set[i:(i+window_len)].copy()
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
        LSTM_training_inputs.append(temp_set)
    LSTM_training_outputs = (training_set['eth_Close'][window_len:].values/training_set['eth_Close'][:-window_len].values)-1
    LSTM_test_inputs = []
    for i in range(len(test_set)-window_len):
        temp_set = test_set[i:(i+window_len)].copy()
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
        LSTM_test_inputs.append(temp_set)
    LSTM_test_outputs = (test_set['eth_Close'][window_len:].values/test_set['eth_Close'][:-window_len].values)-1
    print(LSTM_training_inputs[0])

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    fig, ax1 = plt.subplots(1,1)
    ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
    ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,5,9]])
    ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
        training_set['eth_Close'][window_len:], label='Actual')
    ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
        ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set['eth_Close'].values[:-window_len])[0], 
        label='Predicted')
    ax1.set_title('Training Set: Single Timepoint Prediction')
    ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
    ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
    ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_model.predict(LSTM_training_inputs))+1)-\
        (training_set['eth_Close'].values[window_len:])/(training_set['eth_Close'].values[:-window_len]))), 
        xy=(0.75, 0.9),  xycoords='axes fraction',
        xytext=(0.75, 0.9), textcoords='axes fraction')
    # figure inset code taken from http://akuederle.com/matplotlib-zoomed-up-inset
    axins = zoomed_inset_axes(ax1, 3.35, loc=10) # zoom-factor: 3.35, location: centre
    axins.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
    axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
        training_set['eth_Close'][window_len:], label='Actual')
    axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
        ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set['eth_Close'].values[:-window_len])[0], 
        label='Predicted')
    axins.set_xlim([datetime.date(2017, 3, 1), datetime.date(2017, 5, 1)])
    axins.set_ylim([10,60])
    axins.set_xticklabels('')
    mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
#     plt.show()
    return fig