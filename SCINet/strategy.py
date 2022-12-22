import os
import torch
import pandas as pd
import numpy as np
from dateutil import parser as ps
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil import parser

class Estimation:
    def __init__(self, args):
        self.args = args
        self.total_loss = []
        self.total_acc = []
        self.total_precision = []
        self.total_recall = []
        self.total_f1 = []
        self.strategy_data = pd.DataFrame()
        self.status_pred = 0
        self.status_true = 0

    def decide_action(self, values, status, threshold=20):
        action = 0
        stay = sum(values == 0)
        long_buy = sum(values == 1)
        short_buy = sum(values == 2)
        if status == 0:
            if long_buy >= threshold:
                action = 1
                status = 1
            if short_buy >= threshold:
                action = 3
                status = 2

        elif status == 1:
            fix_long = long_buy - short_buy
            if fix_long < (threshold / 2):
                action = 2
                status = 0

        elif status == 2:
            fix_short = short_buy - long_buy
            if fix_short < (threshold / 2):
                action = 4
                status = 0
        else:
            pass

        return [action, stay, long_buy, short_buy], status


    def run_batch(self, pred, true, val):
        columns = ['date', 'op', 'hi', 'lo', 'cl']
        tmp_val_df = pd.DataFrame(val[:,0,:].detach().cpu().numpy(), columns=columns)
        calc_pred = pred.detach().cpu().numpy().argmax(axis=2)
        calc_true = true.detach().cpu().numpy().argmax(axis=2)
        trades = []
        for i in range(true.shape[0]):
            action_pred, status_pred = self.decide_action(calc_pred[i], self.status_pred, threshold=20)
            action_true, status_true = self.decide_action(calc_true[i], self.status_true, threshold=20)
            trades.append(action_pred + action_true)

            self.status_pred = status_pred
            self.status_true = status_true

        trades_df = pd.DataFrame(trades, columns=['ac_p', 'st_p', 'lb_p', 'sb_p', 'ac_t', 'st_t', 'lb_t', 'sb_t'])
        tmp_val_df = pd.concat([tmp_val_df, trades_df], axis=1)

        reshape_dims = true.shape[0] * true.shape[1]
        acc = accuracy_score(y_true=true.detach().cpu().reshape(reshape_dims, 3).argmax(axis=1),
                                 y_pred=pred.detach().cpu().reshape(reshape_dims, 3).argmax(axis=1))
        precision = precision_score(y_true=true.detach().cpu().reshape(reshape_dims, 3).argmax(axis=1),
                                        y_pred=pred.detach().cpu().reshape(reshape_dims, 3).argmax(axis=1), average='macro')
        recall = recall_score(y_true=true.detach().cpu().reshape(reshape_dims, 3).argmax(axis=1),
                                  y_pred=pred.detach().cpu().reshape(reshape_dims, 3).argmax(axis=1), average='macro')
        f1 = f1_score(y_true=true.detach().cpu().reshape(reshape_dims, 3).argmax(axis=1),
                          y_pred=pred.detach().cpu().reshape(reshape_dims, 3).argmax(axis=1), average='macro')

        self.total_acc.append(acc)
        self.total_precision.append(precision)
        self.total_recall.append(recall)
        self.total_f1.append(f1)
        self.strategy_data = pd.concat([self.strategy_data, tmp_val_df])

    def calc_trade_achievement(self, strategy_data):
        achievement_df = strategy_data[strategy_data['ac_p'] != 0]
        total_profit = 0
        for price, trade in achievement_df[['cl', 'ac_p']].values:
            profit = 0
            if trade == 1:
                buy_price_long = price
            elif trade == 2:
                sell_price_long = price
                profit = sell_price_long - buy_price_long
            elif trade == 3:
                sell_price_short = price
            elif trade == 4:
                buy_price_short = price
                profit = sell_price_short - buy_price_short
            if profit != 0:
                total_profit += profit

        return total_profit

    def run(self, epoch):
        acc_out = np.average(self.total_acc)
        precision_out = np.average(self.total_precision)
        recall_out = np.average(self.total_recall)
        f1_out = np.average(self.total_f1)
        strategy_data = self.strategy_data.sort_values(by='date').reset_index(drop=True)

        if epoch + 1 >= 10:
            total_profit = self.calc_trade_achievement(strategy_data)
            print('Total Profit {}'.format(total_profit))
            strategy_data.to_csv('strategy_data.csv')

        return acc_out, precision_out, recall_out, f1_out


def plot_pred():
    pred_data = pd.read_csv('strategy_data.csv')
    pred_data['date'] = pred_data['date'].apply(lambda x: parser.parse(str(x)))
    plot_data = pred_data.reset_index(drop=True).sort_values(by='date')

    target_col = 'cl'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data['date'],
                             y=plot_data[target_col],
                             line=dict(color='rgba(17, 250, 244, 1)'),
                             fillcolor='rgba(17, 250, 244, 1)',
                             fill=None,
                             name=target_col))

    long_df = plot_data[plot_data['ac_p'] == 1]
    fig.add_trace(go.Scatter(x=long_df['date'],
                             y=long_df[target_col],
                             mode='markers',
                             line=dict(color='rgba(10, 250, 144, 1)'),
                             marker=dict(color='rgba(10, 250, 144, 1)', size=10, opacity=0.8, symbol='star'),
                             name='long_buy'))

    short_df = plot_data[plot_data['ac_p'] == 2]
    fig.add_trace(go.Scatter(x=short_df['date'],
                             y=short_df[target_col],
                             mode='markers',
                             line=dict(color='rgba(170, 25, 14, 1)'),
                             marker=dict(color='rgba(170, 25, 14, 1)', size=10, opacity=0.8, symbol='star'),
                             name='long_sell'))

    long_df = plot_data[plot_data['ac_p'] == 3]
    fig.add_trace(go.Scatter(x=long_df['date'],
                             y=long_df[target_col],
                             mode='markers',
                             line=dict(color='rgba(100, 125, 244, 1)'),
                             marker=dict(color='rgba(100, 125, 244, 1)', size=10, opacity=0.8, symbol='star'),
                             name='short_buy'))

    short_df = plot_data[plot_data['ac_p'] == 4]
    fig.add_trace(go.Scatter(x=short_df['date'],
                             y=short_df[target_col],
                             mode='markers',
                             line=dict(color='rgba(17, 125, 114, 1)'),
                             marker=dict(color='rgba(17, 125, 114, 1)', size=10, opacity=0.8, symbol='star'),
                             name='short_sell'))

    fig.update_layout(title='推論結果の可視化',
                      plot_bgcolor='white',
                      xaxis=dict(showline=True,
                                 linewidth=1,
                                 linecolor='lightgrey',
                                 tickfont_color='grey',
                                 ticks='inside'),
                      yaxis=dict(title='BTC価格',
                                 showline=True,
                                 linewidth=1,
                                 linecolor='lightgrey',
                                 tickfont_color='grey',
                                 ticks='inside'))
    fig.show()

if __name__ == '__main__':
    plot_pred()