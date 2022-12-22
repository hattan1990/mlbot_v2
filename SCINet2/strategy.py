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


    def run_batch(self, pred, true, val):
        columns = ['date', 'op', 'hi', 'lo', 'cl']
        tmp_val_df = pd.DataFrame(val[:,0,:].detach().cpu().numpy(), columns=columns)
        calc_pred = pred.detach().cpu().numpy().argmax(axis=1)
        calc_true = true.detach().cpu().numpy().argmax(axis=1)
        tmp_val_df['true'] = calc_true
        tmp_val_df['pred'] = calc_pred

        acc = accuracy_score(y_true=true.detach().cpu().argmax(axis=1),
                             y_pred=pred.detach().cpu().argmax(axis=1))
        precision = precision_score(y_true=true.detach().cpu().argmax(axis=1),
                                        y_pred=pred.detach().cpu().argmax(axis=1), average='macro')
        recall = recall_score(y_true=true.detach().cpu().argmax(axis=1),
                                  y_pred=pred.detach().cpu().argmax(axis=1), average='macro')
        f1 = f1_score(y_true=true.detach().cpu().argmax(axis=1),
                          y_pred=pred.detach().cpu().argmax(axis=1), average='macro')

        self.total_acc.append(acc)
        self.total_precision.append(precision)
        self.total_recall.append(recall)
        self.total_f1.append(f1)
        self.strategy_data = pd.concat([self.strategy_data, tmp_val_df])

    def run(self, epoch):
        acc_out = np.average(self.total_acc)
        precision_out = np.average(self.total_precision)
        recall_out = np.average(self.total_recall)
        f1_out = np.average(self.total_f1)
        strategy_data = self.strategy_data.sort_values(by='date').reset_index(drop=True)

        if epoch + 1 >= 10:
            profit_df = strategy_data[strategy_data['pred'] != 0]
            profit_df.loc[profit_df['pred'] == 1, 'profit'] = profit_df['cl'] - profit_df['op']
            profit_df.loc[profit_df['pred'] == 2, 'profit'] = profit_df['op'] - profit_df['cl']
            total_profit = profit_df['profit'].sum()
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