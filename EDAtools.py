import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote(cols_to_drop, df, class_val):
    smote_df = df.drop(cols_to_drop, axis=1)
    classes = df["award_share"] > class_val
    sm = SMOTE(random_state=42)
    smote_df, classes = sm.fit_resample(smote_df, classes)
    # indexing synthetic data
    smote_df['is_synthetic'] = smote_df.index >= len(df)
    # creating dfs for modelling
    final_df = smote_df.merge(df, how='left')
    return final_df


def calc_3sigma_cutoffs(cols: list, df):
    cutoffs = {}
    for col in cols:
        cutoff = df[df['received_vote'] == 1][col].mean() - 3 * df[df['received_vote'] == 1][col].std()
        cutoffs[col] = cutoff
    return cutoffs


def create_pie_chart(df):
    plt.pie(df['received_vote'].value_counts(),
            autopct=lambda pct: func(pct, df['received_vote'].value_counts()),
            pctdistance=1.25)
    plt.title('Players that Received MIP Votes Post Filtering')
    plt.legend(['Received Votes', 'No Votes'])
    plt.show(block=True)


def func(pct, allvals):
    absolute = int(np.round(pct / 100. * np.sum(allvals)))
    return "{:.1f}%/n({:d})".format(pct, absolute)


def vote_diff_plot(cols, df):
    for col in cols:
        plt.hist(df[df['received_vote'] == 1][col], bins=10, alpha=0.7, density=True, label='Received Votes')
        plt.hist(df[df['received_vote'] == 0][col], bins=10, alpha=0.7, density=True, label='No Votes')
        plt.ylabel('Proportion')
        plt.xlabel(col)
        plt.legend(['Received Votes', 'No Votes'])
        plt.savefig(f'C:/Users/sierr/Desktop/MSDS/Capstone/{col}.png')
        plt.clf()



