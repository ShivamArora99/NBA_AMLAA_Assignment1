# src/visualization/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.metrics import roc_curve, auc

def auc_roc_curve(y_true, y_pred_proba):
    """
    Plot the ROC curve and calculate AUC for a set of true labels and predicted probabilities.

    Args:
        y_true (np.ndarray or pd.Series): True binary labels.
        y_pred_proba (np.ndarray or pd.Series): Predicted probabilities for the positive class.

    Returns:
        None: Displays the ROC curve plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_histograms(df, columns, ncols=2):
    """
    Plot histograms for specified columns of a DataFrame.
    """
    n = len(columns)
    nrows = n // ncols + (n % ncols > 0)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    ax = ax.flat

    for i, col in enumerate(columns):
        sns.histplot(df[col], ax=ax[i], kde=True)
        ax[i].set_title(f'Histogram of {col}', fontsize=14)

    for i in range(n, len(ax)):
        ax[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_boxplots(df):
    """
    Plot boxplots for all numeric columns in the DataFrame.
    """
    numeric_data = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=numeric_data, orient='h')
    plt.title('Boxplots for All Numeric Columns')
    plt.show()

def distribution_analysis(df):
    """
    Plot distribution analysis for 'ast' and 'per' columns.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['ast'], kde=True, color='orange')
    plt.title('Distribution Analysis of ast')
    plt.xlabel('Assists')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(df['AST_per'], kde=True, color='blue')
    plt.title('Distribution Analysis of AST_per')
    plt.xlabel('Assist Percentage')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def free_throw_analysis(df):
    """
    Plot distribution analysis for 'FTM', 'FTA', and 'FT_per' columns.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(df['FTM'], kde=True, color='green')
    plt.title('Distribution Analysis of FTM')
    plt.xlabel('Free Throws Made')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    sns.histplot(df['FTA'], kde=True, color='purple')
    plt.title('Distribution Analysis of FTA')
    plt.xlabel('Free Throw Attempts')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    sns.histplot(df['FT_per'], kde=True, color='red')
    plt.title('Distribution Analysis of FT_per')
    plt.xlabel('Free Throw Percentage')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def top_teams_no_drafted(df):
    """
    Plot pie chart for top 10 teams with no drafted players.
    """
    top_10 = df['team'].value_counts().nlargest(10).index
    top_10 = df[df['team'].isin(top_10)]
    no_drafted = top_10[top_10['drafted'] == 0]
    counts = no_drafted['team'].value_counts().reset_index()
    counts.columns = ['team', 'count']

    fig = px.pie(counts, names='team', values='count', color='team',
                 color_discrete_sequence=px.colors.qualitative.Pastel,
                 title='Top 10 Teams with No Drafted Players')
    fig.update_layout(xaxis_title='Team', yaxis_title='Counts')
    fig.show()

def yearly_free_throw_analysis(df):
    """
    Plot yearly analysis for free throw percentages.
    """
    yearly_ftp = df.groupby('year')['FT_per'].sum().reset_index()
    yearly_ftp['year'] = pd.to_datetime(yearly_ftp['year'], format='%Y').dt.year

    plt.figure(figsize=(12, 6))
    fig = sns.relplot(data=yearly_ftp, x='year', y='FT_per', kind='line', height=5, aspect=2)
    fig.set(title='Free Throws Percentage Yearly Analysis', ylabel='Free Throws Percentage', xlabel="Year")

    sns.regplot(data=yearly_ftp, x='year', y='FT_per', scatter=False, ax=fig.ax, line_kws={'color': 'black', 'linestyle': '--'})

    fig.ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(False)
    plt.show()

def yearly_block_percentage_analysis(df):
    """
    Plot yearly analysis for block percentages.
    """
    yearly_blk = df.groupby('year')['blk_per'].sum().reset_index()
    yearly_blk['year'] = pd.to_datetime(yearly_blk['year'], format='%Y').dt.year

    fig = px.bar(yearly_blk, x='year', y='blk_per', title='Block Percentage Yearly Analysis',
                 labels={'blk_per': 'Block Percentage', 'year': 'Year'}, color_discrete_sequence=['khaki'])
    fig.add_scatter(x=yearly_blk['year'], y=yearly_blk['blk_per'].rolling(window=2).mean(),
                    mode='lines', line=dict(color='red', dash='dash'), name='Trendline')
    fig.update_layout(xaxis_title='Year', yaxis_title='Block Percentage', xaxis=dict(tickmode='linear'), template='plotly_white')
    fig.show()

def yearly_games_played_analysis(df):
    """
    Plot yearly analysis for number of games played.
    """
    yearly_gp = df.groupby('year')['GP'].sum().reset_index()
    yearly_gp['year'] = pd.to_datetime(yearly_gp['year'], format='%Y').dt.year

    plt.figure(figsize=(12, 6))
    fig = sns.relplot(data=yearly_gp, x='year', y='GP', kind='line', height=5, aspect=2)
    fig.set(title='Number of Games Played Yearly Analysis', ylabel='Number of Games Played', xlabel="Year")

    sns.regplot(data=yearly_gp, x='year', y='GP', scatter=True, ax=fig.ax, line_kws={'color': 'purple', 'linestyle': '--'})

    fig.ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.show()

def conference_count_plot(df):
    """
    Plot count of different conferences.
    """
    fig = px.histogram(df, x='conf', color_discrete_sequence=['lightcoral'])
    fig.update_layout(title='Count Plot of Conference', xaxis_title='Conf', yaxis_title='Count', bargap=0.2)
    fig.show()

def top_teams_steals(df):
    """
    Plot top 20 teams by total steals.
    """
    steals_team = df.groupby('team')['stl'].sum().reset_index()
    top_20 = steals_team.nlargest(20, 'stl')
    fig = px.bar(top_20, x='team', y='stl', color='team', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(title='Top 20 Teams by Total Steals', xaxis_title='Team', yaxis_title='Total Steals')
    fig.show()


def eda(df):
  df.columns = [train_col.lower() for train_col in df.columns]
  #  Top 10 Teams
  top_10 = df['team'].value_counts().nlargest(10).index
  top_10 = df[df['team'].isin(top_10)]
  no_drafted = top_10[top_10['drafted'] == 0]

  fig = px.bar(no_drafted, x='team', title='Top Team with Undrafted Players Analysis',
              labels={'team': 'Top most Teams'}, color_discrete_sequence=['burlywood'])
  fig.update_layout(
      width=800,
      height=500,
      xaxis_title='team',
      yaxis_title='Counts of Undrafted Players',
      xaxis=dict(tickmode='linear'),
      template='plotly_white'
  )

  # Show the plot
  fig.show()

  yearly_ftp = df.groupby('year')['ft_per'].sum().reset_index()
  yearly_ftp['year'] = pd.to_datetime(yearly_ftp['year'], format='%Y').dt.year
  fig = px.line(yearly_ftp,
                x='year',
                y='ft_per',
                title='Free Throws Attempts Yearly Analysis',
                labels={'ft_per': 'Free Throws Percentage', 'year': 'Year'},
                template='plotly_white')

  fig.add_scatter(x=yearly_ftp['year'],
                  y=yearly_ftp['ft_per'].rolling(window=2).mean(),
                  mode='lines',
                  line=dict(color='red', dash='dash'),
                  name='Trendline')

  # Customize the layout
  fig.update_layout(
      width=800,
      height=500,
      xaxis_title='Year',
      yaxis_title='Free Throws Percentage',
      xaxis=dict(tickmode='linear'),
      template='plotly_white'
  )

  # Show the plot
  fig.show()


  top_5 = df['team'].value_counts().nlargest(5).index
  top_5 = df[df['team'].isin(top_5)]
  yearly_team = top_5.groupby(['year','team'])['gp'].sum().reset_index()
  yearly_team['year'] = pd.to_datetime(yearly_team['year'], format='%Y').dt.year
  yearly_team=yearly_team[yearly_team['year'] >=2016]
  fig = px.bar(yearly_team, x='team', y='gp', facet_col='year', barmode='group', title='Top 5 teams with number of games played across seasons 2016-18',
              labels={'gp': 'Number of Games Played', 'team': 'Top 5 Teams',}, color_discrete_sequence=['lightcoral'])

  fig.update_layout(
      width=1100,
      height=500,
      xaxis_title='Top 5 Teams',
      yaxis_title='Number of Games Played',
      xaxis=dict(tickmode='linear'),
      template='plotly_white'
  )

  # Show the plot
  fig.show()

  top_10 = df['team'].value_counts().nlargest(10).index
  top_10_df = df[df['team'].isin(top_10)]

  perf_team = top_10_df.groupby('team')[['twop_per', 'tp_per']].sum().reset_index()
  perf_team = perf_team.melt(id_vars='team',
                                    value_vars=['twop_per', 'tp_per'],
                                    var_name='Pointer Type',
                                    value_name='Total')
  fig = px.bar(perf_team, x='team', y='Total', color='Pointer Type',barmode='group',
              title='Top 10 Teams Performance Metrics: Two and Three Pointers',labels={'team': 'Team', 'Total': 'Total Points'},color_discrete_sequence=['lightcoral', 'deeppink'])

  # Update layout
  fig.update_layout(
      width=1100,
      height=500,
      xaxis_title='Top 10 Teams',
      yaxis_title='Total Points',
      xaxis=dict(tickmode='linear'),
      template='plotly_white'
  )

  # Show the plot
  fig.show()

  top_10_a = df['team'].value_counts().nlargest(10).index
  top_10_df1 = df[df['team'].isin(top_10_a)]

  perf_team1 = top_10_df1.groupby('team')[['stl_per', 'blk_per']].sum().reset_index()
  perf_team1 = perf_team1.melt(id_vars='team',
                                    value_vars=['stl_per', 'blk_per'],
                                    var_name='Pointer Type',
                                    value_name='Total')
  fig = px.bar(perf_team1, x='team', y='Total', color='Pointer Type',barmode='group',
              title='Top 10 Teams Performance Metrics: Steals and Blocks Percentages',labels={'team': 'Team', 'Total': 'Total steal and block points'},color_discrete_sequence=['lightgreen', 'olive'])

  # Update layout
  fig.update_layout(
      width=1100,
      height=500,
      xaxis_title='Top 10 Teams',
      yaxis_title='Total Points',
      xaxis=dict(tickmode='linear'),
      template='plotly_white'
  )

  # Show the plot
  fig.show()


  top_101 = df['team'].value_counts().nlargest(10).index
  top_10 = df[df['team'].isin(top_101)]
  drafted = top_10[top_10['drafted'] ==1]
  counts = drafted['team'].value_counts().reset_index()
  counts.columns = ['team', 'count']

  fig = px.pie(counts, names='team', values='count', color='team',color_discrete_sequence=["blue","red"],
                title='Top Teams with Drafted Players')
  fig.update_layout(
        width=800,
        height=500,
          xaxis_title='Team',
          yaxis_title='Counts'
      )
  fig.show()



