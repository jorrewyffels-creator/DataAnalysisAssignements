#!/usr/bin/env python
# coding: utf-8

# # descriptive.ipynb
# 
# This notebook calculated descriptive statistics in detail. The main notebook reports just a summary.
# 
# The `display_summary_table` and `plot_descriptive` functions below are called from the main notebook.
# 
# <br>
# <br>

# In[1]:

from parse_data import df


from IPython.display import display,Markdown #,HTML
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd


def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )



# Below the previously developed `parse_data.ipynb` notebook is run. See that notebook for details.

# In[2]:


get_ipython().run_line_magic('run', "'parse_data.py'")

df.describe(include='all')


# To create a custom display of descriptive statistics, let's first define functions that will calculate central tendency and dispersion metrics.
# 
# Refer also to [this notebook](https://github.com/0todd0000/OpenBook-DataAnalysisPracticeInPythonAndJupyter/blob/master/Lessons/Lesson04/5-Examples/DescriptiveStatsExamples.ipynb) for details regarding how create custom descriptive statistics tables.

# In[3]:


def central(x, print_output=True):
    x0     = np.mean( x )
    x1     = np.median( x )
    x2     = stats.mode( x ).mode
    return x0, x1, x2


def dispersion(x, print_output=True):
    y0 = np.std( x ) # standard deviation
    y1 = np.min( x )  # minimum
    y2 = np.max( x )  # maximum
    y3 = y2 - y1      # range
    y4 = np.percentile( x, 25 ) # 25th percentile (i.e., lower quartile)
    y5 = np.percentile( x, 75 ) # 75th percentile (i.e., upper quartile)
    y6 = y5 - y4 # inter-quartile range
    return y0,y1,y2,y3,y4,y5,y6


# <br>
# 
# Let's now assemble and display a central tendency table:
# 
# </br>

# In[4]:


def display_central_tendency_table(num=1):
    display_title('Central tendency summary statistics.', pref='Table', num=num, center=False)
    df_central = df.select_dtypes(include='number').apply(lambda x: central(x), axis=0)
    round_dict = {'grades': 1, 'stress': 1, 'density': 6, 'sugar': 3} #change this !!!!!!!!!
    df_central = df_central.round( round_dict )
    row_labels = 'mean', 'median', 'mode'
    df_central.index = row_labels
    display( df_central )

display_central_tendency_table(num=1)


# <br>
# 
# Let's repeat for a dispersion table:
# 
# </br>

# In[5]:


def display_dispersion_table(num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    round_dict            = {'quality': 3, 'acidity': 3, 'density': 6, 'sugar': 3}
    df_dispersion = (
    df.select_dtypes(include="number")
      .apply(lambda x: dispersion(x), axis=0)
      .round(round_dict)
)
    row_labels_dispersion = 'st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR'
    df_dispersion.index   = row_labels_dispersion
    display( df_dispersion )

display_dispersion_table(num=2)


# The variables were already given easier names in the parsa_data file, so we don't need to do that anymore!

# In[ ]:





# Let's create scatterplots for the DV (quality) vs. each of the three IVs (acid, density, sugar):

# In[6]:


fig,axs = plt.subplots( 1, 5, figsize=(20,5), tight_layout=True )
plt.yticks([1, 2, 3], ['Low', 'Moderate', 'High'])
axs[0].scatter( df['studyhours'], df['grades'], alpha=0.2, color='b' )
axs[1].scatter( df['echours'], df['grades'], alpha=0.2, color='r' )
axs[2].scatter( df['sleephours'], df['grades'], alpha=0.2, color='g' )
axs[3].scatter( df['socialhours'], df['grades'], alpha=0.2, color='m' )
axs[4].scatter( df['activityhours'], df['grades'], alpha=0.2, color='c' )

xlabels = 'Study hours', 'Extracurricualar hours', 'Sleep hours', 'Social hours', 'Physically active hours'
[ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
axs[0].set_ylabel('Grades')
[ax.set_yticklabels([])  for ax in axs[0:]]
plt.show()


# Next let's add regression lines and correlation coefficients to each plot:

# In[7]:


for col in ['studyhours', 'echours', 'sleephours', 'socialhours', 'activityhours']:
    print(col, len(df[col]), len(df['grades']))


# In[8]:


def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r

def plot_regression_line(ax, x, y, **kwargs):
    a,b   = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0 + b, a*x1 + b
    ax.plot([x0,x1], [y0,y1], **kwargs)


# In[9]:


fig, axs = plt.subplots(1, 5, figsize=(20,5), tight_layout=True)

ivs = [df['studyhours'].values,
       df['echours'].values,
       df['sleephours'].values,
       df['socialhours'].values,
       df['activityhours'].values]

colors = ['b', 'r', 'g', 'm', 'c']

for ax, x, c in zip(axs, ivs, colors):
    ax.scatter(x, df['grades'].values, alpha=0.2, color=c)
    plot_regression_line(ax, x, df['grades'].values, color='k', ls='-', lw=2)
    r = corrcoeff(x, df['grades'].values)
    ax.text(0.7, 0.3, f'r = {r:.3f}', color=c,
            transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

xlabels = ['Study hours', 'Extracurricular hours', 'Sleep hours', 'Social hours', 'Physically active hours']
[ax.set_xlabel(s) for ax, s in zip(axs, xlabels)]
axs[0].set_ylabel('Grades')
[ax.set_yticklabels([]) for ax in axs[1:]]
plt.show()


# In[ ]:





# * The first plot has a high correlation.
# * The correlation for plots 2,3 and 4 is very low.
# * The last plot shows a low correlation. 
# 
# Since it is very plausible that the correlation between lifestyle factors and grades could be different, it may be interesting to give the high and low grades different colors for the variables that did not show a good correlation. We will color everyting above the median grade red and everything below blue. Let's also draw separate regression lines.
# 

# In[10]:


import matplotlib.pyplot as plt
import numpy as np

ivs = ['echours', 'sleephours', 'socialhours', 'activityhours']
xlabels = ['Extracurricular hours', 'Sleep hours', 'Social hours', 'Physically active hours']
colors = ['b', 'g', 'm', 'c']

threshold = df['grades'].median()
low_grades = df[df['grades'] <= threshold]
high_grades = df[df['grades'] > threshold]

fig, axs = plt.subplots(1, len(ivs), figsize=(20,5), tight_layout=True)
for ax, col, label, c in zip(axs, ivs, xlabels, colors):
    if len(low_grades) > 0:
        ax.scatter(low_grades[col], low_grades['grades'], alpha=0.3, color='b', label='Low grades')
        plot_regression_line(ax, low_grades[col].values, low_grades['grades'].values, color='b', lw=2)
        r_low = np.corrcoef(low_grades[col], low_grades['grades'])[0,1]
        ax.text(0.05, 0.95, f'Low r={r_low:.2f}', transform=ax.transAxes, color='b', fontsize=10, verticalalignment='top')
    if len(high_grades) > 0:
        ax.scatter(high_grades[col], high_grades['grades'], alpha=0.3, color='r', label='High grades')
        plot_regression_line(ax, high_grades[col].values, high_grades['grades'].values, color='r', lw=2)
        r_high = np.corrcoef(high_grades[col], high_grades['grades'])[0,1]
        ax.text(0.05, 0.85, f'High r={r_high:.2f}', transform=ax.transAxes, color='r', fontsize=10, verticalalignment='top')
    ax.set_xlabel(label)
    ax.set_ylabel('Grades' if col==ivs[0] else '')
    ax.legend()
plt.show()


# Seperating the low and high scoring students does not seem to add any info to the analysis.
# 
# Let's make a plot that show all of the plots to display in tha main file.

# In[11]:


def plot_descriptive():
    """
    Generates two separate figures, one below the other, by executing 
    the two plotting blocks sequentially.
    """
    
    # --- PLOT 1: Scatter plots with single regression line for all variables ---
    
    fig, axs = plt.subplots(1, 5, figsize=(20,5), tight_layout=True)

    ivs = [df['studyhours'].values,
           df['echours'].values,
           df['sleephours'].values,
           df['socialhours'].values,
           df['activityhours'].values]

    colors = ['b', 'r', 'g', 'm', 'c']

    for ax, x, c in zip(axs, ivs, colors):
        ax.scatter(x, df['grades'].values, alpha=0.2, color=c)
        plot_regression_line(ax, x, df['grades'].values, color='k', ls='-', lw=2)
        r = corrcoeff(x, df['grades'].values)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c,
                transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    xlabels = ['Study hours', 'Extracurricular hours', 'Sleep hours', 'Social hours', 'Physically active hours']
    [ax.set_xlabel(s) for ax, s in zip(axs, xlabels)]
    axs[0].set_ylabel('Grades')
    [ax.set_yticklabels([]) for ax in axs[1:]]
    plt.show()

    # --- PLOT 2: Scatter plots with two regression lines (High/Low Grades) ---
    
    # Redefining imports inside the function is unnecessary if they're in previous cells, 
    # but I'll keep the logic as provided in your second block.
    # import matplotlib.pyplot as plt # Already imported in notebook setup
    # import numpy as np # Already imported in notebook setup

    ivs = ['echours', 'sleephours', 'socialhours', 'activityhours']
    xlabels = ['Extracurricular hours', 'Sleep hours', 'Social hours', 'Physically active hours']
    colors = ['b', 'g', 'm', 'c']

    threshold = df['grades'].median()
    low_grades = df[df['grades'] <= threshold]
    high_grades = df[df['grades'] > threshold]

    fig, axs = plt.subplots(1, len(ivs), figsize=(20,5), tight_layout=True)
    for ax, col, label, c in zip(axs, ivs, xlabels, colors):
        if len(low_grades) > 0:
            ax.scatter(low_grades[col], low_grades['grades'], alpha=0.3, color='b', label='Low grades')
            plot_regression_line(ax, low_grades[col].values, low_grades['grades'].values, color='b', lw=2)
            r_low = np.corrcoef(low_grades[col], low_grades['grades'])[0,1]
            ax.text(0.05, 0.95, f'Low r={r_low:.2f}', transform=ax.transAxes, color='b', fontsize=10, verticalalignment='top')
        if len(high_grades) > 0:
            ax.scatter(high_grades[col], high_grades['grades'], alpha=0.3, color='r', label='High grades')
            plot_regression_line(ax, high_grades[col].values, high_grades['grades'].values, color='r', lw=2)
            r_high = np.corrcoef(high_grades[col], high_grades['grades'])[0,1]
            ax.text(0.05, 0.85, f'High r={r_high:.2f}', transform=ax.transAxes, color='r', fontsize=10, verticalalignment='top')
        ax.set_xlabel(label)
        ax.set_ylabel('Grades' if col==ivs[0] else '')
        ax.legend()
    plt.show()
    
# Example usage:
plot_descriptive()

import matplotlib.pyplot as plt
if __name__ == "__main__":
    display_central_tendency_table(num=1)
    display_dispersion_table(num=2)
    plot_descriptive()


