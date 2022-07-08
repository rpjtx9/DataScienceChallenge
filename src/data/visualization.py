import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn as sns
from .dataset import get_listing_price_dataframe
import pandas as pd

pd.set_option('display.max_row', None)

df = get_listing_price_dataframe()

def density_plot(ColumnName):
    # Get a list of cylinder sizes
    plot_data = df.dropna(subset = ['Dealer_Listing_Price'])
    plot_data = plot_data[ColumnName].value_counts()
    plot_data = list(plot_data.index)

    figsize(12,10)

    for datapoint in plot_data:
        subset = df[df[ColumnName] == datapoint]

        sns.kdeplot(subset['Dealer_Listing_Price'].dropna(), label = datapoint, shade = False, alpha = 0.8);

    plt.xlabel('Dealer Listing Price', size = 20); plt.ylabel('Density', size = 20);

    plt.title('Density Plot of Dealer Listing Price by ' + ColumnName,  size = 28);

    plt.legend()
    plt.show()

def scatter_plot(ColumnName):
    # Get a list of cylinder sizes
    plot_data = df.dropna(subset = ['Dealer_Listing_Price'])
    plot_data = plot_data[ColumnName]

    figsize(12,10)

    sns.scatterplot(data = df, x = df[ColumnName], y = df['Dealer_Listing_Price'])

    plt.xlabel(ColumnName, size = 20); plt.ylabel('Dealer_Listing_Price', size = 20);

    plt.title('Scatter Plot of Dealer Listing Price by ' + ColumnName,  size = 28);

    plt.legend()
    plt.show()

def get_price_correlation(x = 20):
    correlations = df.corr()['Dealer_Listing_Price'].sort_values()

    print(f"The {x} most negative correlations are: \n{correlations.head(x)}", '\n')

    correlations = df.corr()['Dealer_Listing_Price'].sort_values(ascending = False)

    print(f"The {x} most positive correlations are: \n{correlations.head(x)}" '\n')

get_price_correlation(100)