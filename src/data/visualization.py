import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn as sns
from .dataset import get_sales_dataframe

df = get_sales_dataframe()

print(df)