**This script forecasts grocery retailer sales using linear methods**

Data for this analysis may be found here: https://www.kaggle.com/competitions/store-sales-time-series-forecasting


```python
#Import libraries and data
import pandas as pd 
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as pth 
import matplotlib.dates as mdate
import matplotlib.patches as patches
import plotly.express as px
import kaleido
from random import randint

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```


```python
data_dir   		= 'data/store_sales/'
holidays_df 	= pd.read_csv(data_dir + 'holidays_events.csv')
oil_df     		= pd.read_csv(data_dir + 'oil.csv')
stores_df  		= pd.read_csv(data_dir + 'stores.csv')
transactions_df = pd.read_csv(data_dir + 'transactions.csv')
train_df		= pd.read_csv(data_dir + "train.csv")
test_df			= pd.read_csv(data_dir + "test.csv")
```

**Exploratory Data Analysis** *Part 1: Holidays*: Variable name and type exploration


```python
print("Columns in Holidays:",[x for x in holidays_df.columns])
#print("Holiday types:",pd.unique(holidays_df['date']))
# Dates are in format YYYY-MM-DD, get unique days, months and years instead of unique dates
days   = pd.unique([x.split('-')[1] + '-' + x.split('-')[2]  for x in holidays_df['date']]); days.sort()
months = pd.unique([x.split('-')[1] for x in holidays_df['date']]); months.sort()
years  = pd.unique([x.split('-')[0] for x in holidays_df['date']]); years.sort()

print("Holiday types:",pd.unique(holidays_df['type']))
print("Holiday types:",pd.unique(holidays_df['locale']))
print("Holiday types:",pd.unique(holidays_df['locale_name']))
print("Holiday types:",pd.unique(holidays_df['description'])[0:10],'...')
print("Holiday types:",pd.unique(holidays_df['transferred']))
```

    Columns in Holidays: ['date', 'type', 'locale', 'locale_name', 'description', 'transferred']
    Holiday types: ['Holiday' 'Transfer' 'Additional' 'Bridge' 'Work Day' 'Event']
    Holiday types: ['Local' 'Regional' 'National']
    Holiday types: ['Manta' 'Cotopaxi' 'Cuenca' 'Libertad' 'Riobamba' 'Puyo' 'Guaranda'
     'Imbabura' 'Latacunga' 'Machala' 'Santo Domingo' 'El Carmen' 'Cayambe'
     'Esmeraldas' 'Ecuador' 'Ambato' 'Ibarra' 'Quevedo'
     'Santo Domingo de los Tsachilas' 'Santa Elena' 'Quito' 'Loja' 'Salinas'
     'Guayaquil']
    Holiday types: ['Fundacion de Manta' 'Provincializacion de Cotopaxi'
     'Fundacion de Cuenca' 'Cantonizacion de Libertad'
     'Cantonizacion de Riobamba' 'Cantonizacion del Puyo'
     'Cantonizacion de Guaranda' 'Provincializacion de Imbabura'
     'Cantonizacion de Latacunga' 'Fundacion de Machala'] ...
    Holiday types: [False  True]
    

It seems like we have a list of holidays with dates. While certain named holidays may have a particular impact, I won't be able to determine that until I look at the sales data. For now I just want to see how the holidays are spread throughout the year and if there are changes by month and year.


```python
# Holidays -----------------

# Create a 12-month simple time series plot with years on Y axis, and days on X axis. Opacity will represent frequency if there is overlap
groups = years
subgroups = months
group_padding    = 1
subgroup_padding = 2

# Create plot, set axes and ticks based on group data
fig, plot = plt.subplots()
xticks = [int(x) for x in subgroups]
if subgroup_padding !=0:
	for x in range(0,subgroup_padding):
		xticks = [-1e99] + xticks + [-1e99]
yticks = [int(x) for x in groups]
if group_padding !=0:
	for x in range(0,group_padding):
		yticks = [-1e99] + yticks + [-1e99]
plt.xticks(xticks)
plt.yticks(yticks)
plot.set_xlim(int(subgroups[0])-subgroup_padding,int(subgroups[len(subgroups)-1])+subgroup_padding)
plot.set_ylim(int(groups[0])-group_padding,int(groups[len(groups)-1])+group_padding)

# Plot holidays with transferred as a different color as these were celebrated on a different day
untransfered_holidays_df = holidays_df.drop(holidays_df[holidays_df['transferred'] == True].index) # Remove transferred holidays
transfered_holidays_df   = holidays_df.drop(holidays_df[holidays_df['transferred'] == False].index) # Remove transferred holidays

dates = [x for x in untransfered_holidays_df['date']]
for z in dates:
	date_array = z.split('-')
	year  = float(date_array[0])
	month = float(date_array[1])
	day   = float(date_array[2])
	# print("Plotting",month + day/31,",",year)
	plt.plot(month + day/31,year,"o",color="b", alpha=.2) # Using day/31 as proportion of month (not accurate but should be good enough)
	# print(x)

dates = [x for x in transfered_holidays_df['date']]
for z in dates:
	date_array = z.split('-')
	year  = float(date_array[0])
	month = float(date_array[1])
	day   = float(date_array[2])
	#print("Plotting",month + day/31,",",year)
	plt.plot(month + day/31,year,"o",color="r", alpha=.2)
	# print(x)
plt.savefig('plots/store_calendar_holidays.png') #https://pythonguides.com/matplotlib-save-as-png/
plt.show()
```


    
![png]({{"images/time_series_images/output_6_0.png" | absolute_url}})
    


Interpretation: there seem to be a few interesting observations:
 - November and December holidays seem very consistent
 - Fewer holidays in 2012
 - April cluster in 2016
 - July cluster in 2014
 - Most transfers in October, but more varied transfers in 2017 and 2016
 
For a linear model, it seems like an interaction effect of month and year would help control for these effects

**Exploratory Data Analysis** *Part 2: Oil*: Variable name and type exploration


```python
print("Columns in oil:",[x for x in oil_df.columns])
print(oil_df.head())
```

    Columns in oil: ['date', 'dcoilwtico']
             date  dcoilwtico
    0  2013-01-01         NaN
    1  2013-01-02       93.14
    2  2013-01-03       92.97
    3  2013-01-04       93.12
    4  2013-01-07       93.20
    

The oil data seems pretty simple, basically just oil price by day with a few missing values. I think plotting the oil price over time may be sufficient to get a sense for this data. Additionally it may be worth exploring if there is a relationship to holidays for a sense of common economic factors that may have influenced both holiday celebration and oil prices that would not necessarily be visible in the data.


```python
# Plot without holidays overlayed
plt.plot(oil_df['date'], oil_df['dcoilwtico'])
plt.gca().xaxis.set_major_locator(mdate.YearLocator())
plt.savefig('plots/oil_prices_over_time.png'); plt.show()

#Plot with holidays overlayed
plt.plot(oil_df['date'], oil_df['dcoilwtico'])
plt.gca().xaxis.set_major_locator(mdate.YearLocator())


plt.plot(holidays_df['date'],[70 for x in holidays_df['date']],"o",color="b", alpha=.05) # Using day/31 as proportion of month (not accurate but should be good enough)


plt.savefig('plots/oil_prices_over_time_with_holidays.png'); plt.show()
```


    
![png]({{"images/time_series_images/output_11_0.png" | absolute_url}})
    



    
![png]({{"images/time_series_images/output_11_1.png" | absolute_url}})
    


The overall trend seems to be split into a high period and a low period, with a crash around late 2014/early 2015. Based on a visual inspection, there doesn't seem to be any relationship between frequency of holidays and oil prices.

**Exploratory Data Analysis** *Part 3: Stores*: Variable name and type exploration


```python
print("Columns in stores:",[x for x in stores_df.columns])
print(stores_df.head())
```

    Columns in stores: ['store_nbr', 'city', 'state', 'type', 'cluster']
       store_nbr           city                           state type  cluster
    0          1          Quito                       Pichincha    D       13
    1          2          Quito                       Pichincha    D       13
    2          3          Quito                       Pichincha    D        8
    3          4          Quito                       Pichincha    D        9
    4          5  Santo Domingo  Santo Domingo de los Tsachilas    D        4
    

The stores dataframe seems pretty simple. Controlling for specific store, city, state, type and cluster is likely to be sufficient to control for store and area-specific effects. Geographic data could be added to this data, especially if oil prices are considered and people may have been willing to travel more with lower oil prices though this is out of scope for the current attempt.

**Exploratory Data Analysis** *Part 4: Transactions*: Variable name and type exploration


```python
print("Columns in stores:",[x for x in transactions_df.columns])
print(transactions_df.head())
stores = [x for x in pd.unique(transactions_df['store_nbr'])]
print(stores)
```

    Columns in stores: ['date', 'store_nbr', 'transactions']
             date  store_nbr  transactions
    0  2013-01-01         25           770
    1  2013-01-02          1          2111
    2  2013-01-02          2          2358
    3  2013-01-02          3          3487
    4  2013-01-02          4          1922
    [25, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 54, 36, 53, 20, 29, 21, 42, 22, 52]
    

Seems pretty straightforward, but we may benefit from understanding how specific stores varied over time and how some stores compare to others. To help with this I will plot transactions over time by store.


```python
# Plot without holidays overlayed

fig = px.line(transactions_df,x='date',y='transactions',color='store_nbr', labels = {'x':'Date','y':'Number of Transactions'})
fig.update_traces(showlegend=True)
fig.show()
#fig.write_image("plots/store_transactions1.png")

```


    
![png]({{"images/time_series_images/output_12.png" | absolute_url}})



It seems like store growth varied by store over time, however there does not appear to be a global general positive or negative trend, but there does appear to be a strong cyclical trend. There are clear points where all stores seem to jump up dramatically. Because the number of stores is somewhat high, it is difficult to get a sense for the distribution of store sizes. For this reason, I will plot a histogram of total number of transactions in the dataset for each store to help understand this metric.


```python
store_transaction_sums = []

for x in stores:
    store_transactions = transactions_df.drop(transactions_df[transactions_df['store_nbr'] != x].index)
    store_transaction_sums.append(sum(store_transactions['transactions']))
store_transaction_sums.sort()

fig = px.histogram(store_transaction_sums, nbins=20, labels = {'x':'Transactions','y':'Number of Stores'})
fig.update_traces(showlegend=False)
fig.show()

```


    
![png]({{"images/time_series_images/output_13.png" | absolute_url}})


It seems that most stores have around 2 million transactions during a period with a few stores that have a much larger number of transactions.

**Exploratory Data Analysis** *Part 5: Training and Test Data*: Variable name and type exploration


```python
print("Columns in train:",[x for x in train_df.columns])
print(train_df.head())
print(pd.unique(train_df['onpromotion']))
```

    Columns in train: ['id', 'date', 'store_nbr', 'family', 'sales', 'onpromotion']
       id        date  store_nbr      family  sales  onpromotion
    0   0  2013-01-01          1  AUTOMOTIVE    0.0            0
    1   1  2013-01-01          1   BABY CARE    0.0            0
    2   2  2013-01-01          1      BEAUTY    0.0            0
    3   3  2013-01-01          1   BEVERAGES    0.0            0
    4   4  2013-01-01          1       BOOKS    0.0            0
    [  0   3   5   1  56  20  19   2   4  18  17  12   6   7  10   9  50   8
      16  42  51  13  15  47  21  40  37  54  24  58  22  59  11  45  25  55
      26  43  35  14  28  46  36  32  53  57  27  39  41  30  29  49  23  48
      44  38  31  52  33  34  61  60 116  86  73 113 102  68 104  93  70  92
     121  72 178 174 161 118 105 172 163 167 142 154 133 180 181 173 165 168
     186 140 149 145 169 188  62  84 111  65 107  63 101  87 125  94 114 171
     153 170 166 141 155 179 192 131 147 151 189  79  74 110  64  67  99 123
     157 117 150 182 162 160 194 135 190  69 108  89 126 156 103 146 132 177
     164 176 112  75 109  91 128 175 187 148 137 184 196 144 158 119 106  66
     100  90 120 115  98 159 152 185 139 143  80 124  71 134 193  78  88 122
     130  81  97 138 191  76  96 198  82  95 195 183 199 200 201 197  77  83
     136 205 204 202 129 206  85 209 211 207 208 203 210 127 213 212 218 216
     217 214 222 220 223 229 225 228 224 231 215 233 230 235 227 221 226 219
     289 245 609 261 322 276 710 511 326 281 718 551 304 639 489 299 243 630
     476 655 446 286 633 435 302 644 470 332 259 702 520 300 241 668 510 237
     626 507 317 624 474 240 672 306 600 383 293 258 646 444 333 279 717 342
     720 547 305 642 452 313 252 664 481 277 307 264 684 479 255 657 441 312
     269 716 528 320 285 726 536 283 628 469 464 677 420 247 629 424 290 473
     294 722 485 297 282 719 543 253 254 678 496 275 741 512 236 234 244 239
     238 242 232 249 250 251 391 539 411 486 407 330 697 246 467 263 591 248
     519 425]
    


```python
fig = px.histogram(train_df,x="family", y="sales", color = "onpromotion")
fig.update_traces(showlegend=False)
fig.show()

```


    
![png]({{"images/time_series_images/output_14.png" | absolute_url}})


It looks like there is wide variation in sales by category type and that being on promotion seems to account for a substantial percentage of sales though not necessarily the majority across categories. 

**Model Development**
The first model will be a very simple model. I want to start with year for time trends, monthly dummy variables for seasonality, and store numbers for unique effects to each store (some are generally larger than others).

- *holidays_df*: Not joining any variables because I believe that month and date are sufficient
- *oil_df*: The only effect that seemed to be relevant from this data seems to be captured by month and year as well
- *stores_df*: Contains regional information.
- *transactions_df*: We do not get any of this data for prediction, therefore it is not useful.


```python
train_df2 = pd.merge(train_df,stores_df,how="left", on=["store_nbr"])
test_df = pd.merge(test_df, stores_df, how="left", on=["store_nbr"])

train_df2['store_nbr'] = ["S" + str(x) for x in pd.Categorical(train_df['store_nbr'])]
test_df['store_nbr'] = ["S" + str(x) for x in pd.Categorical(test_df['store_nbr'])]

train_df2['onpromotion'] = pd.Categorical(train_df['onpromotion'])
train_df2['city'] = pd.Categorical(train_df2['city'])
train_df2['state'] = pd.Categorical(train_df2['state'])
train_df2['type'] = pd.Categorical(train_df2['type'])
train_df2['cluster'] = pd.Categorical(train_df2['cluster'])

train_df2['month'] = pd.Categorical([x.split('-')[1] for x in train_df2['date']])
train_df2['year'] = pd.Categorical([x.split('-')[0] for x in train_df2['date']])

test_df['month'] = pd.Categorical([x.split('-')[1] for x in test_df['date']])
test_df['year'] = pd.Categorical([x.split('-')[0] for x in test_df['date']])
```


```python
# Create dummy variables and fit model using a linear regression
# Using a simpler variable set for this model to reduce potential multicollinearity
train_df2x = train_df2
# dummy_columns = ['store_nbr','city','state','type','cluster','month','family']
dummy_columns = ['store_nbr','month']
train_df2f = pd.get_dummies(train_df2x[dummy_columns])
train_df2f[['year']] = train_df2x[['year']]
train_df2fx = train_df2f = train_df2f.loc[ : , train_df2f.columns != 'sales']

x_var_names = train_df2fx.columns
X = train_df2fx[train_df2fx.columns].to_numpy()
y = train_df2x['sales'].to_numpy()
model = LinearRegression().fit(X,y)
rsquared = r2_score(y, model.predict(X))
```


```python
# This function looks for dummy columns created by matching pd.get_dummies suffixed dummies
def get_created_dummies(df, dummy_roots):
    all_columns = [x for x in train_df2f.columns]
    created_dummies = []
    for x in all_columns:
        x_found = False
        for z in dummy_roots:
            if x[0:len(z)+1] == z+"_":
                x_found = True
        if x_found:
            created_dummies.append(x)
    return(created_dummies)

# This function looks at a dummy column and then looks for the root of that dummy in a list of columns converted to dummies
def dummy_root_match(dummy_column, column_list):
    for z in column_list:
        if dummy_column[0:len(z)+1] == z+"_":
            return z
        else:
            pass
#             print("No match for",dummy_column[0:len(z)+1],"and",z+"_")

# Because dummy variables are coded based on the test data frame, dummy values in the test data frame should match the test dataframe
def calculate_test_dummies(train_dfx, test_dfx, dummy_columns, dummy_roots):
    for x in dummy_columns:
        if randint(1, 10) == 10:
            test_dfx = test_dfx.copy()
        test_dfx[x] = [0 for x in test_dfx[test_dfx.columns[0]]] # Set initial value to 0
        root_column = dummy_root_match(x,dummy_roots)
        column_value = x[len(root_column)+1:] # substring the name of the dummy variable to get associated value to match
        test_dfx[x] = np.where(test_dfx[root_column] == column_value, 1, 0)
    return test_dfx
```


```python
# Recreate dummy columns to match training data and prepare dataframe for prediction generation
full_dummy_columns = get_created_dummies(train_df2f, test_df)

test_df_with_dummies = calculate_test_dummies(train_df2f, test_df, full_dummy_columns, dummy_columns)
X_test = test_df_with_dummies[train_df2fx.columns].to_numpy()

train_df_with_dummies = calculate_test_dummies(train_df2f, train_df2, full_dummy_columns, dummy_columns)
X_train_p = train_df_with_dummies[train_df2fx.columns].to_numpy()
```


```python
# Create predictions for test and train data
test_df['predicted_sales'] = [-1 for x in test_df[test_df.columns[0]]]
train_df['predicted_sales'] = [-1 for x in train_df[train_df.columns[0]]]

# This is a very slow way to do this
for i in range(len(test_df[test_df.columns[0]])):
    test_df['predicted_sales'][i] = model.predict(X_test[i].reshape(1, -1))
    
for i in range(len(train_df[train_df.columns[0]])):
    train_df['predicted_sales'][i] = model.predict(X_train_p[i].reshape(1, -1))

print(train_df[['predicted_sales','sales']])
```

             predicted_sales     sales
    0              97.591717     0.000
    1              97.591717     0.000
    2              97.591717     0.000
    3              97.591717     0.000
    4              97.591717     0.000
    ...                  ...       ...
    3000883       611.852906   438.133
    3000884       611.852906   154.553
    3000885       611.852906  2419.729
    3000886       611.852906   121.000
    3000887       611.852906    16.000
    
    [3000888 rows x 2 columns]
    


```python
print("R Squared",rsquared)
```

    R Squared 0.054255020962306055
    

Now we have some basic predictions. Not the greatest R^2, but we could probably improve the model in a few ways with some more work. Some of the things we could try:
 - Use variance inflation factor to re-add some of the other variables without introducing too much multicollinearity
 - Transform sales from raw dollars to help account for distribution/skewness
 - Drop some product categories that don't seem sufficiently similar to others
 
Apart from adjusting the base model, one thing to consider is that each product family really may need a separate model. For example, home appliances may have different seasonal trends than school supplies. 

**Approach 2: Separate Models for each family**


```python
train_df2 = pd.merge(train_df,stores_df,how="left", on=["store_nbr"])
train_df2['store_nbr'] = ["S" + str(x) for x in pd.Categorical(train_df['store_nbr'])]

train_df2['onpromotion'] = pd.Categorical(train_df['onpromotion'])
train_df2['city'] = pd.Categorical(train_df2['city'])
train_df2['state'] = pd.Categorical(train_df2['state'])
train_df2['type'] = pd.Categorical(train_df2['type'])
train_df2['cluster'] = pd.Categorical(train_df2['cluster'])

train_df2['month'] = pd.Categorical([x.split('-')[1] for x in train_df2['date']])
train_df2['year'] = pd.Categorical([x.split('-')[0] for x in train_df2['date']])

test_df['month'] = pd.Categorical([x.split('-')[1] for x in test_df['date']])
test_df['year'] = pd.Categorical([x.split('-')[0] for x in test_df['date']])

categories = pd.unique(train_df2['family'])

# Trying approach: Create a separate model for each category
models = {}
rsquareds = {}
for category_x in categories:
    train_df2x = train_df2.drop(train_df2[train_df2['family'] != category_x].index) # Remove observations from other categories

    # Create dummy variables and select columns
#     dummy_columns = ['store_nbr','city','state','type','cluster','month']
    dummy_columns = ['store_nbr','month']
    train_df2f = pd.get_dummies(train_df2x[dummy_columns])
#     train_df2f[['month','year','onpromotion']] = train_df2x[['month','year','onpromotion']]
    train_df2f[['year']] = train_df2x[['year']]
    train_df2fx = train_df2f = train_df2f.loc[ : , train_df2f.columns != 'sales']

    # Fit the model and create predictions
    x_var_names = train_df2fx.columns
    X = train_df2fx[train_df2fx.columns].to_numpy()
    y = train_df2x['sales'].to_numpy()
    models[category_x] = LinearRegression().fit(X,y)
    rsquareds[category_x] = r2_score(y, models[category_x].predict(X))
print(rsquareds)
```

    {'AUTOMOTIVE': 0.43898906148165007, 'BABY CARE': 0.07841997845659898, 'BEAUTY': 0.5073389778291613, 'BEVERAGES': 0.6748703072361368, 'BOOKS': 0.08179405173987131, 'BREAD/BAKERY': 0.7323304307667908, 'CELEBRATION': 0.26914229845858084, 'CLEANING': 0.6442212760054946, 'DAIRY': 0.7827037141250968, 'DELI': 0.6743269848007809, 'EGGS': 0.6350315452824302, 'FROZEN FOODS': 0.34466350410268354, 'GROCERY I': 0.6518294043807398, 'GROCERY II': 0.48418241228733483, 'HARDWARE': 0.20163560031796657, 'HOME AND KITCHEN I': 0.26325563474185887, 'HOME AND KITCHEN II': 0.2891739869876374, 'HOME APPLIANCES': 0.24353243307589012, 'HOME CARE': 0.513491883887448, 'LADIESWEAR': 0.39931777345945063, 'LAWN AND GARDEN': 0.36065207206802263, 'LINGERIE': 0.30718346435851984, 'LIQUOR,WINE,BEER': 0.28772608031935676, 'MAGAZINES': 0.4354899201011696, 'MEATS': 0.4198228985644261, 'PERSONAL CARE': 0.5659734771531607, 'PET SUPPLIES': 0.49505063729906473, 'PLAYERS AND ELECTRONICS': 0.4464489892586293, 'POULTRY': 0.7178383796923353, 'PREPARED FOODS': 0.8333714638487404, 'PRODUCE': 0.5601530267908643, 'SCHOOL AND OFFICE SUPPLIES': 0.08109508170175705, 'SEAFOOD': 0.8328586076248261}
    


```python
full_dummy_columns = get_created_dummies(train_df2f, test_df)

test_df_with_dummies = calculate_test_dummies(train_df2f, test_df, full_dummy_columns, dummy_columns)
X_test = test_df_with_dummies[train_df2fx.columns].to_numpy()

train_df_with_dummies = calculate_test_dummies(train_df2f, train_df2, full_dummy_columns, dummy_columns)
X_train_p = train_df_with_dummies[train_df2fx.columns].to_numpy()


test_df['predicted_sales'] = [-1 for x in test_df[test_df.columns[0]]]
train_df['predicted_sales'] = [-1 for x in train_df[train_df.columns[0]]]

# This is a very slow way to do this
for i in range(len(test_df[test_df.columns[0]])):
    test_df['predicted_sales'][i] = models[test_df['family'][i]].predict(X_test[i].reshape(1, -1))
    
for i in range(len(train_df[train_df.columns[0]])):
    train_df['predicted_sales'][i] = models[train_df['family'][i]].predict(X_train_p[i].reshape(1, -1))

print(train_df[['predicted_sales','sales']])
```

             predicted_sales     sales
    0               1.395264     0.000
    1              -0.161312     0.000
    2               0.460705     0.000
    3             264.152344     0.000
    4               0.011611     0.000
    ...                  ...       ...
    3000883       551.534180   438.133
    3000884        96.911743   154.553
    3000885      2241.740234  2419.729
    3000886        18.871490   121.000
    3000887        17.350037    16.000
    
    [3000888 rows x 2 columns]
    

Now we have R Squared values that are much better and predicted sales for each store that seem much closer to their product category. In addition to the approaches that we could use to improve the simpler base model that we use for each product category, we could truncate the predictions at 0 since it is not possible to have a negative prediction.  
