# 1. Build data preprocessing pipeline:
#     store
# 1. Explore and select promising models using cross_val_score until "GridSearch":
#     store before fitting as GridSearch to recalculate hyperparameters and parameters
#     store .best_estimator- after fitting as model recalculate parameters

# 2. creating a single pipeline that does the full data preparation plus the final prediction:
#     * preprocessing including feature selection 
#     * predicting
    

# store only fited model (whole model, not only .best_estimator):
# ##difference:
# #loaded_gridsearch.fit() # fit hyperparameters and parameters
# #loaded_gridsearch.best_estimator_.fit() # fit parameters

# use loaded_gridsearch.best_params_ to print learned hyperparameter



import pandas as pd
import numpy as np

df=pd.read_pickle("./data.pkl")


## Take a Quick Look at the Data Structure

df.head()

df.plot(figsize=(16,9))

df.shape

# search for rows with NaN
df[df.isnull().any(1)].head()

df.shape

df = df.dropna()

df.info()

df.describe()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
#plt.savefig("./attribute_histogram_plots.png")
plt.show()

## Create a Test Set
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# stratified sampling: eusure that test set is representative of whole dataset

# categorize very important coninuous numerical featuer and apply statified sampling,
# so that this featuere´s proportions in the test set are almost identical to those in the full dataset

df["temp"].hist()

df["temp_cat"]=pd.cut(df["temp"],
                            bins=[-1, 0, 5, 10, 15, 20, np.inf],
                            labels=[1,2,3,4,5, 6])

df["temp_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["temp_cat"]):
    strat_train_set = df.iloc[train_index]
    strat_test_set = df.iloc[test_index]

strat_test_set["temp_cat"].value_counts() / len(strat_test_set)

df["temp_cat"].value_counts() / len(df)

for set_ in (strat_train_set, strat_test_set, df):
    set_.drop("temp_cat", axis=1, inplace=True)

## Discover and Visualize the Data to Gain Insights
# explore copy of training set!
# possible insights: we need to drop some outliners

ireland = strat_train_set.copy()

ireland.head()

ireland.plot(kind='scatter', x='temp', y='price_da_sem', alpha=0.1, figsize=(16,9)) # set alpha to visualize density of data points
#plt.savefig("./scatter_plot.png")

ireland.loc[ireland['price_da_sem']>300,'price_da_sem']=300
ireland.max()

ireland.plot(kind="scatter", x="onshore_sem", y="load_sem", alpha=0.4,
    s=ireland["temp"]**2, label="temperature", figsize=(16,9),
    c="price_da_sem", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()

## Pearson´s r (standard correlation coefficient)
# correlation coefficient only measures linear correlation, it misses out on nonlinear relationships

ireland.corr().round(3)

ireland.corr()['price_da_sem'].round(3).sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ['price_da_sem','load_sem','onshore_sem', 'load_gb', 'solar_gb', 'wind_gb']
scatter_matrix(ireland[attributes], figsize=(16,9))
plt.show()

import seaborn as sns
sns.set(font_scale=1.2)
fig = sns.pairplot(data=ireland[attributes], height=3.5)

import matplotlib.style as style
style.use('default')

## Experimenting with Attribute Combinations

ireland.columns

ireland['residual_sem'] = ireland['load_sem']-ireland['onshore_sem']-ireland['rad_diffuse']-ireland['rad_direct']
ireland['residual_gb'] = ireland['load_gb']-ireland['wind_gb']-ireland['solar_gb']
ireland['residual_sem_fc'] = ireland['load_forecast_sem']-ireland['onshore_sem']-ireland['rad_diffuse']-ireland['rad_direct']

ireland.corr()['price_da_sem'].round(3).sort_values(ascending=False)

ireland.plot(kind="scatter", x="load_sem", y="load_gb", alpha=0.4,
    s=ireland['onshore_sem'], label="onshore", figsize=(16,9),
    c="price_da_sem", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
#plt.savefig("./scatter_plot.png")

# exploration is an iterative process
# once you get a prototype running you can analyze its output to gain more insights and come back to this exploration step

## Prepare the Data for Machine Learning Algorithms
# revert to clean training set

# separate predictors and labels, since we don´t necessarily want to apply the same transformations to the predictors
# and the target values!!!!!!!!!!!!!

ireland=strat_train_set.drop("price_da_sem", axis=1)
ireland_labels=strat_train_set["price_da_sem"].copy()

#ireland.to_pickle("./ireland.pkl")
#ireland_labels.to_pickle("./ireland_labels.pkl")

ireland = pd.read_pickle("./ireland.pkl")
ireland_labels = pd.read_pickle("./ireland_labels.pkl")

## Data Cleaning - class SimpleImputer

# SimpleImputer takes care of missing values:
# replaces each attribute´s missing values with the median of that attribute

ireland[ireland.isnull().any(axis=1)]

# without SimpleImputer
#median = housing["total_bedrooms"].median()
#sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# create copy of data without any text attributes as median only possible for numerical attributes

imputer.fit(ireland)

imputer.strategy #estimator´s hyperparameter

imputer.statistics_ #estimator´s learned parameter

ireland.median().values

X = imputer.transform(ireland) #NumPy array

ireland_tr = pd.DataFrame(X, columns=ireland.columns,
                          index=ireland.index)

## Custom Transformers

np.c_[np.array([1,2,3]), np.array([4,5,6])]

ireland.head()



from sklearn.base import BaseEstimator, TransformerMixin

# column index of attributes we need to create new attributes
onshore_sem_ix,load_forecast_sem_ix,load_sem_ix,load_gb_ix,wind_gb_ix,solar_gb_ix,rad_direct_ix,rad_diffuse_ix=0,1,2,3,4,5,7,8

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, residual_sem_fc = True): # no *args or **kargs
        self.residual_sem_fc = residual_sem_fc
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        residual_sem = X[:, load_sem_ix] - X[:, onshore_sem_ix] - X[:, rad_diffuse_ix] - X[:, rad_direct_ix]
        residual_gb = X[:, load_gb_ix] - X[:, wind_gb_ix] - X[:, solar_gb_ix]
        if self.residual_sem_fc:
            residual_sem = X[:, load_forecast_sem_ix] - X[:, onshore_sem_ix] - X[:, rad_diffuse_ix] - X[:, rad_direct_ix]
            return np.c_[X, residual_sem, residual_gb,
                         residual_sem]
        else:
            return np.c_[X, residual_sem, residual_gb]

attr_adder = CombinedAttributesAdder(residual_sem_fc=False)
ireland_extra_attribs = attr_adder.transform(ireland.values)

ireland_extra_attribs

ireland_extra_attribs = pd.DataFrame(
    ireland_extra_attribs,
    columns=list(ireland.columns)+["residual_sem", "residual_gb"],
    index=ireland.index)
ireland_extra_attribs.head()


## Pipeline for preprocessing the numerical attributes

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(residual_sem_fc=False)),
        ('std_scaler', StandardScaler()),
    ])

ireland_num_tr = num_pipeline.fit_transform(ireland)
ireland_num_tr


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_attribs = list(ireland)
cat_attribs = []

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

ireland_prepared = full_pipeline.fit_transform(ireland)

ireland_prepared

# -> only fit_transform attributes, not labels
# -> fit_transform training data as well as test data

## Select and train a model

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(ireland_prepared, ireland_labels)
ireland_predictions = lin_reg.predict(ireland_prepared)


from sklearn.metrics import mean_squared_error, mean_absolute_error

lin_mse = mean_squared_error(ireland_labels, ireland_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_mae = mean_absolute_error(ireland_labels, ireland_predictions)
lin_rmse, lin_mae

from sklearn.model_selection import cross_val_score

scores = cross_val_score(lin_reg, ireland_prepared, ireland_labels,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)

pd.Series(lin_rmse_scores).describe()

# 17.33 > 17.27
# 
# 
# -> slightly underfitting


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(ireland_prepared, ireland_labels)
ireland_predictions = forest_reg.predict(ireland_prepared)

forest_mse = mean_squared_error(ireland_labels, ireland_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse, forest_mse

forest_scores = cross_val_score(forest_reg, ireland_prepared, ireland_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

pd.Series(forest_rmse_scores).describe()

# 5.54 > 14.9
# 
# 
# -> overfitting!!

## GridSearch

### random forest model

# To check which parameters are needed in the pipeline:
forest_reg.get_params().keys()

from sklearn.model_selection import GridSearchCV

param_grid_forest = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}
    # then try 6 (2×3) combinations with bootstrap set as False
  ]

#forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search_forest = GridSearchCV(forest_reg, param_grid_forest, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search_forest.fit(ireland_prepared, ireland_labels)

grid_search_forest.best_params_

grid_search_forest.best_estimator_

grid_search_forest.best_estimator_.feature_importances_ #attribute bei forest_reg

feature_importances = grid_search_forest.best_estimator_.feature_importances_ #attribute bei forest_reg

cvres = grid_search_forest.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

extra_attribs = ["residual_sem", "residual_gb"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
#cat_encoder = full_pipeline.named_transformers_["cat"]
#cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs #+ cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

### Linear regression model

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(PolynomialFeatures(2), lin_reg)

pipe.get_params().keys()

param_grid = [
    # try 2 combinations of hyperparameters
    {'linearregression__fit_intercept': [True,False]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'linearregression__fit_intercept': [True,False], 'polynomialfeatures__degree':[1,2,3]},
  ]

# train across 5 folds, that's a total of (2+6)*5=40 rounds of training 
grid_search = GridSearchCV(pipe, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(ireland_prepared, ireland_labels)

grid_search.best_params_

final_model = grid_search.best_estimator_

final_model

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

grid_search.best_estimator_.named_steps['linearregression']

grid_search.best_estimator_.named_steps['linearregression'].coef_

grid_search.best_estimator_.named_steps['polynomialfeatures'].get_feature_names()

grid_search.predict(ireland_prepared)


## Evaulation on test set

strat_test_set.head()

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("price_da_sem", axis=1)
y_test = strat_test_set["price_da_sem"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse

# We can compute a 95% confidence interval for the test RMSE:

from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

# We could compute the interval manually like this: (as learned in edx course)

m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

# Alternatively, we could use a z-scores rather than t-scores: (as learned in edx course)

zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)

## instead of GridSearchCV also RandomizedSearchCV possible:

#from sklearn.model_selection import RandomizedSearchCV
#from scipy.stats import randint

#param_distribs = {
#        'n_estimators': randint(low=1, high=200),
#        'max_features': randint(low=1, high=8),
#    }

#forest_reg = RandomForestRegressor(random_state=42)
#rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
#rnd_search.fit(housing_prepared, housing_labels)

#cvres = rnd_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(np.sqrt(-mean_score), params)

## Complete Pipeline

full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("forest_reg", RandomForestRegressor(random_state=42))
    ])

full_pipeline_with_predictor.named_steps.preparation

# To check which parameters are needed in the pipeline:
full_pipeline_with_predictor.get_params().keys()

from sklearn.model_selection import GridSearchCV

param_grid_forest_complete = [
    {'preparation__num__attribs_adder__residual_sem_fc' : [True,False], \
     'forest_reg__n_estimators': [3, 10, 30], 'forest_reg__max_features': [2, 4, 6, 8]}
  ]

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search_forest_complete = GridSearchCV(full_pipeline_with_predictor, param_grid_forest_complete, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search_forest_complete.fit(ireland, ireland_labels)

grid_search_forest_complete.best_estimator_.named_steps

grid_search_forest_complete.best_estimator_.named_steps['preparation']

grid_search_forest_complete.best_estimator_.named_steps['forest_reg']

feature_importances = grid_search_forest_complete.best_estimator_.named_steps['forest_reg'].feature_importances_ #attribute bei forest_reg
feature_importances

cvres = grid_search_forest_complete.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

extra_attribs = ["residual_sem", "residual_gb"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
#cat_encoder = full_pipeline.named_transformers_["cat"]
#cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs #+ cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

attributes

feature_importances


final_model_complete = grid_search_forest_complete.best_estimator_ # model including trained hyperparameter

X_test = strat_test_set.drop("price_da_sem", axis=1)
y_test = strat_test_set["price_da_sem"].copy()

#X_test_prepared = full_pipeline_with_predictor.transform(X_test) #not needed as transformation part of pipeline
final_predictions = final_model_complete.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# if data preparation part of pipeline:
# -> fit and predict pure data, not prepared data 
# see Exercise solution 4. in 02_end_to_end_machine_learning

final_rmse


## Store everything

# data
# pipeline with default values: only useful for preprocessing
# best models to fit parameters (and hyperparameters if needed)
# complete preprocessing and prediction model to fit parameters (and hyperparameters if needed)
# rmse of complete model

df.to_pickle("./ml_df.pkl")

from sklearn.externals import joblib

# preprocessing pipeline
joblib.dump(full_pipeline, 'ml_preprocessing_pipeline.pkl')

# prediction model
joblib.dump(grid_search, 'ml_model_.pkl')

# complete preprocessing and prediction model
joblib.dump(grid_search_forest_complete, 'ml_complete_model.pkl')

joblib.dump(final_rmse, 'ml_rmse.pkl')

## How to use saved model

loaded_gridsearch = joblib.load('ml_complete_model.pkl')

loaded_gridsearch.best_params_

loaded_gridsearch.best_estimator_.named_steps['forest_reg']

loaded_gridsearch.best_estimator_.named_steps['forest_reg'].feature_importances_ 

### fit only new parameter: .best_estimator_.fit

loaded_gridsearch.best_estimator_.fit(ireland[:10], ireland_labels[:10]) #fit parameter

loaded_gridsearch.best_params_ #hyperparameters stayed the same

loaded_gridsearch.best_estimator_.named_steps['forest_reg']

loaded_gridsearch.best_estimator_.named_steps['forest_reg'].feature_importances_ #new parameters

### fit new hyperparamter and parameter: .fit

loaded_gridsearch.fit(ireland[:10], ireland_labels[:10]) #fit new hyperparameter and parameter

loaded_gridsearch.best_params_ #new hyperparameters

loaded_gridsearch.best_estimator_.named_steps['forest_reg']

loaded_gridsearch.best_estimator_.named_steps['forest_reg'].feature_importances_ #new parameters


# You can use both .predict() or .best_estimator_.predict() - result is the same

loaded_gridsearch.predict(X_test[:10])

loaded_gridsearch.best_estimator_.predict(X_test[:10])


X_test = strat_test_set.drop("price_da_sem", axis=1)
y_test = strat_test_set["price_da_sem"].copy()

#X_test_prepared = full_pipeline_with_predictor.transform(X_test) #not needed as transformation part of pipeline
final_predictions = loaded_gridsearch.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
