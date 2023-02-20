## Problem statement

# Build a machine learning model which can predict the strength of a mixture for given composition of ingredients like
# cement,slag,ash,water,superplastic,coarseagg,fineagg,age.

# Importing necessary libraries

# Dataframe manipulation and analysis libraries
import pandas as pd
import numpy as np

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries to filter warnings
import warnings

warnings.filterwarnings('ignore')

# Multicolinearity test package
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Data preparation libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Model evaluation libraries
from sklearn.metrics import r2_score, mean_squared_error

# Machine Learning models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost

# Feature decomposition library
from sklearn.decomposition import PCA

# Clustering Libraries
from sklearn.cluster import KMeans

# Recursive feature elemination library
from sklearn.feature_selection import RFE

# Learning curve analysis library
from sklearn.model_selection import learning_curve

# Loading the dataframe
df = pd.read_excel('Construction data.xlsx')

# Exploratory Data Analysis - EDA
print(df.head().to_string())
print(df.info())


def custom_summary(mydf):
    cols = []
    for i in df.columns:
        if mydf[i].dtype != object:
            cols.append(i)
    result = pd.DataFrame(columns=cols,
                          index=['Datatype', 'Count', 'Min', 'Q1', 'Q2', 'Q3', 'Max', 'Mean', 'Std_dev', 'Skew', 'Kurt',
                                 'Range', 'IQR', 'Skew_comment', 'Kurt_comment', 'Outlier_comment'])
    for i in result.columns:
        result.loc['Datatype'] = mydf[i].dtype
        result.loc['Count'] = mydf[i].count()
        result.loc['Min', i] = mydf[i].min()
        result.loc['Q1', i] = mydf[i].quantile(0.25)
        result.loc['Q2', i] = mydf[i].quantile(0.5)
        result.loc['Q3', i] = mydf[i].quantile(0.75)
        result.loc['Max', i] = mydf[i].quantile(1)
        result.loc['Mean', i] = round(mydf[i].mean(), 2)
        result.loc['Std_dev', i] = round(mydf[i].std(), 2)
        result.loc['Skew', i] = round(mydf[i].skew(), 2)
        result.loc['Kurt', i] = round(mydf[i].kurt(), 2)
        result.loc['Range', i] = mydf[i].max() - mydf[i].min()
        result.loc['IQR', i] = mydf[i].quantile(0.75) - mydf[i].quantile(0.25)

        if result.loc['Skew', i] <= -1:
            sk_label = 'Highly negatively skewed'
        elif -1 < result.loc['Skew', i] <= -0.5:
            sk_label = 'Moderately negatively skewed'
        elif -0.5 < result.loc['Skew', i] <= 0:
            sk_label = 'Approximately normally distributed(-ve)'
        elif 0 < result.loc['Skew', i] <= 0.5:
            sk_label = 'Approximately normally distributed(+ve)'
        elif 0.5 < result.loc['Skew', i] <= 1:
            sk_label = 'Moderately positively skewed'
        elif result.loc['Skew', i] >= 1:
            sk_label = 'Highly positively skewed'
        else:
            sk_label = 'error'
        result.loc['Skew_comment', i] = sk_label

        if result.loc['Kurt', i] < -1:
            kt_label = 'Highly platykurtic'
        elif -1 < result.loc['Kurt', i] <= -0.5:
            kt_label = 'moderataly platykurtic'
        elif -0.5 < result.loc['Kurt', i] <= 0.5:
            kt_label = 'mesokurtic curve'
        elif 0.5 < result.loc['Kurt', i] <= 1:
            kt_label = 'moderataly leptokurtic'
        elif result.loc['Kurt', i] > 1:
            kt_label = 'highly leptokurtic'
        else:
            kt_label = 'error'
        result.loc['Kurt_comment', i] = kt_label

        lw = result.loc['Q1', i] - (1.5 * result.loc['IQR', i])
        uw = result.loc['Q3', i] + (1.5 * result.loc['IQR', i])

        if len([x for x in mydf[i] if x < lw or x > uw]) > 0:

            outlier_label = 'Have outliers'

        else:

            outlier_label = 'No outliers'

        result.loc['Outlier_comment', i] = outlier_label

    return result


print(custom_summary(df).to_string())


# Checking for outliers using boxplot

def replace_outlier(mydf, col, method='Quartile', strategy='median'):
    if method == 'Quartile':
        Q1 = mydf[col].quantile(0.25)
        Q2 = mydf[col].quantile(0.50)
        Q3 = mydf[col].quantile(0.75)
        IQR = Q3 - Q1
        LW = Q1 - 1.5 * IQR
        UW = Q3 + 1.5 * IQR
    elif method == 'standard deviation':
        mean = mydf[col].mean()
        std = mydf[col].std()
        LW = mean - (2 * std)
        UW = mean + (2 * std)
    else:
        print('Pass a correct method')
    outliers = mydf.loc[(mydf[col] < LW) | (mydf[col] > UW), col]
    outliers_density = round(len(outliers) / len(mydf), 2)
    if len(outliers) == 0:
        print(f'feature {col} does not have any outliers')
    else:
        print(f'feature {col} has any outliers')
        print(f'Total number of outliers in this {col} is:'(len(outliers)))
        print(f'Outliers percentage in {col} is {outliers_density * 100}%')
    if strategy == 'median':

        mydf.loc[(mydf[col] < LW), col] = Q1
        mydf.loc[(mydf[col] > UW), col] = Q3
    elif strategy == 'mean':
        mydf.loc[(mydf[col] < LW), col] = mean
        mydf.loc[(mydf[col] > UW), col] = mean
    else:
        print('Pass the correct strategy')
    return mydf


def odt_plots(mydf, col):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))
    sns.boxplot(mydf[col], ax=ax1)
    ax1.set_title(col + ' boxplot')
    ax1.set_xlabel('values')
    ax1.set_ylabel('boxplot')
    mydf_out = replace_outlier(mydf, col)
    sns.boxplot(mydf_out[col], ax=ax2)
    ax2.set_title(col + 'boxplot')
    ax2.set_xlabel('values')
    ax2.set_ylabel('boxplot')
    plt.show()


for col in df.drop('strength', axis=1).columns:
    print(odt_plots(df, col))

# Multivariate analysis using regression
for col in df.columns:
    if col != 'strength':
        fig, ax1 = plt.subplots(figsize=(10, 5))
        print(sns.regplot(x=df[col], y=df['strength'], ax=ax1).set_title(f'relationship between {col} and sytrength'))
        plt.show()

# Multi-Colinearity check
# Stage 1 - Correlation heatmap
corr = df.corr()
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr, annot=True)
plt.show()


## Multicolinearity test
# Stage 2: Variane Inflating Factor(VIF)
#     formula for VIF = 1/(1-R2)
#         1. Regress every independent variable with each other and find the R2 score
#         2. find out VIF using above formula
#         3. if VIF is more than 5 for any independent variable we can conclude that multi-colinearity exist.

def VIF(independent_variables):
    vif = pd.DataFrame()
    vif['vif'] = [variance_inflation_factor(independent_variables.values, i) for i in
                  range(independent_variables.shape[1])]
    vif['independent_variables'] = independent_variables.columns
    vif = vif.sort_values(by=['vif'], ascending=False)  # to sort the values in descending order
    return vif


print(VIF(df.drop('strength', axis=1)))


# Correlation with target feature
def CWT(data, tcol):
    independent_variables = data.drop(tcol, axis=1).columns
    corr_result = []
    for col in independent_variables:
        corr_result.append(data[tcol].corr(data[col]))
    result = pd.DataFrame([independent_variables, corr_result],
                          index=['independent variables', 'correlation']).T  # T is for transpose
    return result.sort_values(by='correlation', ascending=False)


print(CWT(df, 'strength'))


# Principal component analysis
def PCA_1(x):
    n_comp = len(x.columns)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    for i in range(1, n_comp):
        pca = PCA(n_components=i)
        p_comp = pca.fit_transform(x)
        evr = np.cumsum(pca.explained_variance_ratio_)
        if evr[i - 1] > 0.9:
            n_components = i
            break
    print('Ecxplained varience ratio after pca is: ', evr)
    # creating a pcs dataframe
    col = []
    for j in range(1, n_components + 1):
        col.append('PC_' + str(j))
    pca_df = pd.DataFrame(p_comp, columns=col)
    return pca_df


transformed_df = PCA_1(df.drop('strength', axis=1))

print(transformed_df)

transformed_df = transformed_df.join(df['strength'], how='left')
print(transformed_df.head())


# Model Building
# 1. Train-test split
# 2. Cross-validation
# 3. Hyperparameter tuning

def train_and_test_split(data, t_col, testsize=0.3):
    x = data.drop(t_col, axis=1)
    y = data[t_col]
    return train_test_split(x, y, test_size=testsize, random_state=1)


def model_builder(model_name, estimator, data, t_col):
    x_train, x_test, y_train, y_test = train_and_test_split(data, t_col)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    accuracy = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return [model_name, accuracy, rmse]


def multiple_models(data, t_col):
    col_names = ['model_name', 'r2_score', 'RMSE']
    result = pd.DataFrame(columns=col_names)
    result.loc[len(result)] = model_builder('LinearRegression', LinearRegression(), data, t_col)
    result.loc[len(result)] = model_builder('Lasso', Lasso(), data, t_col)
    result.loc[len(result)] = model_builder('Ridge', Ridge(), data, t_col)
    result.loc[len(result)] = model_builder('DecisionTreeRegressor', DecisionTreeRegressor(), data, t_col)
    result.loc[len(result)] = model_builder('KneighborRegressor', KNeighborsRegressor(), data, t_col)
    result.loc[len(result)] = model_builder('RandomForestRegressor', RandomForestRegressor(), data, t_col)
    result.loc[len(result)] = model_builder('SVR', SVR(), data, t_col)
    result.loc[len(result)] = model_builder('AdaBoostRegressor', AdaBoostRegressor(), data, t_col)
    result.loc[len(result)] = model_builder('GradientBoostingRegressor', GradientBoostingRegressor(), data, t_col)
    result.loc[len(result)] = model_builder('XGBRegressor', XGBRegressor(), data, t_col)
    return result.sort_values(by='r2_score', ascending=False)


print(multiple_models(transformed_df, 'strength'))


# Cross Validation

def kfoldCV(x, y, fold=10):
    score_lr = cross_val_score(LinearRegression(), x, y, cv=fold)
    score_las = cross_val_score(Lasso(), x, y, cv=fold)
    score_ri = cross_val_score(Ridge(), x, y, cv=fold)
    score_dt = cross_val_score(DecisionTreeRegressor(), x, y, cv=fold)
    score_kn = cross_val_score(KNeighborsRegressor(), x, y, cv=fold)
    score_rf = cross_val_score(RandomForestRegressor(), x, y, cv=fold)
    score_svr = cross_val_score(SVR(), x, y, cv=fold)
    score_ab = cross_val_score(AdaBoostRegressor(), x, y, cv=fold)
    score_gb = cross_val_score(GradientBoostingRegressor(), x, y, cv=fold)
    score_xb = cross_val_score(XGBRegressor(), x, y, cv=fold)

    model_names = ['LinearRegression', 'Lasso', 'Ridge', 'DecisionTreeRegressor', 'KNeighborsRegressor',
                   'RandomForestRegressor', 'SVR', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'XGBRegressor']
    scores = [score_lr, score_las, score_ri, score_dt, score_kn, score_rf, score_svr, score_ab, score_gb, score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names, score_mean, score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result, columns=['model_names', 'cv_score', 'cv_std'])
    return kfold_df.sort_values(by='cv_score', ascending=False)


print(kfoldCV(transformed_df.drop('strength', axis=1), transformed_df['strength']))


# Hyperparameter Tuning
def tuning(x, y, fold=10):
    # parameters grids for different models
    param_las = {
        'alpha': [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40,
                  50, 60, 70, 80, 90, 100, 200, 300, 400, 500]}
    param_rd = {
        'alpha': [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40,
                  50, 60, 70, 80, 90, 100, 200, 300, 400, 500]}
    param_dtr = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 'max_depth': [3, 5, 7, 9, 11], 'max_features': [1, 2, 3, 4, 5, 6, 7, 'auto', 'log2', 'sqrt']}
    param_knn = {'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    param_svr = {'gamma': ['scale', 'auto'], 'C': [0.1, 1, 1.5, 2]}
    param_rf = {'max_depth': [3, 5, 7, 9, 11], 'max_features': [1, 2, 3, 4, 5, 6, 7, 'auto', 'log2', 'sqrt'],
                'n_estimators': [50, 100, 150, 200]}
    param_ab = {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.1, 0.5, 0.7, 1, 5, 10, 20, 50, 100]}
    param_gb = {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.1, 0.5, 0.7, 1, 5, 10, 20, 50, 100]}
    param_xb = {'eta': [0.1, 0.5, 10.7, 1, 5, 10, 20], 'max_depth': [3, 5, 7, 9, 10], 'gamma': [0, 10, 20, 50],
                'reg_lambda': [0, 1, 3, 5, 7, 10], 'alpha': [0, 1, 3, 5, 7, 10]}
    # Creating Model object
    tune_las = GridSearchCV(Lasso(), param_las, cv=fold)
    tune_rd = GridSearchCV(Ridge(), param_rd, cv=fold)
    tune_dtr = GridSearchCV(DecisionTreeRegressor(), param_dtr, cv=fold)
    tune_knn = GridSearchCV(KNeighborsRegressor(), param_knn, cv=fold)
    tune_svr = GridSearchCV(SVR(), param_svr, cv=fold)
    tune_rf = GridSearchCV(RandomForestRegressor(), param_rf, cv=fold)
    tune_ab = GridSearchCV(AdaBoostRegressor(), param_ab, cv=fold)
    tune_gb = GridSearchCV(GradientBoostingRegressor(), param_gb, cv=fold)
    tune_xb = GridSearchCV(XGBRegressor(), param_xb, cv=fold)
    # Model fitting
    tune_las.fit(x, y)
    tune_rd.fit(x, y)
    tune_dtr.fit(x, y)
    tune_knn.fit(x, y)
    tune_svr.fit(x, y)
    tune_rf.fit(x, y)
    tune_ab.fit(x, y)
    tune_gb.fit(x, y)
    tune_xb.fit(x, y)

    tune = [tune_rf, tune_xb, tune_gb, tune_las, tune_rd, tune_knn, tune_svr, tune_dtr, tune_ab]
    models = ['RF', 'XB', 'GB', 'lasso', 'RD', 'AB', 'KNN', 'SVR', 'DTR']
    for i in range(len(tune)):
        print('model:', models[i])
        print('Best_params:', tune[i].best_params_)


print(tuning(transformed_df.drop('strength', axis=1), transformed_df['strength']))


def cv_post_hpt(x, y, fold=10):
    score_lr = cross_val_score(LinearRegression(), x, y, cv=fold)
    score_las = cross_val_score(Lasso(alpha=0.1), x, y, cv=fold)
    score_rd = cross_val_score(Ridge(alpha=9), x, y, cv=fold)
    score_dt = cross_val_score(DecisionTreeRegressor(criterion='friedman_mse', max_depth=11, max_features=3), x, y,
                               cv=fold)
    score_kn = cross_val_score(KNeighborsRegressor(weights='distance', algorithm='brute'), x, y, cv=fold)
    score_rf = cross_val_score(RandomForestRegressor(max_depth=11, max_features='auto', n_estimators=150), x, y,
                               cv=fold)
    score_svr = cross_val_score(SVR(gamma='scale', C=2), x, y, cv=fold)
    score_ab = cross_val_score(AdaBoostRegressor(n_estimators=100, learning_rate=1), x, y, cv=fold)
    score_gb = cross_val_score(GradientBoostingRegressor(n_estimators=200, learning_rate=0.1), x, y, cv=fold)
    score_xb = cross_val_score(XGBRegressor(eta=0.1, max_depth=7, gamma=0, reg_lambda=1, alpha=7), x, y, cv=fold)

    model_names = ['LinearRegression', 'RandomForestRegressor', 'Lasso', 'Ridge', 'DecisionTreeRegressor',
                   'KNeighborsRegressor', 'SVR', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'XGBRegressor']
    scores = [score_lr, score_rf, score_las, score_rd, score_dt, score_kn, score_svr, score_ab, score_gb, score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names, score_mean, score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result, columns=['model_names', 'cv_score', 'cv_std'])
    return kfold_df.sort_values(by='cv_score', ascending=False)


print(cv_post_hpt(transformed_df.drop('strength', axis=1), transformed_df['strength']))

# Clusteriing
# Using clustering to check if it can help us improve the accuracy
labels = KMeans(n_clusters=2, random_state=2)
clusters = labels.fit_predict(df.drop('strength', axis=1))
sns.scatterplot(x=df['cement'], y=df['strength'], hue=clusters)


def clustering(x, tcol, clusters):
    column = list(set(list(x.columns)) - set(list('strength')))
    # column = list(x.column)
    r = int(len(column) / 2)
    if len(column) % 2 == 0:
        r = r
    else:
        r += 1  # same as r+1
    f, ax = plt.subplots(r, 2, figsize=(15, 15))
    a = 0
    for row in range(r):
        for col in range(2):
            if a != len(column):
                ax[row][col].scatter(x[tcol], x[column[a]], c=clusters)
                ax[row][col].set_xlabel(tcol)
                ax[row][col].set_ylabel(column[a])
                a += 1


x = df.drop('strength', axis=1)
for col in x.columns:
    clustering(x, col, clusters)

new_df = df.join(pd.DataFrame(clusters, columns=['cluster']), how='left')

new_f = new_df.groupby('cluster')['cement'].agg(['mean', 'median'])

cluster_df = new_df.merge(new_f, on='cluster', how='left')
print(cluster_df.head())

# Model Evaluation on clustered dataset
print(multiple_models(cluster_df, 'strength'))

print(kfoldCV(cluster_df.drop('strength', axis=1), cluster_df['strength']))

print(cv_post_hpt(cluster_df.drop('strength', axis=1), cluster_df['strength']))

# Feature Importance
x_train, x_test, y_train, y_test = train_and_test_split(cluster_df, 'strength')

xgb = XGBRegressor()
xgb.fit(x_train, y_train)

print(xgboost.plot_importance(xgb))

subset_df = cluster_df[['age', 'cement', 'water', 'coarseagg', 'fineagg', 'strength']]

print(cv_post_hpt(subset_df.drop('strength', axis=1), subset_df['strength']))

# RFE
new__df = cluster_df

rfe = RFE(estimator=XGBRegressor())

rfe.fit(new__df.drop('strength', axis=1), new__df['strength'])

print(rfe.support_)

print(new__df.columns)

final_df = cluster_df[['cement', 'slag', 'superplastic', 'age', 'strength']]

print(cv_post_hpt(final_df.drop('strength', axis=1), final_df['strength']))


# Learning curve ananylis
def generate_learning_curve(model_name, estimater, x, y):
    train_size, train_score, test_score = learning_curve(estimater, x, y, cv=10)
    train_score_mean = np.mean(train_score, axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    plt.plot(train_size, train_score_mean, c='blue')
    plt.plot(train_size, test_score_mean, c='red')
    plt.xlabel('Samples')
    plt.ylabel('Scores')
    plt.title('Learning curve for ' + model_name)
    plt.legend(('Training accuray', 'Testing accuracy'))


generate_learning_curve('Linear regression', LinearRegression(), cluster_df.drop('strength', axis=1),
                        cluster_df['strength'])

model_names = [LinearRegression(), Lasso(), Ridge(), SVR(), DecisionTreeRegressor(), KNeighborsRegressor(),
               RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), XGBRegressor()]
for i, model in enumerate(model_names):
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(5, 2, i + 1)
    generate_learning_curve(type(model).__name__, model, cluster_df.drop('strength', axis=1), cluster_df['strength'])
