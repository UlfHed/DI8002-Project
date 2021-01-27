import pandas as pd
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



def main():
    # ---------------------------------------- Read and init data. ---------------------------------------- #
    
    df = pd.read_csv('Datasets/ADM-cmiyc_data-small.csv', header=None, low_memory=False, skiprows=1)    # First row is the column names.
    df.columns = pd.read_csv('Datasets/ADM-cmiyc_data-small.csv', header=None, low_memory=False, nrows=1).values[0]

    print('Number of Features (Columns):', len(df.columns))
    print('Number of readings (Rows):', len(df.index))

    # ---------------------------------------- State Variables. ---------------------------------------- #

    run_train = True
    run_outliers = True
    run_clustering = True
    run_feature_selection = True

    # ---------------------------------------- Global Variables. ---------------------------------------- #
    
    global rstate, cv
    rstate = 42
    cv = 5

    # ---------------------------------------- Preprocessing. ---------------------------------------- #
    
    # Check & Remove missing values (rows).
    check_missing_values(df)
    df = df.dropna()   
    df = df.reset_index(drop=True)

    print(len(df))
    exit()

    # Convert string time to timestamp.
    df = string_timestamp(df)

    # Encode label values
    df.rename(columns = {'target': 'label'}, inplace=True)
    df['label_encoded'] = np.where(df['label'] == 'Other', 0, df['label'])    # Normal is 0.
    df['label_encoded'] = np.where(df['label_encoded'] != 0, 1, df['label_encoded'])    # Not normal (0) is attack.
    df['label_encoded'] = df['label_encoded'].astype('int')
    class_map = {0: 'Other', 1: 'Alice'}

    # Label frequency.
    af = label_frequency(df)
    plot_bar_label_frequency(af, 'Alice vs. Other', 'State', 'Frequency', 'Images/CMIYC/Label_Frequency_BAR_CMIYC.png')
    plot_pie_label_frequency(af, 'Alice vs. Other', 'Images/CMIYC/Label_Frequency_PIE_CMIYC.png')

    # Outliers.
    if run_outliers == True:
        df_outliers = outliers(df)
        save('Data/CMIYC/outlier_data_CMIYC', df_outliers)
    df_outliers = load('Data/CMIYC/outlier_data_CMIYC')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('Outlier Data:\n%s'%(df_outliers))
    plot_bar_outliers(df_outliers, 'Feature Outliers (Fliers)', 'Number of Fliers', 'Feature', 'Images/CMIYC/Feature_Outliers_BAR_CMIYC.png')

    # PCA Reduction [Can't run scikit-learn PCA in native Python, see Jupyter Notebook].
    pca_components = load('Reduction/CMIYC/PCA_CMIYC')
    dfpca = pd.DataFrame()
    dfpca['X'] = pca_components[:, 0]
    dfpca['Y'] = pca_components[:, 1]
    dfpca['Z'] = pca_components[:, 2]
    dfpca['label'] = df['label'].values
    dfpca['label_encoded'] = df['label_encoded'].values
    plot_3d(dfpca, 'PCA - 3 Principle Components', 'PC 1', 'PC 2', 'PC 3', 'Images/CMIYC/PCA_Scatter_CMIYC.png')

    # TSNE Reduction [Can't run scikit-learn TSNE in native Python, see Jupyter Notebook].
    tsne_result = load('Reduction/CMIYC/TSNE_CMIYC')
    dftsne = pd.DataFrame()
    dftsne['X'] = tsne_result[:, 0]
    dftsne['Y'] = tsne_result[:, 1]
    dftsne['Z'] = tsne_result[:, 2]
    dftsne['label'] = df['label'].values
    dftsne['label_encoded'] = df['label_encoded'].values
    plot_3d(dftsne, 't-SNE - 3 Dimensions, perplexity=30', 'X', 'Y', 'Z', 'Images/CMIYC/TSNE_Scatter_CMIYC.png')

    # ---------------------------------------- Models. ---------------------------------------- #

    datasets = [df, dfpca, dftsne]  # Actually [df, mif, dfpca, dftsne].
    dataset_names = ['Full Dataset', 'Most Important Features', 'PCA Reduction', 't-SNE Reduction']
    for i in range(len(dataset_names)):
        data = datasets[i]
        dataset_name = dataset_names[i]
        print('\n'+'> '*10+'DATA: %s'%(dataset_name)+' <'*10)

        if dataset_name == 'Most Important Features':
            X_train, X_test, y_train, y_test, X, y = data['X_train'], data['X_test'], data['y_train'], data['y_test'], data['X'], data['y']
            # This data is already scaled from the run of Full Dataset.
        else:
            # Separate data and label.
            X = np.array(data.drop(['label', 'label_encoded'], axis=1))
            y = np.array(data['label_encoded'])            
            # 80% Training, 20% testing.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rstate)

            # Scale Data.
            X_train = mms_scale(X_train)
            X_test = mms_scale(X_test)

        # ---------------------------------------- Features Selection. ---------------------------------------- #
        if run_feature_selection == True and dataset_name == 'Full Dataset':
            n_trees = 1000
            clf_ETC = ETC_train(X_train, y_train, n_trees)
            important_features = pd.Series(clf_ETC.feature_importances_, index=data.drop(['label', 'label_encoded'], axis=1).columns).sort_values(ascending=False)
            important_features_1p = important_features[important_features > 0.01]   # Only features above 1% importance.    
            # Bar Graphs. 
            plot_bar_important_features(important_features, 'Feature Importance (%s features) (%s Trees)'%(len(important_features), n_trees), 'Relative Importance', 'Feature', 'Images/CMIYC/IF_Bar_%s_%s.png'%(len(important_features), n_trees))
            plot_bar_important_features(important_features_1p, 'Feature Importance (%s features) (%s Trees)'%(len(important_features_1p), n_trees), 'Relative Importance', 'Feature', 'Images/CMIYC/IF_Bar_%s_%s_1p.png'%(len(important_features_1p), n_trees))

            r_col_indexes = []
            for col in data.drop(['label', 'label_encoded'], axis=1).columns:
                if col not in important_features_1p.index:
                    r_col_indexes.append(data.drop(['label', 'label_encoded'], axis=1).columns.tolist().index(col))
            X_train_mif = np.delete(X_train, r_col_indexes, axis=1)
            X_test_mif = np.delete(X_test, r_col_indexes, axis=1)
            X_mif = np.delete(X, r_col_indexes, axis=1)
            mif = {'X_train': X_train_mif, 'X_test': X_test_mif, 'y_train': y_train, 'y_test': y_test, 'X': X_mif, 'y': y}
            datasets.insert(1, mif)

        # ---------------------------------------- Unsupervised Learning Models [Find K]. ---------------------------------------- #

        if run_clustering == True:
            # KMEANS
            print('\nRunning K-Means clustering...')
            start_time = time.time()
            X = mms_scale(X)
            s_scores = []
            inertias = []
            c_scores = []
            h_scores = []
            k_values = list(range(2, 21))
            for n in k_values:
                clf_KMEANS = KMeans(n_clusters=n, random_state=rstate).fit(X)
                s_scores.append(silhouette_score(X, clf_KMEANS.labels_))
                inertias.append(clf_KMEANS.inertia_)
                c_scores.append(completeness_score(y, clf_KMEANS.labels_))
                h_scores.append(homogeneity_score(y, clf_KMEANS.labels_))
            plot_line_graph(k_values, s_scores, 'Cluster Number (k)', 'Silhouette Score', 'KMEANS - %s - Cluster Number (k) vs. Silhouette Score'%(dataset_name), 'Images/CMIYC/K_vs_Silhouette_score_%s_KMEANS.png'%(dataset_name.replace(' ', '_')))
            plot_line_graph(k_values, inertias, 'Cluster Number (k)', 'Inertia', 'KMEANS - %s - Cluster Number (k) vs. Inertia'%(dataset_name), 'Images/CMIYC/K_vs_Inertias_%s_KMEANS.png'%(dataset_name.replace(' ', '_')))
            plot_line_graph(k_values, c_scores, 'Cluster Number (k)', 'Completeness Score', 'KMEANS - %s - Cluster Number (k) vs. Completeness Score'%(dataset_name), 'Images/CMIYC/K_vs_Completeness_%s_KMEANS.png'%(dataset_name.replace(' ', '_')))
            plot_line_graph(k_values, h_scores, 'Cluster Number (k)', 'Homogeneity Score', 'KMEANS - %s - Cluster Number (k) vs. Homogeneity Score'%(dataset_name), 'Images/CMIYC/K_vs_homogeneity_%s_KMEANS.png'%(dataset_name.replace(' ', '_')))
            print('KMEANS Complete.\n--Total time: %.4f Seconds.'%(time.time()-start_time))
            # DBSCAN
            print('\nRunning DBSCAN Clustering...')
            start_time = time.time()
            eps_values = [i/10 for i in range(5, 55, 5)]
            n_cluster_values = []
            n_noise_values = []
            c_scores = []
            h_scores = []
            for n in eps_values:
                clf_DBSCAN = DBSCAN(eps=n).fit(X)
                n_cluster_values.append(len(np.unique(clf_DBSCAN.labels_)))
                n_noise_values.append(np.sum(np.array(clf_DBSCAN.labels_) == -1, axis= 0))
                c_scores.append(completeness_score(y, clf_DBSCAN.labels_))
                h_scores.append(homogeneity_score(y, clf_DBSCAN.labels_))
            plot_line_graph(eps_values, n_cluster_values, 'Epsilon', 'Cluster Number', 'DBSCAN - %s - Epsilon vs. Cluster Number'%(dataset_name), 'Images/CMIYC/eps_vs_cluster_BC_%s_DBSCAN.png'%(dataset_name.replace(' ', '_')))
            plot_line_graph(eps_values, n_noise_values, 'Epsilon', 'Noise', 'DBSCAN - %s - Epsilon vs. Noise'%(dataset_name), 'Images/CMIYC/eps_vs_noise_BC_%s_DBSCAN.png'%(dataset_name.replace(' ', '_')))
            plot_line_graph(eps_values, c_scores, 'Epsilon', 'Completness Score', 'DBSCAN - %s - Epsilon vs. Completeness Score'%(dataset_name), 'Images/CMIYC/eps_vs_Completeness_BC_%s_DBSCAN.png'%(dataset_name.replace(' ', '_')))
            plot_line_graph(eps_values, h_scores, 'Epsilon', 'Homogeneity Score', 'DBSCAN - %s - Epsilon vs. Homogeneity Score'%(dataset_name), 'Images/CMIYC/eps_vs_homogeneity_BC_%s_DBSCAN.png'%(dataset_name.replace(' ', '_')))
            print('DBSCAN Complete.\n--Total time: %.4f Seconds.'%(time.time()-start_time))

        # ---------------------------------------- Prediction Models. ---------------------------------------- #
        if run_train == True:
            start_time = time.time()
            # Decision Tree Classifier.
            clf_DT = DT_train(X_train, y_train, cv)
            save('Models/CMIYC/%s/DT'%(dataset_name), clf_DT)
            # Support Vector Machine Classifier.
            clf_SVM = SVM_train(X_train, y_train, cv)
            save('Models/CMIYC/%s/SVM'%(dataset_name), clf_SVM)
            # Multi-Layer Perceptron Classifier.
            clf_MLP = MLP_train(X_train, y_train, cv)
            save('Models/CMIYC/%s/MLP'%(dataset_name), clf_MLP)

            print('Training complete.\n--Total Training time: %.4f Seconds.'%(time.time()-start_time))

        print('\n'+'* '*10+'Dataset: '+str(dataset_name)+'. [' + str(len(X[0]))+' features]'+' *'*10)
        print('Number of unique classes: %s'%(len(np.unique(y))))

        # ---------------------------------------- Supervised Learning. ---------------------------------------- #
        clf_DT = load('Models/CMIYC/%s/DT'%(dataset_name))
        start_time = time.time()
        # Test.
        y_pred_test = clf_DT.predict(X_test)
        y_proba_test_DT = clf_DT.predict_proba(X_test)
        y_score_test = accuracy_score(y_test, y_pred_test)
        ul = unique_labels(y_test, y_pred_test, class_map)
        c_report_test = classification_report(y_test, y_pred_test, target_names=ul['label_names'])
        # Output.
        print('\nDecision Tree Model (DT):')
        print('-'*4+'Best parameters (GridSearchCV): %s'%(clf_DT.best_params_))        
        print('-'*4+'Best Score (GridSearchCV) (f1 CV score): %.2f'%(clf_DT.best_score_))
        print('-'*4+'Standard Deviation (GridSearchCV) (f1 CV score): %.2f'%(clf_DT.cv_results_['std_test_score'][clf_DT.best_index_]))
        print('-'*4+'Test:')
        print('-'*8+'Accuracy score: %.2f'%(y_score_test))
        print('-'*8+'Classification report:\n', c_report_test)
        print('-'*4+'Prediction & results time: %.4f seconds.'%(time.time()-start_time))        
        # Confusion Matrix.
        plot_cf_matrix(clf_DT, X_test, y_test, ul['label_names'], dataset_name, 'DT', 'Images/CMIYC/CM_%s_DT.png'%(dataset_name.replace(' ', '_')))

        # ---------------------------------------- 2.2.2. Suport vector Machine Model (SVM). ---------------------------------------- #
        clf_SVM = load('Models/CMIYC/%s/SVM'%(dataset_name))
        start_time = time.time()
        # Test.
        y_pred_test = clf_SVM.predict(X_test)
        y_proba_test_SVM = clf_SVM.predict_proba(X_test)
        y_score_test = accuracy_score(y_test, y_pred_test)
        ul = unique_labels(y_test, y_pred_test, class_map)
        c_report_test = classification_report(y_test, y_pred_test, target_names=ul['label_names'])
        # Output.
        print('\nSupport Vector Machine Model (SVM):')
        print('-'*4+'Best parameters (GridSearchCV): %s'%(clf_SVM.best_params_))
        print('-'*4+'Best Score (GridSearchCV) (f1 CV score): %.2f'%(clf_SVM.best_score_))
        print('-'*4+'Standard Deviation (GridSearchCV) (f1 CV score): %.2f'%(clf_DT.cv_results_['std_test_score'][clf_DT.best_index_]))
        print('-'*4+'Test:')
        print('-'*8+'Accuracy score: %.2f'%(y_score_test))
        print('-'*8+'Classification report:\n', c_report_test)
        print('-'*4+'Prediction & results time: %.4f seconds.'%(time.time()-start_time))
        # Confusion Matrix.
        plot_cf_matrix(clf_SVM, X_test, y_test, ul['label_names'], dataset_name, 'SVM', 'Images/CMIYC/CM_%s_SVM.png'%(dataset_name.replace(' ', '_')))

        # ---------------------------------------- 2.2.3. Multi-Layer Perceptron - Neural Network Model (MLP). ---------------------------------------- #
        clf_MLP = load('Models/CMIYC/%s/MLP'%(dataset_name))
        start_time = time.time()
        # Test.
        y_pred_test = clf_MLP.predict(X_test)
        y_proba_test_MLP = clf_MLP.predict_proba(X_test)
        y_score_test = accuracy_score(y_test, y_pred_test)
        ul = unique_labels(y_test, y_pred_test, class_map)
        c_report_test = classification_report(y_test, y_pred_test, target_names=ul['label_names'])
        # Output.
        print('\nMulti-Layer Perceptron Model (MLP):')
        print('-'*4+'Best parameters (GridSearchCV): %s'%(clf_MLP.best_params_))
        print('-'*4+'Best Score (GridSearchCV) (f1 CV score): %.2f'%(clf_MLP.best_score_))
        print('-'*4+'Standard Deviation (GridSearchCV) (f1 CV score): %.2f'%(clf_DT.cv_results_['std_test_score'][clf_DT.best_index_]))
        print('-'*4+'Test:')
        print('-'*8+'Accuracy score: %.2f'%(y_score_test))
        print('-'*8+'Classification report:\n', c_report_test)
        print('-'*4+'Prediction & results time: %.4f seconds.'%(time.time()-start_time))
        # Confusion Matrix.
        plot_cf_matrix(clf_MLP, X_test, y_test, ul['label_names'], dataset_name, 'MLP', 'Images/CMIYC/CM_%s_MLP.png'%(dataset_name.replace(' ', '_')))

        # Plot ROC Curve for all 3 classifiers.
        plot_ROC(y_test, (y_proba_test_DT, y_proba_test_SVM, y_proba_test_MLP), ('DT', 'SVM', 'MLP'), dataset_name, 'Images/CMIYC/ROC_BC_%s.png'%(dataset_name))


def plot_ROC(y, prediction_prob, cnames, dataset_name, fname):
    """
    Plot ROC curve for the classifiers in the binary classification case.
    """
    plt.figure(figsize=(20, 10))
    for i in range(len(cnames)):
        fpr, tpr, _ = roc_curve(y, prediction_prob[i][:,1])
        auc = roc_auc_score(y, prediction_prob[i][:,1])
        plt.plot(fpr, tpr, label='%s (AUC = %.2f)'%(cnames[i], auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Binary Classification - %s'%(dataset_name))
    plt.legend(loc="lower right")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def plot_cf_matrix(clf, X, y, label_names, dataset_name, clf_name, fname):
    """
    Confusuion matrix with plot_confusion_matrix.
    """
    _, ax = plt.subplots(figsize=(20,20))
    plot_confusion_matrix(clf, X, y, ax=ax, normalize='true', cmap='Blues', display_labels=label_names, values_format='.1%')
    plt.title('Confusion Matrix - %s - %s'%(dataset_name, clf_name))
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def unique_labels(y_test, y_pred_test, class_map):
    """
    Returns the TOTAL unique labels, names and values as dictionary.
    """
    labels = {}
    labels['label_values'] = np.unique(np.concatenate((y_test, y_pred_test)))   # Find the TOTAL number of unique labels.
    ul_named = []
    for i in labels['label_values']:
        ul_named.append(class_map[i])
    labels['label_names'] = ul_named
    return labels

def plot_line_graph(X, Y, xlabel, ylabel, title, fname):
    """
    Simple X Y Plot.
    """
    plt.figure(figsize=(20, 21))
    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(X)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def plot_bar_important_features(important_features, title, xlabel, ylabel, fname):
    """
    Bar graph of important_features.
    """
    plt.figure(figsize=(20, 21))
    plt.barh(important_features.index.astype(str).tolist(), important_features.values.tolist())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def MLP_train(X, y, cv):
    """
    Training Multi-Layer Pereptron model. Returns the model object.
    """
    start_time = time.time()
    print('\n'+ '# '*10+'[Training] Multi-Layer Perceptron (MLP):'+ ' #'*10)
    parameters = {
        'solver':('lbfgs', 'sgd', 'adam'),
        'hidden_layer_sizes':((33,), (66,), (100,)),
        'learning_rate_init':(0.001, 0.01, 0.1)
    }
    print('-'*2+'Grid Search Parameters:')
    print(parameters)
    clf = MLPClassifier(random_state=rstate)
    clf = GridSearchCV(clf, parameters, cv=cv, scoring='f1')
    clf.fit(X, y)
    print('-'*2+'GridSearch Results:')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame(clf.cv_results_))
    print('> '*2+'Training time: %.4f seconds.'%(time.time()-start_time))
    return clf

def SVM_train(X, y, cv):
    """
    Training Support Vector Machine model. Returns the model object.
    """
    rev = 10000
    start_time = time.time()
    print('\n'+ '# '*10+'[Training] Support Vector Machine (SVM) (max_iter='+str(rev)+'):'+ ' #'*10)
    parameters = {
        'kernel':('linear', 'poly', 'rbf'),
        'C':(1, 5, 10),
        'gamma':('scale', 'auto')
    }
    print('-'*2+'Grid Search Parameters:')
    print(parameters)
    clf = svm.SVC(random_state=rstate, probability=True, max_iter=rev) # Max_iter infinite as default, stuck on large datasets > 10^4.
    clf = GridSearchCV(clf, parameters, cv=cv, scoring='f1')
    clf.fit(X, y)
    print('-'*2+'GridSearch Results:')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame(clf.cv_results_))
    print('> '*2+'Training time: %.4f seconds.'%(time.time()-start_time))
    return clf

def DT_train(X, y, cv):
    """
    Training Decision Tree model. Returns the model object.
    """
    start_time = time.time()
    print('\n'+ '# '*10+'[Training] Decision Tree Model (DT):'+ ' #'*10)
    parameters = {
        'max_depth':(1, 5, 10),
        'max_features':('auto', 'sqrt', 'log2'),
        'min_samples_leaf':(1, 5, 10)
    }
    print('-'*2+'Grid Search Parameters:')
    print(parameters)
    clf = tree.DecisionTreeClassifier(random_state=rstate)
    clf = GridSearchCV(clf, parameters, cv=cv, scoring='f1')
    clf.fit(X, y)
    print('-'*2+'GridSearch Results:')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame(clf.cv_results_))
    print('> '*2+'Training time: %.4f seconds.'%(time.time()-start_time))
    return clf

def ETC_train(X, y, n_trees):
    """
    Returns the trained model for ExtraTreeClassifier. Returns the model object.
    """
    start_time = time.time()
    print('\n'+ '# '*10+'[Training] ExtraTreeClassifier Model (ETC):'+ ' #'*10)
    clf = ExtraTreesClassifier(n_estimators=n_trees).fit(X, y)
    print('> '*2+'Training time: %.4f seconds.'%(time.time()-start_time))
    return clf

def mms_scale(values):
    """
    Scale values with MinMaxScaler().
    """
    mms = MinMaxScaler()
    return mms.fit_transform(values)

def plot_3d(df, title, xlabel, ylabel, zlabel, fname):
    """
    Plot the 3d graph. Colors, see https://stackoverflow.com/questions/28999287/generate-random-colors-rgb.
    """
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1,1,1, projection='3d')
    hex_colors = ['#e6194B', '#a9a9a9', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6']
    ulabels = df['label'].unique()
    for i in range(len(ulabels)):
        df_label = df.loc[df['label'] == ulabels[i]]
        ax.scatter(df_label['X'], df_label['Y'], df_label['Z'], c=hex_colors[i], label=ulabels[i], alpha=0.6)
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.legend()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def plot_bar_outliers(df_outliers, title, xlabel, ylabel, fname):
    """
    Bar graph for outliers (fliers from boxplot).
    """
    feature_names = df_outliers.drop(['Feature', 'label'], axis=1).columns.tolist()
    n_outliers = df_outliers.drop(['Feature', 'label'], axis=1).iloc[0].tolist()
    vals, names = (list(t) for t in zip(*sorted(zip(n_outliers, feature_names), reverse=True)))
    plt.figure(figsize=(20, 21))
    plt.barh(names, vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def outliers(df):
    """
    Train Local Outlier Factor Model. Return dataframe for various outlier metrics.
    """
    # LocalOutlierFactor.
    start_time = time.time()
    print('\n'+ '# '*10+'[Training] Local Outlier Factor Model (LOF):'+ ' #'*10)
    clf =  LocalOutlierFactor()
    y_pred = clf.fit_predict(df.drop(['label', 'label_encoded'], axis=1))
    print('> '*2+'Training and prediction time: %.4f seconds.'%(time.time()-start_time))
    # Dataframe with various metrics.
    metrics = ['fliers', 'Q1', 'Q3', 'IQR', 'min', 'max', 'median', 'LOF_inliers', 'LOF_outliers', 'LOF_outlier_factor']
    df_outliers = pd.DataFrame()
    df_outliers['Feature'] = metrics
    bp = plt.boxplot([df[i] for i in df.drop(['label', 'label_encoded'], axis=1).columns])
    for i in range(len(df.drop(['label', 'label_encoded'], axis=1).columns)):
        vals = []
        # Fliers.
        vals.append(len(bp['fliers'][i].get_ydata()))
        # Q1.
        vals.append(df[df.drop(['label', 'label_encoded'], axis=1).columns[i]].quantile(0.25))
        # Q3.
        vals.append(df[df.drop(['label', 'label_encoded'], axis=1).columns[i]].quantile(0.75))
        # IQR.
        vals.append(df[df.drop(['label', 'label_encoded'], axis=1).columns[i]].quantile(0.75) - df[df.drop(['label', 'label_encoded'], axis=1).columns[i]].quantile(0.25))
        # Min.
        vals.append(df[df.drop(['label', 'label_encoded'], axis=1).columns[i]].min())
        # Max.
        vals.append(df[df.drop(['label', 'label_encoded'], axis=1).columns[i]].max())
        # Median.
        vals.append(df[df.drop(['label', 'label_encoded'], axis=1).columns[i]].median())
        # Local Outlier Factor.
        vals.append(y_pred.tolist().count(1))    # Inliers.
        vals.append(y_pred.tolist().count(-1))  # Outliers.
        vals.append(clf.negative_outlier_factor_)
        # Add column and data.
        df_outliers[df.columns[i]] = vals
    plt.close()
    return df_outliers

def save(fname, data):
    """
    Storing values in cpickle file.
    """
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def load(fname):
    """
    Reading stored values from cpickle file.
    """
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def LOF_train(df):
    """
    Train Local Outlier Factor Model. Returns model object and Prediction results.
    """
    start_time = time.time()
    print('\n'+ '# '*10+'[Training] Local Outlier Factor Model (LOF):'+ ' #'*10)
    clf =  LocalOutlierFactor()
    y_pred = clf.fit_predict(df.drop(['label', 'label_encoded'], axis=1))
    print('> '*2+'Training and prediction time: %.4f seconds.'%(time.time()-start_time))
    return clf, y_pred

def plot_pie_label_frequency(af, title, fname):
    """
    Pie Chart of attack frequencies.
    """
    plt.figure(figsize=(10, 7))
    vals, keys = (list(t) for t in zip(*sorted(zip(af.values(), af.keys()), reverse=True)))
    plt.pie(vals)
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(0.9,1.025), labels=['%s, %.2f%%'%(l, (s/sum(vals))*100) for l, s in zip(keys, vals)])
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def plot_bar_label_frequency(af, title, xlabel, ylabel, fname):
    """
    Bar plot of attack frequencies.
    """
    plt.figure(figsize=(20, 12))
    vals, keys = (list(t) for t in zip(*sorted(zip(af.values(), af.keys()), reverse=True)))
    plt.bar(keys, vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def label_frequency(df):
    """
    Return a dictionary of each unique label frequency.
    """
    af = {}
    for uv in df['label'].unique():
        af[uv] = df['label'].tolist().count(uv)
    return af

def string_timestamp(df):
    """
    Convert string timestamp 2014-03-24 16:31:05 to seconds.
    """
    time_columns = [df.columns[i] for i in range(len(df.columns)) if 'time' in df.columns[i]]
    for i in range(len(time_columns)):
        timestamps = []
        for j in range(len(df[time_columns[i]])):
            timestamps.append(time.mktime(datetime.datetime.strptime(str(df[time_columns[i]][j]), '%Y-%m-%d %H:%M:%S').timetuple()))
        df = df.drop(time_columns[i], axis=1)
        df[time_columns[i]] = timestamps
    return df

def check_missing_values(df):
    """
    Returns nr of missing values for each column, if there are any.
    """
    for i in range(len(df.columns)):
        n_missing_values = df[df.columns[i]].isnull().sum()
        if n_missing_values != 0:
            print('Column %s missing %s value(s) (%.2f%% of column values).'%(df.columns[i], n_missing_values, (n_missing_values/len(df[df.columns[i]].index))*100))
    c = 0
    for i in range(len(df.values)):
        n_missing_values = df.iloc[i].isnull().sum()
        if n_missing_values != 0:
            c += 1
    print('> Number of rows containing missing values %s (%.2f%% of whole dataset).'%(c, (c/len(df.values))*100))

if __name__ == "__main__":
    main()