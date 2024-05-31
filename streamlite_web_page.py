
from chembl_webresource_client.new_client import new_client
import streamlit as st
import pandas as pd
import numpy as np
import imblearn
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

######### SIDEBAR SPECIFICATION #########################
# Sidebar - Specify parameter settings
mode = st.sidebar.radio('Choose a Mode:', ['Custom Action Mode', 'AutoML Mode'])
st.sidebar.header('Set Parameters')
if mode == 'Custom Action Mode':
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    st.sidebar.subheader("Choose classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression", "Random Forest", "Gradient Boosting"))
    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01)
        max_iter = st.sidebar.slider("Maximum iterations", 100, 500)
    if classifier == 'Random Forest':
        st.sidebar.subheader('Hyperparameters')
        max_depth = st.sidebar.number_input('max_depth', 16)
        n_estimators = st.sidebar.number_input('n_estimators', 256)
    if classifier == 'Gradient Boosting':
        st.sidebar.subheader('Hyperparameters')
        n_estimators_GB = st.sidebar.number_input("n_estimators", 5, 500)
        max_depth_GB = st.sidebar.number_input('max_depth', 1, 10)
        learning_rate_GB = st.sidebar.number_input('learning rate' ,1, 10)
if mode == 'AutoML Mode':
    st.sidebar.button("This AutoML system is designed for efficient operation on datasets with low means and constant variances. "
                 "Features undergo specialized preprocessing to ensure close-to-zero means and constant variances. To address "
                 "dataset imbalances, the Synthetic Minority Over-sampling Technique (SMOTE) algorithm is applied. Hyperparameter "
                 "tuning utilizes the Grid Search method for systematically finding the optimal combination. In summary, the AutoML system "
                 "streamlines data preprocessing, imbalance correction, and hyperparameter optimization, offering users a simplified approach to "
                 "building reliable machine learning models.")
#########################################################################
############################ MAIN PANEL ############################

st.write("""# DrugAutoML""")

target_id = st.text_input("Enter your CHEMBL target id", "Type Here...")
col1, col2, col3, col4= st.columns(4)

with col1:
    button1 = st.button('Raw Data')

with col2:
    button2 = st.button('Cleaned Data')

with col3:
    button3 = st.button('Fingerprints')

with col4:
    button4 = st.button('Results')

if(button1):
    activities = new_client.activity.filter(target_chembl_id__in=[target_id],
                                            assay_type='B').only(['molecule_chembl_id',
                                                                  'canonical_smiles',
                                                                  'activity_comment',
                                                                  'type', 'units', 'relation', 'value'])
    st.dataframe(activities)
if(button2):
    activities = new_client.activity.filter(target_chembl_id__in=[target_id],
                                            assay_type='B').only(['molecule_chembl_id',
                                                                  'canonical_smiles',
                                                                  'activity_comment',
                                                                  'type', 'units', 'relation', 'value'])
    data = pd.DataFrame(activities)
    # deleting blank smiles format
    data = data.dropna(subset=['canonical_smiles'])
    print(data.shape)
    # selecting units as % and nM (nanomolar)
    data = data[data['units'].str.contains('%|nM', na=False)]
    # selecting activity type
    kd = r'\b[kK][dD]\b'
    ki = r'\b[kK][iI]\b'
    ic50 = r'\b[iI][cC]50\b'
    ec50 = r'\b[eE][cC]50\b'
    inhibition = r'\b[iI][nN][hH]'  # INH
    potency = r'\b[pP][oO][tT][eE][nN][cC][yY]\b'
    activity_list = [kd, ki, ic50, ec50, inhibition, potency]
    activity_result = '|'.join(activity_list)
    data = data[data['type'].str.contains(activity_result, na=False)]
    # standardization of activity types
    data['type'].mask(data['type'].str.contains(kd), 'kd', inplace=True)
    data['type'].mask(data['type'].str.contains(ki), 'ki', inplace=True)
    data['type'].mask(data['type'].str.contains(ic50), 'ic50', inplace=True)
    data['type'].mask(data['type'].str.contains(ec50), 'ec50', inplace=True)
    data['type'].mask(data['type'].str.contains(inhibition), 'inhibition', inplace=True)
    data['type'].mask(data['type'].str.contains(potency), 'potency', inplace=True)
    # creating activity response column using activity comments, relations and values
    # based on the activity comments
    not_active = r'\b[nN][oO][tT]\b'
    inactive = r'\b[iI][nN][aA][cC][tT][iI][vV][eE]\b'
    comment_inactive_keywords = [not_active, inactive]
    comment_inactive_keywords = '|'.join(comment_inactive_keywords)
    data['response'] = np.where(data['activity_comment'].str.contains(comment_inactive_keywords), 'non-active', 'NaN')
    # based on the relations and values
    data['value'] = data['value'].astype(float)  # converting values from string to float
    # for  inhibition
    inhibition_threshold = 11
    data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
             (data['relation'].isin(['>', '>=', '=', 'None'])), 'response'] = 'active'
    data.loc[(data['type'] == 'inhibition') & (data['value'] < inhibition_threshold), 'response'] = 'non-active'
    data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
             (data['relation'].isin(['<', '<='])), 'response'] = 'non-active'
    # for kd, ki, ic50, ec50 and potency
    inactivity_threshold = 39999
    data.loc[(data['type'] != 'inhibition') & (data['value'] > inactivity_threshold), 'response'] = 'non-active'
    activity_threshold = 1001
    data.loc[(data['type'] != 'inhibition') & (data['value'] < activity_threshold) &
             (data['relation'].isin(['<', '<=', '=', 'None'])), 'response'] = 'active'
    # exctracting grey area
    data = data[data['response'].str.contains('active|non-active', na=False)]

    # cleaning duplicative compounds
    data = data.sort_values(by=['molecule_chembl_id', 'type', 'value'])
    data = data.drop_duplicates(subset='molecule_chembl_id', keep='first')
    # re-indexing
    data.reset_index(drop=True, inplace=True)

    s = data['response']
    freq = s.value_counts(normalize=True) * 100
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.set_facecolor("black")
    freq.plot.barh(ax=ax, stacked=True)
    st.pyplot(fig)
    st.dataframe(data)
if (button3):
    activities = new_client.activity.filter(target_chembl_id__in=[target_id],
                                            assay_type='B').only(['molecule_chembl_id',
                                                                  'canonical_smiles',
                                                                  'activity_comment',
                                                                  'type', 'units', 'relation', 'value'])
    data = pd.DataFrame(activities)
    # deleting blank smiles format
    data = data.dropna(subset=['canonical_smiles'])
    print(data.shape)
    # selecting units as % and nM (nanomolar)
    data = data[data['units'].str.contains('%|nM', na=False)]
    # selecting activity type
    kd = r'\b[kK][dD]\b'
    ki = r'\b[kK][iI]\b'
    ic50 = r'\b[iI][cC]50\b'
    ec50 = r'\b[eE][cC]50\b'
    inhibition = r'\b[iI][nN][hH]'  # INH
    potency = r'\b[pP][oO][tT][eE][nN][cC][yY]\b'
    activity_list = [kd, ki, ic50, ec50, inhibition, potency]
    activity_result = '|'.join(activity_list)
    data = data[data['type'].str.contains(activity_result, na=False)]
    # standardization of activity types
    data['type'].mask(data['type'].str.contains(kd), 'kd', inplace=True)
    data['type'].mask(data['type'].str.contains(ki), 'ki', inplace=True)
    data['type'].mask(data['type'].str.contains(ic50), 'ic50', inplace=True)
    data['type'].mask(data['type'].str.contains(ec50), 'ec50', inplace=True)
    data['type'].mask(data['type'].str.contains(inhibition), 'inhibition', inplace=True)
    data['type'].mask(data['type'].str.contains(potency), 'potency', inplace=True)
    # creating activity response column using activity comments, relations and values
    # based on the activity comments
    not_active = r'\b[nN][oO][tT]\b'
    inactive = r'\b[iI][nN][aA][cC][tT][iI][vV][eE]\b'
    comment_inactive_keywords = [not_active, inactive]
    comment_inactive_keywords = '|'.join(comment_inactive_keywords)
    data['response'] = np.where(data['activity_comment'].str.contains(comment_inactive_keywords), 'non-active', 'NaN')
    # based on the relations and values
    data['value'] = data['value'].astype(float)  # converting values from string to float
    # for  inhibition
    inhibition_threshold = 11
    data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
             (data['relation'].isin(['>', '>=', '=', 'None'])), 'response'] = 'active'
    data.loc[(data['type'] == 'inhibition') & (data['value'] < inhibition_threshold), 'response'] = 'non-active'
    data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
             (data['relation'].isin(['<', '<='])), 'response'] = 'non-active'
    # for kd, ki, ic50, ec50 and potency
    inactivity_threshold = 39999
    data.loc[(data['type'] != 'inhibition') & (data['value'] > inactivity_threshold), 'response'] = 'non-active'
    activity_threshold = 1001
    data.loc[(data['type'] != 'inhibition') & (data['value'] < activity_threshold) &
             (data['relation'].isin(['<', '<=', '=', 'None'])), 'response'] = 'active'
    # exctracting grey area
    data = data[data['response'].str.contains('active|non-active', na=False)]

    # cleaning duplicative compounds
    data = data.sort_values(by=['molecule_chembl_id', 'type', 'value'])
    data = data.drop_duplicates(subset='molecule_chembl_id', keep='first')
    # re-indexing
    data.reset_index(drop=True, inplace=True)
    # molecular fingerprints (ecfp)
    ecfp_fingerprints = []
    for value in data['canonical_smiles']:
        mol = Chem.MolFromSmiles(value)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
        fp_str = fp.ToBitString()
        ecfp_fingerprints.append(fp_str)
    ecfp_fingerprints_data = pd.DataFrame(ecfp_fingerprints)
    ecfp_data = pd.DataFrame(ecfp_fingerprints_data[0].apply(lambda x: pd.Series(list(x))))
    ecfp_data.columns = ['ecfp' + str(i + 1) for i in range(len(ecfp_data.columns))]
    ecfp_data.astype(float)
    st.dataframe(ecfp_data)
if (button4):
    if mode == 'Custom Action Mode':
        if classifier == 'Logistic Regression':
            activities = new_client.activity.filter(target_chembl_id__in=[target_id],
                                                    assay_type='B').only(['molecule_chembl_id',
                                                                          'canonical_smiles',
                                                                          'activity_comment',
                                                                          'type', 'units', 'relation', 'value'])
            data = pd.DataFrame(activities)
            # deleting blank smiles format
            data = data.dropna(subset=['canonical_smiles'])
            # selecting units as % and nM (nanomolar)
            data = data[data['units'].str.contains('%|nM', na=False)]
            # selecting activity type
            kd = r'\b[kK][dD]\b'
            ki = r'\b[kK][iI]\b'
            ic50 = r'\b[iI][cC]50\b'
            ec50 = r'\b[eE][cC]50\b'
            inhibition = r'\b[iI][nN][hH]'  # INH
            potency = r'\b[pP][oO][tT][eE][nN][cC][yY]\b'
            activity_list = [kd, ki, ic50, ec50, inhibition, potency]
            activity_result = '|'.join(activity_list)
            data = data[data['type'].str.contains(activity_result, na=False)]
            # standardization of activity types
            data['type'].mask(data['type'].str.contains(kd), 'kd', inplace=True)
            data['type'].mask(data['type'].str.contains(ki), 'ki', inplace=True)
            data['type'].mask(data['type'].str.contains(ic50), 'ic50', inplace=True)
            data['type'].mask(data['type'].str.contains(ec50), 'ec50', inplace=True)
            data['type'].mask(data['type'].str.contains(inhibition), 'inhibition', inplace=True)
            data['type'].mask(data['type'].str.contains(potency), 'potency', inplace=True)
            # creating activity response column using activity comments, relations and values
            # based on the activity comments
            not_active = r'\b[nN][oO][tT]\b'
            inactive = r'\b[iI][nN][aA][cC][tT][iI][vV][eE]\b'
            comment_inactive_keywords = [not_active, inactive]
            comment_inactive_keywords = '|'.join(comment_inactive_keywords)
            data['response'] = np.where(data['activity_comment'].str.contains(comment_inactive_keywords), 'non-active',
                                        'NaN')
            # based on the relations and values
            data['value'] = data['value'].astype(float)  # converting values from string to float
            # for  inhibition
            inhibition_threshold = 11
            data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                     (data['relation'].isin(['>', '>=', '=', 'None'])), 'response'] = 'active'
            data.loc[(data['type'] == 'inhibition') & (data['value'] < inhibition_threshold), 'response'] = 'non-active'
            data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                     (data['relation'].isin(['<', '<='])), 'response'] = 'non-active'
            # for kd, ki, ic50, ec50 and potency
            inactivity_threshold = 39999
            data.loc[(data['type'] != 'inhibition') & (data['value'] > inactivity_threshold), 'response'] = 'non-active'
            activity_threshold = 1001
            data.loc[(data['type'] != 'inhibition') & (data['value'] < activity_threshold) &
                     (data['relation'].isin(['<', '<=', '=', 'None'])), 'response'] = 'active'
            # exctracting grey area
            data = data[data['response'].str.contains('active|non-active', na=False)]

            # cleaning duplicative compounds
            data = data.sort_values(by=['molecule_chembl_id', 'type', 'value'])
            data = data.drop_duplicates(subset='molecule_chembl_id', keep='first')
            # re-indexing
            data.reset_index(drop=True, inplace=True)
            # molecular fingerprints (ecfp)
            ecfp_fingerprints = []
            for value in data['canonical_smiles']:
                mol = Chem.MolFromSmiles(value)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
                fp_str = fp.ToBitString()
                ecfp_fingerprints.append(fp_str)
            ecfp_fingerprints_data = pd.DataFrame(ecfp_fingerprints)
            ecfp_data = pd.DataFrame(ecfp_fingerprints_data[0].apply(lambda x: pd.Series(list(x))))
            ecfp_data.columns = ['ecfp' + str(i + 1) for i in range(len(ecfp_data.columns))]
            ecfp_data.astype(float)
            # named input and output data as X and Y
            data['response'] = data['response'].map({'non-active': 0, 'active': 1})
            X = ecfp_data
            Y = data['response']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
            base_LR = LogisticRegression(C=C, max_iter=max_iter)
            base_LR.fit(X_train, Y_train)
            Y_pred_lR = base_LR.predict(X_test)
            # predict probabilities
            pred_prob = base_LR.predict_proba(X_test)
            # roc curve for models
            fpr, tpr, thresh = roc_curve(Y_test, pred_prob[:, 1], pos_label=1)
            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(Y_test))]
            p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)
            # auc scores
            auc_score = roc_auc_score(Y_test, pred_prob[:, 1])
            auc_score = round(auc_score, 3)
            plt.style.use('dark_background')
            # plot roc curves
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.plot(fpr, tpr, linestyle='--', color='orange', label='Logistic Regression, AUC=' + str(auc_score))
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')
            plt.legend(loc='best')
            LR_roc = plt.savefig('ROC', dpi=300)
            st.pyplot(LR_roc)
            metrics_base_LR = {'Accuracy':     round(accuracy_score(Y_test, Y_pred_lR), 3),
                               'Precision':    round(precision_score(Y_test, Y_pred_lR), 3),
                               'Recall':       round(recall_score(Y_test, Y_pred_lR), 3),
                               'F1-Score':     round(f1_score(Y_test, Y_pred_lR), 3),
                               'AUC':          auc_score}
            st.dataframe(metrics_base_LR, width=250, height=250)
        if classifier == 'Random Forest':
            activities = new_client.activity.filter(target_chembl_id__in=[target_id],
                                                    assay_type='B').only(['molecule_chembl_id',
                                                                          'canonical_smiles',
                                                                          'activity_comment',
                                                                          'type', 'units', 'relation', 'value'])
            data = pd.DataFrame(activities)
            # deleting blank smiles format
            data = data.dropna(subset=['canonical_smiles'])
            # selecting units as % and nM (nanomolar)
            data = data[data['units'].str.contains('%|nM', na=False)]
            # selecting activity type
            kd = r'\b[kK][dD]\b'
            ki = r'\b[kK][iI]\b'
            ic50 = r'\b[iI][cC]50\b'
            ec50 = r'\b[eE][cC]50\b'
            inhibition = r'\b[iI][nN][hH]'  # INH
            potency = r'\b[pP][oO][tT][eE][nN][cC][yY]\b'
            activity_list = [kd, ki, ic50, ec50, inhibition, potency]
            activity_result = '|'.join(activity_list)
            data = data[data['type'].str.contains(activity_result, na=False)]
            # standardization of activity types
            data['type'].mask(data['type'].str.contains(kd), 'kd', inplace=True)
            data['type'].mask(data['type'].str.contains(ki), 'ki', inplace=True)
            data['type'].mask(data['type'].str.contains(ic50), 'ic50', inplace=True)
            data['type'].mask(data['type'].str.contains(ec50), 'ec50', inplace=True)
            data['type'].mask(data['type'].str.contains(inhibition), 'inhibition', inplace=True)
            data['type'].mask(data['type'].str.contains(potency), 'potency', inplace=True)
            # creating activity response column using activity comments, relations and values
            # based on the activity comments
            not_active = r'\b[nN][oO][tT]\b'
            inactive = r'\b[iI][nN][aA][cC][tT][iI][vV][eE]\b'
            comment_inactive_keywords = [not_active, inactive]
            comment_inactive_keywords = '|'.join(comment_inactive_keywords)
            data['response'] = np.where(data['activity_comment'].str.contains(comment_inactive_keywords), 'non-active',
                                        'NaN')
            # based on the relations and values
            data['value'] = data['value'].astype(float)  # converting values from string to float
            # for  inhibition
            inhibition_threshold = 11
            data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                     (data['relation'].isin(['>', '>=', '=', 'None'])), 'response'] = 'active'
            data.loc[(data['type'] == 'inhibition') & (data['value'] < inhibition_threshold), 'response'] = 'non-active'
            data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                     (data['relation'].isin(['<', '<='])), 'response'] = 'non-active'
            # for kd, ki, ic50, ec50 and potency
            inactivity_threshold = 39999
            data.loc[(data['type'] != 'inhibition') & (data['value'] > inactivity_threshold), 'response'] = 'non-active'
            activity_threshold = 1001
            data.loc[(data['type'] != 'inhibition') & (data['value'] < activity_threshold) &
                     (data['relation'].isin(['<', '<=', '=', 'None'])), 'response'] = 'active'
            # exctracting grey area
            data = data[data['response'].str.contains('active|non-active', na=False)]

            # cleaning duplicative compounds
            data = data.sort_values(by=['molecule_chembl_id', 'type', 'value'])
            data = data.drop_duplicates(subset='molecule_chembl_id', keep='first')
            # re-indexing
            data.reset_index(drop=True, inplace=True)
            # molecular fingerprints (ecfp)
            ecfp_fingerprints = []
            for value in data['canonical_smiles']:
                mol = Chem.MolFromSmiles(value)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
                fp_str = fp.ToBitString()
                ecfp_fingerprints.append(fp_str)
            ecfp_fingerprints_data = pd.DataFrame(ecfp_fingerprints)
            ecfp_data = pd.DataFrame(ecfp_fingerprints_data[0].apply(lambda x: pd.Series(list(x))))
            ecfp_data.columns = ['ecfp' + str(i + 1) for i in range(len(ecfp_data.columns))]
            ecfp_data.astype(float)
            # named input and output data as X and Y
            data['response'] = data['response'].map({'non-active': 0, 'active': 1})
            X = ecfp_data
            Y = data['response']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
            base_RF = RandomForestClassifier(max_depth = max_depth, n_estimators= n_estimators)
            base_RF.fit(X_train, Y_train)
            Y_pred_RF = base_RF.predict(X_test)
            # predict probabilities
            pred_prob = base_RF.predict_proba(X_test)
            # roc curve for models
            fpr, tpr, thresh = roc_curve(Y_test, pred_prob[:, 1], pos_label=1)
            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(Y_test))]
            p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)
            # auc scores
            auc_score = roc_auc_score(Y_test, pred_prob[:, 1])
            auc_score = round(auc_score, 3)
            plt.style.use('dark_background')
            # plot roc curves
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.plot(fpr, tpr, linestyle='--', color='orange', label='Random Forests, AUC=' + str(auc_score))
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')
            plt.legend(loc='best')
            RF_roc = plt.savefig('ROC', dpi=300)
            st.pyplot(RF_roc)
            metrics_base_RF = {'Accuracy':     round(accuracy_score(Y_test, Y_pred_RF), 3),
                               'Precision':    round(precision_score(Y_test, Y_pred_RF), 3),
                               'Recall':       round(recall_score(Y_test, Y_pred_RF), 3),
                               'F1-Score':     round(f1_score(Y_test, Y_pred_RF), 3),
                               'AUC':          auc_score}
            st.dataframe(metrics_base_RF, width=250, height=250)
        if classifier == 'Gradient Boosting':
            activities = new_client.activity.filter(target_chembl_id__in=[target_id],
                                                    assay_type='B').only(['molecule_chembl_id',
                                                                          'canonical_smiles',
                                                                          'activity_comment',
                                                                          'type', 'units', 'relation', 'value'])
            data = pd.DataFrame(activities)
            # deleting blank smiles format
            data = data.dropna(subset=['canonical_smiles'])
            # selecting units as % and nM (nanomolar)
            data = data[data['units'].str.contains('%|nM', na=False)]
            # selecting activity type
            kd = r'\b[kK][dD]\b'
            ki = r'\b[kK][iI]\b'
            ic50 = r'\b[iI][cC]50\b'
            ec50 = r'\b[eE][cC]50\b'
            inhibition = r'\b[iI][nN][hH]'  # INH
            potency = r'\b[pP][oO][tT][eE][nN][cC][yY]\b'
            activity_list = [kd, ki, ic50, ec50, inhibition, potency]
            activity_result = '|'.join(activity_list)
            data = data[data['type'].str.contains(activity_result, na=False)]
            # standardization of activity types
            data['type'].mask(data['type'].str.contains(kd), 'kd', inplace=True)
            data['type'].mask(data['type'].str.contains(ki), 'ki', inplace=True)
            data['type'].mask(data['type'].str.contains(ic50), 'ic50', inplace=True)
            data['type'].mask(data['type'].str.contains(ec50), 'ec50', inplace=True)
            data['type'].mask(data['type'].str.contains(inhibition), 'inhibition', inplace=True)
            data['type'].mask(data['type'].str.contains(potency), 'potency', inplace=True)
            # creating activity response column using activity comments, relations and values
            # based on the activity comments
            not_active = r'\b[nN][oO][tT]\b'
            inactive = r'\b[iI][nN][aA][cC][tT][iI][vV][eE]\b'
            comment_inactive_keywords = [not_active, inactive]
            comment_inactive_keywords = '|'.join(comment_inactive_keywords)
            data['response'] = np.where(data['activity_comment'].str.contains(comment_inactive_keywords), 'non-active',
                                        'NaN')
            # based on the relations and values
            data['value'] = data['value'].astype(float)  # converting values from string to float
            # for  inhibition
            inhibition_threshold = 11
            data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                     (data['relation'].isin(['>', '>=', '=', 'None'])), 'response'] = 'active'
            data.loc[(data['type'] == 'inhibition') & (data['value'] < inhibition_threshold), 'response'] = 'non-active'
            data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                     (data['relation'].isin(['<', '<='])), 'response'] = 'non-active'
            # for kd, ki, ic50, ec50 and potency
            inactivity_threshold = 39999
            data.loc[(data['type'] != 'inhibition') & (data['value'] > inactivity_threshold), 'response'] = 'non-active'
            activity_threshold = 1001
            data.loc[(data['type'] != 'inhibition') & (data['value'] < activity_threshold) &
                     (data['relation'].isin(['<', '<=', '=', 'None'])), 'response'] = 'active'
            # exctracting grey area
            data = data[data['response'].str.contains('active|non-active', na=False)]

            # cleaning duplicative compounds
            data = data.sort_values(by=['molecule_chembl_id', 'type', 'value'])
            data = data.drop_duplicates(subset='molecule_chembl_id', keep='first')
            # re-indexing
            data.reset_index(drop=True, inplace=True)
            # molecular fingerprints (ecfp)
            ecfp_fingerprints = []
            for value in data['canonical_smiles']:
                mol = Chem.MolFromSmiles(value)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
                fp_str = fp.ToBitString()
                ecfp_fingerprints.append(fp_str)
            ecfp_fingerprints_data = pd.DataFrame(ecfp_fingerprints)
            ecfp_data = pd.DataFrame(ecfp_fingerprints_data[0].apply(lambda x: pd.Series(list(x))))
            ecfp_data.columns = ['ecfp' + str(i + 1) for i in range(len(ecfp_data.columns))]
            ecfp_data.astype(float)
            # named input and output data as X and Y
            data['response'] = data['response'].map({'non-active': 0, 'active': 1})
            X = ecfp_data
            Y = data['response']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
            base_GB = GradientBoostingClassifier(n_estimators=n_estimators_GB, max_depth=max_depth_GB,
                                                 learning_rate=learning_rate_GB)
            base_GB.fit(X_train, Y_train)
            Y_pred_GB = base_GB.predict(X_test)
            # predict probabilities
            pred_prob = base_GB.predict_proba(X_test)
            # roc curve for models
            fpr, tpr, thresh = roc_curve(Y_test, pred_prob[:, 1], pos_label=1)
            # roc curve for tpr = fpr
            random_probs = [0 for i in range(len(Y_test))]
            p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)
            # auc scores
            auc_score = roc_auc_score(Y_test, pred_prob[:, 1])
            auc_score = round(auc_score, 3)
            plt.style.use('dark_background')
            # plot roc curves
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.plot(fpr, tpr, linestyle='--', color='orange', label='Gradient Boosting, AUC=' + str(auc_score))
            plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
            # title
            plt.title('ROC curve')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')
            plt.legend(loc='best')
            GB_roc = plt.savefig('ROC', dpi=300)
            st.pyplot(GB_roc)
            metrics_base_RF = {'Accuracy':     round(accuracy_score(Y_test, Y_pred_GB), 3),
                               'Precision':    round(precision_score(Y_test, Y_pred_GB), 3),
                               'Recall':       round(recall_score(Y_test, Y_pred_GB), 3),
                               'F1-Score':     round(f1_score(Y_test, Y_pred_GB), 3),
                               'AUC':          auc_score}
            st.dataframe(metrics_base_RF, width=250, height=250)
    if mode == 'AutoML Mode':
        if mode == 'Custom Action Mode':
            if classifier == 'Logistic Regression':
                activities = new_client.activity.filter(target_chembl_id__in=[target_id],
                                                        assay_type='B').only(['molecule_chembl_id',
                                                                              'canonical_smiles',
                                                                              'activity_comment',
                                                                              'type', 'units', 'relation', 'value'])
                data = pd.DataFrame(activities)
                # deleting blank smiles format
                data = data.dropna(subset=['canonical_smiles'])
                # selecting units as % and nM (nanomolar)
                data = data[data['units'].str.contains('%|nM', na=False)]
                # selecting activity type
                kd = r'\b[kK][dD]\b'
                ki = r'\b[kK][iI]\b'
                ic50 = r'\b[iI][cC]50\b'
                ec50 = r'\b[eE][cC]50\b'
                inhibition = r'\b[iI][nN][hH]'  # INH
                potency = r'\b[pP][oO][tT][eE][nN][cC][yY]\b'
                activity_list = [kd, ki, ic50, ec50, inhibition, potency]
                activity_result = '|'.join(activity_list)
                data = data[data['type'].str.contains(activity_result, na=False)]
                # standardization of activity types
                data['type'].mask(data['type'].str.contains(kd), 'kd', inplace=True)
                data['type'].mask(data['type'].str.contains(ki), 'ki', inplace=True)
                data['type'].mask(data['type'].str.contains(ic50), 'ic50', inplace=True)
                data['type'].mask(data['type'].str.contains(ec50), 'ec50', inplace=True)
                data['type'].mask(data['type'].str.contains(inhibition), 'inhibition', inplace=True)
                data['type'].mask(data['type'].str.contains(potency), 'potency', inplace=True)
                # creating activity response column using activity comments, relations and values
                # based on the activity comments
                not_active = r'\b[nN][oO][tT]\b'
                inactive = r'\b[iI][nN][aA][cC][tT][iI][vV][eE]\b'
                comment_inactive_keywords = [not_active, inactive]
                comment_inactive_keywords = '|'.join(comment_inactive_keywords)
                data['response'] = np.where(data['activity_comment'].str.contains(comment_inactive_keywords),
                                            'non-active',
                                            'NaN')
                # based on the relations and values
                data['value'] = data['value'].astype(float)  # converting values from string to float
                # for  inhibition
                inhibition_threshold = 11
                data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                         (data['relation'].isin(['>', '>=', '=', 'None'])), 'response'] = 'active'
                data.loc[
                    (data['type'] == 'inhibition') & (data['value'] < inhibition_threshold), 'response'] = 'non-active'
                data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                         (data['relation'].isin(['<', '<='])), 'response'] = 'non-active'
                # for kd, ki, ic50, ec50 and potency
                inactivity_threshold = 39999
                data.loc[
                    (data['type'] != 'inhibition') & (data['value'] > inactivity_threshold), 'response'] = 'non-active'
                activity_threshold = 1001
                data.loc[(data['type'] != 'inhibition') & (data['value'] < activity_threshold) &
                         (data['relation'].isin(['<', '<=', '=', 'None'])), 'response'] = 'active'
                # exctracting grey area
                data = data[data['response'].str.contains('active|non-active', na=False)]

                # cleaning duplicative compounds
                data = data.sort_values(by=['molecule_chembl_id', 'type', 'value'])
                data = data.drop_duplicates(subset='molecule_chembl_id', keep='first')
                # re-indexing
                data.reset_index(drop=True, inplace=True)
                # molecular fingerprints (ecfp)
                ecfp_fingerprints = []
                for value in data['canonical_smiles']:
                    mol = Chem.MolFromSmiles(value)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
                    fp_str = fp.ToBitString()
                    ecfp_fingerprints.append(fp_str)
                ecfp_fingerprints_data = pd.DataFrame(ecfp_fingerprints)
                ecfp_data = pd.DataFrame(ecfp_fingerprints_data[0].apply(lambda x: pd.Series(list(x))))
                ecfp_data.columns = ['ecfp' + str(i + 1) for i in range(len(ecfp_data.columns))]
                ecfp_data.astype(float)
                # named input and output data as X and Y
                data['response'] = data['response'].map({'non-active': 0, 'active': 1})
                X = ecfp_data
                Y = data['response']
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
                base_LR = LogisticRegression(C=C, max_iter=max_iter)
                base_LR.fit(X_train, Y_train)
                Y_pred_lR = base_LR.predict(X_test)
                # predict probabilities
                pred_prob = base_LR.predict_proba(X_test)
                # roc curve for models
                fpr, tpr, thresh = roc_curve(Y_test, pred_prob[:, 1], pos_label=1)
                # roc curve for tpr = fpr
                random_probs = [0 for i in range(len(Y_test))]
                p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=1)
                # auc scores
                auc_score = roc_auc_score(Y_test, pred_prob[:, 1])
                auc_score = round(auc_score, 3)
                plt.style.use('dark_background')
                # plot roc curves
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.plot(fpr, tpr, linestyle='--', color='orange', label='Logistic Regression, AUC=' + str(auc_score))
                plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
                # title
                plt.title('ROC curve')
                # x label
                plt.xlabel('False Positive Rate')
                # y label
                plt.ylabel('True Positive rate')
                plt.legend(loc='best')
                LR_roc = plt.savefig('ROC', dpi=300)
                st.pyplot(LR_roc)
                metrics_base_LR = {'Accuracy': round(accuracy_score(Y_test, Y_pred_lR), 3),
                                   'Precision': round(precision_score(Y_test, Y_pred_lR), 3),
                                   'Recall': round(recall_score(Y_test, Y_pred_lR), 3),
                                   'F1-Score': round(f1_score(Y_test, Y_pred_lR), 3),
                                   'AUC': auc_score}
                st.dataframe(metrics_base_LR, width=250, height=250)
    if mode == 'AutoML Mode':
        activities = new_client.activity.filter(target_chembl_id__in=[target_id],
                                                assay_type='B').only(['molecule_chembl_id',
                                                                      'canonical_smiles',
                                                                      'activity_comment',
                                                                      'type', 'units', 'relation', 'value'])
        data = pd.DataFrame(activities)
        # deleting blank smiles format
        data = data.dropna(subset=['canonical_smiles'])
        # selecting units as % and nM (nanomolar)
        data = data[data['units'].str.contains('%|nM', na=False)]
        # selecting activity type
        kd = r'\b[kK][dD]\b'
        ki = r'\b[kK][iI]\b'
        ic50 = r'\b[iI][cC]50\b'
        ec50 = r'\b[eE][cC]50\b'
        inhibition = r'\b[iI][nN][hH]'  # INH
        potency = r'\b[pP][oO][tT][eE][nN][cC][yY]\b'
        activity_list = [kd, ki, ic50, ec50, inhibition, potency]
        activity_result = '|'.join(activity_list)
        data = data[data['type'].str.contains(activity_result, na=False)]
        # standardization of activity types
        data['type'].mask(data['type'].str.contains(kd), 'kd', inplace=True)
        data['type'].mask(data['type'].str.contains(ki), 'ki', inplace=True)
        data['type'].mask(data['type'].str.contains(ic50), 'ic50', inplace=True)
        data['type'].mask(data['type'].str.contains(ec50), 'ec50', inplace=True)
        data['type'].mask(data['type'].str.contains(inhibition), 'inhibition', inplace=True)
        data['type'].mask(data['type'].str.contains(potency), 'potency', inplace=True)
        # creating activity response column using activity comments, relations and values
        # based on the activity comments
        not_active = r'\b[nN][oO][tT]\b'
        inactive = r'\b[iI][nN][aA][cC][tT][iI][vV][eE]\b'
        comment_inactive_keywords = [not_active, inactive]
        comment_inactive_keywords = '|'.join(comment_inactive_keywords)
        data['response'] = np.where(data['activity_comment'].str.contains(comment_inactive_keywords), 'non-active',
                                    'NaN')
        # based on the relations and values
        data['value'] = data['value'].astype(float)  # converting values from string to float
        # for  inhibition
        inhibition_threshold = 11
        data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                 (data['relation'].isin(['>', '>=', '=', 'None'])), 'response'] = 'active'
        data.loc[(data['type'] == 'inhibition') & (data['value'] < inhibition_threshold), 'response'] = 'non-active'
        data.loc[(data['type'] == 'inhibition') & (data['value'] >= inhibition_threshold) &
                 (data['relation'].isin(['<', '<='])), 'response'] = 'non-active'
        # for kd, ki, ic50, ec50 and potency
        inactivity_threshold = 39999
        data.loc[(data['type'] != 'inhibition') & (data['value'] > inactivity_threshold), 'response'] = 'non-active'
        activity_threshold = 1001
        data.loc[(data['type'] != 'inhibition') & (data['value'] < activity_threshold) &
                 (data['relation'].isin(['<', '<=', '=', 'None'])), 'response'] = 'active'
        # exctracting grey area
        data = data[data['response'].str.contains('active|non-active', na=False)]

        # cleaning duplicative compounds
        data = data.sort_values(by=['molecule_chembl_id', 'type', 'value'])
        data = data.drop_duplicates(subset='molecule_chembl_id', keep='first')
        # re-indexing
        data.reset_index(drop=True, inplace=True)
        # molecular fingerprints (ecfp)
        ecfp_fingerprints = []
        for value in data['canonical_smiles']:
            mol = Chem.MolFromSmiles(value)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False)
            fp_str = fp.ToBitString()
            ecfp_fingerprints.append(fp_str)
        ecfp_fingerprints_data = pd.DataFrame(ecfp_fingerprints)
        ecfp_data = pd.DataFrame(ecfp_fingerprints_data[0].apply(lambda x: pd.Series(list(x))))
        ecfp_data.columns = ['ecfp' + str(i + 1) for i in range(len(ecfp_data.columns))]
        ecfp_data.astype(float)
        # named input and output data as X and Y
        data['response'] = data['response'].map({'non-active': 0, 'active': 1})
        X = ecfp_data
        Y = data['response']
        #### Splitting dataset %80 and %20
        from sklearn.model_selection import train_test_split

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True,
                                                            test_size=0.20, random_state=42)
        base_LR = LogisticRegression(max_iter=1000)
        base_LR.fit(X_train, Y_train)
        Y_pred_lR = base_LR.predict(X_test)
        pred_prob_LR = base_LR.predict_proba(X_test)
        fpr_LR, tpr_LR, thresh = roc_curve(Y_test, pred_prob_LR[:, 1], pos_label=1)
        random_probs_LR = [0 for i in range(len(Y_test))]
        p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs_LR, pos_label=1)
        auc_score_LR = roc_auc_score(Y_test, pred_prob_LR[:, 1])
        auc_score_LR = round(auc_score_LR, 3)
        metrics_base_LR = {'Accuracy': round(accuracy_score(Y_test, Y_pred_lR), 3),
                           'Precision': round(precision_score(Y_test, Y_pred_lR), 3),
                           'Recall': round(recall_score(Y_test, Y_pred_lR), 3),
                           'F1-Score': round(f1_score(Y_test, Y_pred_lR), 3),
                           'AUC': auc_score_LR}
        metrics_base_LR_final = pd.DataFrame(metrics_base_LR, index=['base_LR'])
        # variance thresholding
        selector = VarianceThreshold()
        filtered_data = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()]
        X_processed = pd.DataFrame(filtered_data, columns=selected_features)
        smote = SMOTE()
        X_smote, Y_smote = smote.fit_resample(X_processed, Y)
        X_auto_train, X_auto_test, Y_auto_train, Y_auto_test = train_test_split(X_smote, Y_smote, shuffle=True,
                                                                                test_size=0.20, random_state=42)

        # parameter grid
        grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"], "solver": ['liblinear']}  # l1 lasso l2 ridge
        auto_LR = LogisticRegression(max_iter=1000)
        auto_LR_grid = GridSearchCV(auto_LR, grid, cv=10)
        auto_LR_grid.fit(X_auto_train, Y_auto_train)
        Y_pred_auto_LR = auto_LR_grid.predict(X_auto_test)
        pred_prob_auto_LR = auto_LR_grid.predict_proba(X_auto_test)
        fpr_auto, tpr_auto, thresh_auto = roc_curve(Y_auto_test, pred_prob_auto_LR[:, 1], pos_label=1)
        random_probs_auto = [0 for i in range(len(Y_auto_test))]
        p_fpr, p_tpr, _ = roc_curve(Y_auto_test, random_probs_auto, pos_label=1)
        auc_score_auto_LR = roc_auc_score(Y_auto_test, pred_prob_auto_LR[:, 1])
        metrics_auto_LR = {'Accuracy': round(accuracy_score(Y_auto_test, Y_pred_auto_LR), 3),
                           'Precision': round(precision_score(Y_auto_test, Y_pred_auto_LR), 3),
                           'Recall': round(recall_score(Y_auto_test, Y_pred_auto_LR), 3),
                           'F1-Score': round(f1_score(Y_auto_test, Y_pred_auto_LR), 3),
                           'AUC': round(auc_score_auto_LR, 3)}
        metrics_auto_LR_final = pd.DataFrame(metrics_auto_LR, index=['auto_LR'])
        final_metrics = pd.concat([metrics_base_LR_final, metrics_auto_LR_final])
        st.dataframe(final_metrics, width=400, height=100)
        # plot roc curves
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.style.use('dark_background')
        plt.plot(fpr_LR, tpr_LR, linestyle='--', color='orange', label='Base Logistic Regression, AUC=' + str(auc_score_LR))
        plt.plot(fpr_auto, tpr_auto, linestyle='--', color='green',
                 label='Auto Logistic Regression, AUC=' + str(auc_score_auto_LR))
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        final_roc = plt.savefig('ROC', dpi=300)
        st.pyplot(final_roc)