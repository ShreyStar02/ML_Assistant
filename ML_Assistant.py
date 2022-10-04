# Importing the required libraries
import numpy as np
import pandas as pd
# import seaborn as sns
# import tensorflow as tf
import pickle
import base64
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from tensorflow import keras
import smote_variants as sv
import streamlit as st
warnings.filterwarnings("ignore")
# sns.set()


# Dataset input
# dataframe = pd.read_csv('C:/Users/vsrbv/OneDrive - vit.ac.in/MUSTARD/Dr. Tina/titanic_test.csv')
# print(dataframe.head())


def dataset_cleaner(df):
    # getting the percentage of null values in each column
    null_per_column = pd.DataFrame(
        df.isnull().sum()/len(df), columns=['Percentage of null values'])
    # removing the column if the percentage of null values is more than 60%
    df = df.drop(
        null_per_column[null_per_column['Percentage of null values'] > 0.60].index, axis=1)
    # checking for the count of unique values in each column
    for column in df.columns:
        if(len(df[column].unique()) <= 10):
            # removing NULL values and performing label encoding
            df[column].fillna(value='NULL', inplace=True)
            label_encoder = preprocessing.LabelEncoder()
            df[column] = label_encoder.fit_transform(df[column])
        # removing the column if the count is more than 10 and the column is non-numeric
        elif(len(df[column].unique()) > 10 and not np.issubdtype(df[column].dtype, np.number)):
            df.drop(column, axis=1, inplace=True)
        # replacing the NULL values with the median of that column if the count is more than 10 and column is numeric
        elif(len(df[column].unique()) > 10 and np.issubdtype(df[column].dtype, np.number)):
            df[column].fillna(df[column].median(), inplace=True)
    # Cleaning the Entire Dataset Using the applymap Function
    df = df.applymap(lambda x: x.strip() if type(x) == str else x)
    return df


def feature_extractor(dataframe):
    # Dataset Cleaning
    dataframe = dataset_cleaner(dataframe)
    # Checking for co-relation in any 2 columns(except last column) and if it is more than 0.9 or less than -0.9, remove 1 column.
    # print("\n\nChecking for co-relation in any 2 columns and if it is more than 0.9 or less than -0.9, remove 1 column.")
    for i in range(len(dataframe.columns)-1):
        for j in range(i+1, len(dataframe.columns)-1):
            if abs(dataframe.iloc[:, i].corr(dataframe.iloc[:, j])) >= 0.9:
                # print("Removing column: ", dataframe.columns[j])
                dataframe = dataframe.drop(columns=dataframe.columns[j])
                break
    # print("\nChecking for co-relation in any 2 columns(except last column) and if it is more than 0.9 or less than -0.9, remove 1 column\n")
    # print(dataframe)
    # checking whether any 2 rows have the same contents. If both the rows are same, delete one row.
    dataframe.drop_duplicates(keep='first', inplace=True)
    # print("\nchecking whether any 2 rows have the same contents. If both the rows are same, delete one row\n")
    # print(dataframe)
    # Checking whether more than 85 percent of the values of a column are unique.
    # If the percentage is more than 85, remove column.
    for i in dataframe.columns:
        if dataframe[i].nunique()/dataframe.shape[0]*100 > 85:
            dataframe.drop(i, inplace=True, axis=1)
    # # checking for co-relation between any column and the output column and if it is less than 0.1 and more that -0.1, remove that column.
    # col_corr = set()  # Set of all the names of deleted columns
    # corr_matrix = dataframe.corr()
    # for i in range(len(corr_matrix.columns)):
    #     for j in range(i):
    #         if (corr_matrix.iloc[i, j] <= 0.1) and (corr_matrix.iloc[i, j] >= -0.1) and (corr_matrix.columns[j] not in col_corr):
    #             column = corr_matrix.columns[i]  # getting the name of column
    #             col_corr.add(column)
    #             if column in dataframe.columns:
    #                 # deleting the column from the dataset
    #                 del dataframe[column]
    # Converting continuous data to numeric data.
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    return dataframe


# Augmentation starts

def get_all_oversamplers():
    os_list_names = []
    os_list_items = sv.get_all_oversamplers()
    for l in os_list_items:
        name = str(l).split("'>")[0].split(".")[-1]
        os_list_names.append(name)
    return os_list_names


def check_os_obj(os_name):
    os_list = {
        'NoSMOTE': sv.NoSMOTE(),
        'SMOTE': sv.SMOTE(),
        'SMOTE_TomekLinks': sv.SMOTE_TomekLinks(),
        'SMOTE_ENN': sv.SMOTE_ENN(),
        'Borderline_SMOTE1': sv.Borderline_SMOTE1(),
        'Borderline_SMOTE2': sv.Borderline_SMOTE2(),
        'ADASYN': sv.ADASYN(),
        'AHC': sv.AHC(),
        'LLE_SMOTE': sv.LLE_SMOTE(),
        'distance_SMOTE': sv.distance_SMOTE(),
        'SMMO': sv.SMMO(),
        'polynom_fit_SMOTE': sv.polynom_fit_SMOTE(),
        'Stefanowski': sv.Stefanowski(),
        'Safe_Level_SMOTE': sv.Safe_Level_SMOTE(),
        'MSMOTE': sv.MSMOTE(),
        'DE_oversampling': sv.DE_oversampling(),
        'SMOBD': sv.SMOBD(),
        'SUNDO': sv.SUNDO(),
        'MSYN': sv.MSYN(),
        'SVM_balance': sv.SVM_balance(),
        'TRIM_SMOTE': sv.TRIM_SMOTE(),
        'SMOTE_RSB': sv.SMOTE_RSB(),
        'ProWSyn': sv.ProWSyn(),
        'SL_graph_SMOTE': sv.SL_graph_SMOTE(),
        'NRSBoundary_SMOTE': sv.NRSBoundary_SMOTE(),
        'LVQ_SMOTE': sv.LVQ_SMOTE(),
        'SOI_CJ': sv.SOI_CJ(),
        'ROSE': sv.ROSE(),
        'SMOTE_OUT': sv.SMOTE_OUT(),
        'SMOTE_Cosine': sv.SMOTE_Cosine(),
        'Selected_SMOTE': sv.Selected_SMOTE(),
        'LN_SMOTE': sv.LN_SMOTE(),
        'MWMOTE': sv.MWMOTE(),
        'PDFOS': sv.PDFOS(),
        'RWO_sampling': sv.RWO_sampling(),
        'NEATER': sv.NEATER(),
        'DEAGO': sv.DEAGO(),
        'Gazzah': sv.Gazzah(),
        'MCT': sv.MCT(),
        'ADG': sv.ADG(),
        'SMOTE_IPF': sv.SMOTE_IPF(),
        'KernelADASYN': sv.KernelADASYN(),
        'MOT2LD': sv.MOT2LD(),
        'V_SYNTH': sv.V_SYNTH(),
        'OUPS': sv.OUPS(),
        'SMOTE_D': sv.SMOTE_D(),
        'SMOTE_PSO': sv.SMOTE_PSO(),
        'CURE_SMOTE': sv.CURE_SMOTE(),
        'SOMO': sv.SOMO(),
        'CE_SMOTE': sv.CE_SMOTE(),
        'ISOMAP_Hybrid': sv.ISOMAP_Hybrid(),
        'Edge_Det_SMOTE': sv.Edge_Det_SMOTE(),
        'CBSO': sv.CBSO(),
        'DBSMOTE': sv.DBSMOTE(),
        'ASMOBD': sv.ASMOBD(),
        'Assembled_SMOTE': sv.Assembled_SMOTE(),
        'SDSMOTE': sv.SDSMOTE(),
        'DSMOTE': sv.DSMOTE(),
        'G_SMOTE': sv.G_SMOTE(),
        'NT_SMOTE': sv.NT_SMOTE(),
        'Lee': sv.Lee(),
        'SPY': sv.SPY(),
        'SMOTE_PSOBAT': sv.SMOTE_PSOBAT(),
        'MDO': sv.MDO(),
        'Random_SMOTE': sv.Random_SMOTE(),
        'ISMOTE': sv.ISMOTE(),
        'VIS_RST': sv.VIS_RST(),
        'GASMOTE': sv.GASMOTE(),
        'A_SUWO': sv.A_SUWO(),
        'SMOTE_FRST_2T': sv.SMOTE_FRST_2T(),
        'AND_SMOTE': sv.AND_SMOTE(),
        'NRAS': sv.NRAS(),
        'AMSCO': sv.AMSCO(),
        'SSO': sv.SSO(),
        'DSRBF': sv.DSRBF(),
        'NDO_sampling': sv.NDO_sampling(),
        'Gaussian_SMOTE': sv.Gaussian_SMOTE(),
        'kmeans_SMOTE': sv.kmeans_SMOTE(),
        'Supervised_SMOTE': sv.Supervised_SMOTE(),
        'SN_SMOTE': sv.SN_SMOTE(),
        'CCR': sv.CCR(),
        'ANS': sv.ANS(),
        'cluster_SMOTE': sv.cluster_SMOTE(),
        'E_SMOTE': sv.E_SMOTE(),
        'ADOMS': sv.ADOMS(),
        'SYMPROD': sv.SYMPROD()
    }
    if os_name not in os_list:
        print("Error: Oversampler name incorrect")
        print(list(os_list.keys()))
        return None
    return os_list[os_name]


def get_os_param(os_obj):
    return os_obj.get_params()


def set_os_param(os_obj, inp_param):
    for key, value in inp_param.items():
        if key not in os_obj.get_params():
            print(key, "is not a parameter of ", str(os_obj))
            print("Returning Oversampler Object")
            return os_obj
    if "random_state" in inp_param:
        if inp_param["random_state"] == None:
            inp_param["random_state"] = np.random
    os_obj.set_params(**inp_param)
    return os_obj


def os_sampling(features, target, os_obj):
    features_updated, target_updated = os_obj.sample(features, target)
    return features_updated, target_updated


def augmentation(df, os_name, ch=0):
    df = feature_extractor(df)
    # ch = input("Do you need to augment the Dataset?(Yes/No): ")
    # if(ch == "Yes" or ch == "yes" or ch == "YES"):
    features = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    # Print Available Oversamplers
    # os_list_names = get_all_oversamplers()
    # print("List of Oversamplers Available:")
    # for i in os_list_names:
    #     print(i)
    # Input Oversampler Name
    # os_name = input("Enter the Name of the Oversampler of your Choice: ")
    # os_name = "MWMOTE"
    os_obj = check_os_obj(os_name)
    inp_param = get_os_param(os_obj)
    inp_param["proportion"] = 2
    os_param = set_os_param(os_obj, inp_param)
    var1 = "Shape Before Oversampling : \n" + \
        str(features.shape) + "\n" + str(target.shape)
    features_updated, target_updated = os_sampling(features, target, os_param)
    var2 = "Shape After Oversampling : \n" + \
        str(features_updated.shape) + "\n" + str(target_updated.shape)
    outputvar = var1 + "\n" + var2
    if(ch == 0):
        return pd.DataFrame(np.concatenate((features_updated, target_updated[:, None]), axis=1), columns=df.columns), outputvar
    else:
        return pd.DataFrame(np.concatenate((features_updated, target_updated[:, None]), axis=1), columns=df.columns)


# Augmentation ends


def rf_model(X_train, X_test, Y_train, Y_test, ret=0):
    # Random Forest
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    cr = classification_report(model_pred, Y_test)
    mat = confusion_matrix(Y_test, model_pred)
    if(ret == 0):
        return accuracy_score(Y_test, model_pred)
    else:
        return mat, cr, accuracy_score(Y_test, model_pred), model, X_test, Y_test, model_pred


def svm_model(X_train, X_test, Y_train, Y_test, ret=0):
    # Support Vector Machine
    model = SVC()
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    cr = classification_report(model_pred, Y_test)
    mat = confusion_matrix(Y_test, model_pred)
    if(ret == 0):
        return accuracy_score(Y_test, model_pred)
    else:
        return mat, cr, accuracy_score(Y_test, model_pred), model, X_test, Y_test, model_pred


def nb_model(X_train, X_test, Y_train, Y_test, ret=0):
    # Naive Bayes
    model = GaussianNB()
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    cr = classification_report(model_pred, Y_test)
    mat = confusion_matrix(Y_test, model_pred)
    if(ret == 0):
        return accuracy_score(Y_test, model_pred)
    else:
        return mat, cr, accuracy_score(Y_test, model_pred), model, X_test, Y_test, model_pred


def sgd_model(X_train, X_test, Y_train, Y_test, ret=0):
    # Stochastic Gradient Dissent
    model = SGDClassifier()
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    cr = classification_report(model_pred, Y_test)
    mat = confusion_matrix(Y_test, model_pred)
    if(ret == 0):
        return accuracy_score(Y_test, model_pred)
    else:
        return mat, cr, accuracy_score(Y_test, model_pred), model, X_test, Y_test, model_pred


def knn_model(X_train, X_test, Y_train, Y_test, ret=0):
    # K Nearest Neighbor
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    cr = classification_report(model_pred, Y_test)
    mat = confusion_matrix(Y_test, model_pred)
    if(ret == 0):
        return accuracy_score(Y_test, model_pred)
    else:
        return mat, cr, accuracy_score(Y_test, model_pred), model, X_test, Y_test, model_pred


def dt_model(X_train, X_test, Y_train, Y_test, ret=0):
    # Decision Trees
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    cr = classification_report(model_pred, Y_test)
    mat = confusion_matrix(Y_test, model_pred)
    if(ret == 0):
        return accuracy_score(Y_test, model_pred)
    else:
        return mat, cr, accuracy_score(Y_test, model_pred), model, X_test, Y_test, model_pred


def lr_model(X_train, X_test, Y_train, Y_test, ret=0):
    # Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    cr = classification_report(model_pred, Y_test)
    mat = confusion_matrix(Y_test, model_pred)
    if(ret == 0):
        return accuracy_score(Y_test, model_pred)
    else:
        return mat, cr, accuracy_score(Y_test, model_pred), model, X_test, Y_test, model_pred


def save_model(model, X_test, Y_test, model_pred):
    # model.save("trained_model")
    # reconstructed_model = keras.models.load_model("trained_model")
    # # np.testing.assert_allclose(model_pred, reconstructed_model.predict(X_test))
    # reconstructed_model.fit(X_test, Y_test)
    # re_model_pred = reconstructed_model.predict(X_test)
    # cr = classification_report(model_pred, Y_test)
    # mat = confusion_matrix(Y_test, re_model_pred)
    # print("\n\n")
    # print(cr)
    # sns.heatmap(mat, annot=True, fmt="d")
    filename = 'trained_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return filename


# def use_model(model, X_test, Y_test, model_pred, filename, df):
#     loaded_model = pickle.load(open(filename, 'rb'))
#     result = loaded_model.score(X_test, Y_test)
#     # cr = classification_report(result, Y_test)
#     # mat = confusion_matrix(Y_test, result)
#     print("\n")
#     # print(cr)
#     # sns.heatmap(mat, annot=True, fmt="d")
#     print("Model Retrieved Successfully! Accuracy = ", end="")
#     print(result*100, end="%\n")
#     # Prediction
#     X_pred = []
#     print("\nEnter the following details in numeric(label encoded format):")
#     for i in range(len(df.columns[:-1])):
#         X_pred.append(input(df.columns[i] + ": "))
#     X_pred = [[float(i) for i in X_pred]]
#     # print(X_pred)
#     Y_pred = loaded_model.predict(X_pred)
#     print("\nModel Predicted Successfully! Your Result is:")
#     print(df.columns[-1], end="")
#     print(": ", end="")
#     print(Y_pred[0])


def model_training(df, os_name):
    # Feature Extraction
    # df = feature_extractor(df)
    # print("\nDataset Cleaned and Essential Features Extracted Successfully! Here is the Feature Extracted Dataset:")
    # print(df)
    # Dataset Augmentation
    if(os_name == "NONE"):
        df = feature_extractor(df)
    else:
        df = augmentation(df, os_name, 1)
    # print("\n\nAugmentation Process Completed!")
    # print(df)
    # Dataset Separation
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]
    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        features, target, test_size=0.2, random_state=10)
    # print("\n\nModel Built after Feature extraction:\n")
    rfc = svmc = nbc = sgdc = knnc = dtc = lrc = 0
    for i in range(5):
        rf_accuracy = rf_model(X_train, X_test, Y_train, Y_test)
        svm_accuracy = svm_model(X_train, X_test, Y_train, Y_test)
        nb_accuracy = nb_model(X_train, X_test, Y_train, Y_test)
        sgd_accuracy = sgd_model(X_train, X_test, Y_train, Y_test)
        knn_accuracy = knn_model(X_train, X_test, Y_train, Y_test)
        dt_accuracy = dt_model(X_train, X_test, Y_train, Y_test)
        lr_accuracy = lr_model(X_train, X_test, Y_train, Y_test)
        if(rf_accuracy >= svm_accuracy and rf_accuracy >= nb_accuracy and rf_accuracy >= sgd_accuracy and rf_accuracy >= knn_accuracy and rf_accuracy >= dt_accuracy and rf_accuracy >= lr_accuracy):
            rfc = rfc + 1
        elif(svm_accuracy >= rf_accuracy and svm_accuracy >= nb_accuracy and svm_accuracy >= sgd_accuracy and svm_accuracy >= knn_accuracy and svm_accuracy >= dt_accuracy and svm_accuracy >= lr_accuracy):
            svmc = svmc + 1
        elif(nb_accuracy >= svm_accuracy and nb_accuracy >= rf_accuracy and nb_accuracy >= sgd_accuracy and nb_accuracy >= knn_accuracy and nb_accuracy >= dt_accuracy and nb_accuracy >= lr_accuracy):
            nbc = nbc + 1
        elif(sgd_accuracy >= svm_accuracy and sgd_accuracy >= nb_accuracy and sgd_accuracy >= rf_accuracy and sgd_accuracy >= knn_accuracy and sgd_accuracy >= dt_accuracy and sgd_accuracy >= lr_accuracy):
            sgdc = sgdc + 1
        elif(knn_accuracy >= svm_accuracy and knn_accuracy >= nb_accuracy and knn_accuracy >= sgd_accuracy and knn_accuracy >= rf_accuracy and knn_accuracy >= dt_accuracy and knn_accuracy >= lr_accuracy):
            knnc = knnc + 1
        elif(dt_accuracy >= svm_accuracy and dt_accuracy >= nb_accuracy and dt_accuracy >= sgd_accuracy and dt_accuracy >= knn_accuracy and dt_accuracy >= rf_accuracy and dt_accuracy >= lr_accuracy):
            dtc = dtc + 1
        else:
            lrc = lrc + 1
    if(rfc >= svmc and rfc >= nbc and rfc >= sgdc and rfc >= knnc and rfc >= dtc and rfc >= lrc):
        mat, cr, acc, model, X_test, Y_test, model_pred = rf_model(
            X_train, X_test, Y_train, Y_test, 1)
    elif(svmc >= rfc and svmc >= nbc and svmc >= sgdc and svmc >= knnc and svmc >= dtc and svmc >= lrc):
        mat, cr, acc, model, X_test, Y_test, model_pred = svm_model(
            X_train, X_test, Y_train, Y_test, 1)
    elif(nbc >= svmc and nbc >= rfc and nbc >= sgdc and nbc >= knnc and nbc >= dtc and nbc >= lrc):
        mat, cr, acc, model, X_test, Y_test, model_pred = nb_model(
            X_train, X_test, Y_train, Y_test, 1)
    elif(sgdc >= svmc and sgdc >= nbc and sgdc >= rfc and sgdc >= knnc and sgdc >= dtc and sgdc >= lrc):
        mat, cr, acc, model, X_test, Y_test, model_pred = sgd_model(
            X_train, X_test, Y_train, Y_test, 1)
    elif(knnc >= svmc and knnc >= nbc and knnc >= sgdc and knnc >= rfc and knnc >= dtc and knnc >= lrc):
        mat, cr, acc, model, X_test, Y_test, model_pred = knn_model(
            X_train, X_test, Y_train, Y_test, 1)
    elif(dtc >= svmc and dtc >= nbc and dtc >= sgdc and dtc >= knnc and dtc >= rfc and dtc >= lrc):
        mat, cr, acc, model, X_test, Y_test, model_pred = dt_model(
            X_train, X_test, Y_train, Y_test, 1)
    else:
        mat, cr, acc, model, X_test, Y_test, model_pred = lr_model(
            X_train, X_test, Y_train, Y_test, 1)
    # print("\n\nModel Built Successfully! Here is your Classification Report: ")
    # print(cr)
    # sns.heatmap(mat, annot=True, fmt="d")
    op = str(mat) + "\n" + str(cr) + "\n" + str(acc)
    pkl_file = bytes(save_model(model, X_test, Y_test,
                     model_pred), encoding='utf8')
    b64 = base64.b64encode(pkl_file).decode()
    return b64, op
    # use_model(model, X_test, Y_test, model_pred, filename, df)
    # return X_train, X_test, Y_train, Y_test

# # Testing the function
# df = dataset_cleaner(dataframe)
# print("\nCleaning done\n")
# print(df)

# dfe = feature_extractor(dataframe)
# print("\nFeatures extracted\n")
# print(dfe)


# input = ""
# df_cleaned = []
st.title("Data cleaning")
st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")
input_c = st.file_uploader("Choose a file to clean:", type=["csv"])
if input_c:
    with st.spinner("Cleaning dataset..."):
        dataframe = pd.read_csv(input_c)
        df_cleaned = dataset_cleaner(dataframe)
        csv_cleaned = df_cleaned.to_csv().encode('utf-8')
    st.download_button("Click to Download", csv_cleaned,
                       "cleaned_file.csv", "text/csv", key='download-csv')

st.title("Feature extraction")
st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")
input_fe = st.file_uploader("Choose a file to extract features:", type=["csv"])
if input_fe:
    with st.spinner("Extracting features..."):
        dataframe = pd.read_csv(input_fe)
        df_fe = feature_extractor(dataframe)
        csv_fe = df_fe.to_csv().encode('utf-8')
    st.download_button("Click to Download", csv_fe,
                       "feature-extracted_file.csv", "text/csv", key='download-csv')

st.title("Dataset Augmentation")
st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")
input_a = st.file_uploader("Choose a file to augment:", type=["csv"])
if input_a:
    with st.spinner("Augmenting Dataset..."):
        dataframe = pd.read_csv(input_a)
        os_name = st.selectbox(
            'Sampling models',
            ('NoSMOTE', 'SMOTE', 'SMOTE_TomekLinks', 'SMOTE_ENN', 'Borderline_SMOTE1', 'Borderline_SMOTE2', 'ADASYN', 'AHC',
             'LLE_SMOTE', 'distance_SMOTE', 'SMMO', 'polynom_fit_SMOTE', 'Stefanowski', 'Safe_Level_SMOTE', 'MSMOTE', 'DE_oversampling', 'SMOBD', 'SUNDO',
             'MSYN', 'SVM_balance', 'TRIM_SMOTE', 'SMOTE_RSB', 'ProWSyn', 'SL_graph_SMOTE', 'NRSBoundary_SMOTE', 'LVQ_SMOTE', 'SOI_CJ', 'ROSE', 'SMOTE_OUT',
             'SMOTE_Cosine', 'Selected_SMOTE', 'LN_SMOTE', 'MWMOTE', 'PDFOS',
             'RWO_sampling', 'NEATER', 'DEAGO', 'Gazzah', 'MCT', 'ADG', 'SMOTE_IPF', 'KernelADASYN', 'MOT2LD', 'V_SYNTH', 'OUPS', 'SMOTE_D', 'SMOTE_PSO', 'CURE_SMOTE', 'SOMO', 'CE_SMOTE', 'ISOMAP_Hybrid',
             'Edge_Det_SMOTE', 'CBSO', 'DBSMOTE', 'ASMOBD', 'Assembled_SMOTE', 'SDSMOTE', 'DSMOTE', 'G_SMOTE', 'NT_SMOTE', 'Lee', 'SPY', 'SMOTE_PSOBAT', 'MDO', 'Random_SMOTE', 'ISMOTE',
             'VIS_RST', 'GASMOTE', 'A_SUWO', 'SMOTE_FRST_2T', 'AND_SMOTE', 'NRAS', 'AMSCO', 'SSO', 'DSRBF', 'NDO_sampling', 'Gaussian_SMOTE', 'kmeans_SMOTE', 'Supervised_SMOTE', 'SN_SMOTE',
             'CCR', 'ANS', 'cluster_SMOTE', 'E_SMOTE', 'ADOMS', 'SYMPROD'))
        df_fe, exp = augmentation(dataframe, os_name)
        csv_fe = df_fe.to_csv().encode('utf-8')
    st.download_button("Click to Download", csv_fe,
                       "augmented_file.csv", "text/csv", key='download-csv')
    st.markdown(exp)


st.title("Model Building")
st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")
input_m = st.file_uploader("Choose a file to build model:", type=["csv"])
if input_m:
    with st.spinner("Building model..."):
        dataframe = pd.read_csv(input_m)
        choice = st.selectbox(
            "Do u need to augment the dataset?",
            ["No", "Yes"]
        )
        if choice == 'Yes':
            os_name = st.selectbox(
                'Oversampling models:',
                ('NoSMOTE', 'SMOTE', 'SMOTE_TomekLinks', 'SMOTE_ENN', 'Borderline_SMOTE1', 'Borderline_SMOTE2', 'ADASYN', 'AHC',
                 'LLE_SMOTE', 'distance_SMOTE', 'SMMO', 'polynom_fit_SMOTE', 'Stefanowski', 'Safe_Level_SMOTE', 'MSMOTE', 'DE_oversampling', 'SMOBD', 'SUNDO',
                 'MSYN', 'SVM_balance', 'TRIM_SMOTE', 'SMOTE_RSB', 'ProWSyn', 'SL_graph_SMOTE', 'NRSBoundary_SMOTE', 'LVQ_SMOTE', 'SOI_CJ', 'ROSE', 'SMOTE_OUT',
                 'SMOTE_Cosine', 'Selected_SMOTE', 'LN_SMOTE', 'MWMOTE', 'PDFOS',
                 'RWO_sampling', 'NEATER', 'DEAGO', 'Gazzah', 'MCT', 'ADG', 'SMOTE_IPF', 'KernelADASYN', 'MOT2LD', 'V_SYNTH', 'OUPS', 'SMOTE_D', 'SMOTE_PSO', 'CURE_SMOTE', 'SOMO', 'CE_SMOTE', 'ISOMAP_Hybrid',
                 'Edge_Det_SMOTE', 'CBSO', 'DBSMOTE', 'ASMOBD', 'Assembled_SMOTE', 'SDSMOTE', 'DSMOTE', 'G_SMOTE', 'NT_SMOTE', 'Lee', 'SPY', 'SMOTE_PSOBAT', 'MDO', 'Random_SMOTE', 'ISMOTE',
                 'VIS_RST', 'GASMOTE', 'A_SUWO', 'SMOTE_FRST_2T', 'AND_SMOTE', 'NRAS', 'AMSCO', 'SSO', 'DSRBF', 'NDO_sampling', 'Gaussian_SMOTE', 'kmeans_SMOTE', 'Supervised_SMOTE', 'SN_SMOTE',
                 'CCR', 'ANS', 'cluster_SMOTE', 'E_SMOTE', 'ADOMS', 'SYMPROD'))
        else:
            os_name = 'NONE'
        pkl_file, stat = model_training(dataframe, os_name)
        st.download_button("Click to Download", pkl_file, "built_model.pkl",
                           "application/octet-stream", key='download-pkl')
        st.markdown(stat)

# model_training(dataframe)
