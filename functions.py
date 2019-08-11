import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, accuracy_score, confusion_matrix, recall_score, average_precision_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
from sklearn.feature_selection import RFECV
import autosklearn.classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
import xgboost as xgb
import re
from sklearn.model_selection import cross_val_score
from scipy.stats import uniform, randint
import pickle
import os.path
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelBinarizer
import sys
from sklearn.multiclass import OneVsOneClassifier
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn.tree import export_graphviz

def classification_pipeline(cancerType, chosen_class, chosen_algorithm, featureselection, featureEngineering, remove_FP=True, user_Features_Include=None, user_Features_Exclude=None, binary_class_threshold=None):
    if (user_Features_Include is not None) or (user_Features_Exclude is not None) or (remove_FP == False):
        
        if (user_Features_Include is None):
            user_Features_Include_None = "None"
        else:
            firsts = [w[0] for w in user_Features_Include]
            user_Features_Include_None = firsts
        
        if (user_Features_Exclude is None):
            user_Features_Exclude_None = "None"
        else:
            firsts = [w[0] for w in user_Features_Exclude]
            user_Features_Exclude_None = firsts

        dataset_name="{}_{}_{}_{}_{}_{}.csv".format(cancerType, chosen_class, featureselection, featureEngineering, remove_FP, user_Features_Include_None, user_Features_Exclude_None)
        
        filename="gs_System_{}_{}_{}_{}_{}".format(cancerType, chosen_class, featureselection, featureEngineering, remove_FP, user_Features_Include_None, user_Features_Exclude_None)

    else:
        filename="gs_{}_{}_{}_{}_{}".format(cancerType, chosen_class, chosen_algorithm, featureselection, featureEngineering)
        dataset_name="{}_{}_{}_{}.csv".format(cancerType, chosen_class, featureselection, featureEngineering)

    if (os.path.isfile("gs/{}".format(filename))):
        print("File ", filename, " found. Loading.")
        print()
        with open("gs/{}".format(filename), 'rb') as gs_file:
            gs = pickle.load(gs_file)
    else:        
        if (os.path.isfile("csv/{}".format(dataset_name))):
            print("File ", dataset_name, " found. Loading.")
            df=pd.read_csv("csv/{}".format(dataset_name))
            
            keepImportantUnknown=featureEngineering
            
        else:
            cancer = pd.read_csv("cancer.csv")

            pancancer_features = ['sample_code', 'AS', 'LOH_n_seg', 'LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'Study Abbreviation', 'Study Name', 'CNV_segs', 'CNV_frac', 'HRD', 'purity', 'Virus_CMV', 'Virus_EBV', 'Virus_HBV', 'Virus_HCV', 'Virus_HHV', 'Virus_HIV', 'Virus_HPV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV', 'Virus_BK_Polyomavirus', 'Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'leucocyte_fraction', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_miRNA', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0', 'Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils', 'totTCR_reads', 'totTCRa_reads', 'totTCRb_reads', 'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'mDNAsi', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Treg_cluster']
            
            #Filter by user_Features_Include        
            if (user_Features_Include is not None):
                if (chosen_class=="IS"):
                    col = 'Treg_cluster'     

                elif (chosen_class=="HA"):
                    col = ['Neoantigen_num', 'numberOfBindingExpressedPMHC']

                elif (chosen_class=="multilabel"):
                    col = ['Neoantigen_num', 'numberOfBindingExpressedPMHC', 'Treg_cluster']

                elif (chosen_class=="multiclass"):
                    col = ['Neoantigen_num', 'numberOfBindingExpressedPMHC', 'Treg_cluster']
                else:
                    col = chosen_class
                    
                if isinstance(col, str):
                    if (col not in user_Features_Include) and (col in dataset):
                        user_Features_Include.append(col)
                else:
                    for c in col:
                        if (c not in user_Features_Include) and (c in dataset):
                            user_Features_Include.append(c)
                #user_Features.append('Study Abbreviation')
                dataset = dataset[user_Features_Include]     

            #Filter by user_Features_Exclude        
            if (user_Features_Exclude is not None):
                if (chosen_class=="IS"):
                    col = 'Treg_cluster'     

                elif (chosen_class=="HA"):
                    col = ['Neoantigen_num', 'numberOfBindingExpressedPMHC']

                elif (chosen_class=="multilabel"):
                    col = ['Neoantigen_num', 'numberOfBindingExpressedPMHC', 'Treg_cluster']

                elif (chosen_class=="multiclass"):
                    col = ['Neoantigen_num', 'numberOfBindingExpressedPMHC', 'Treg_cluster']
                else:
                    col = chosen_class
                    
                if isinstance(col, str):
                    for feat in user_Features_Exclude:
                        if (feat != col):
                            dataset = dataset.drop(feat, axis=1)
                else:
                    for feat in user_Features_Exclude:
                        if (feat not in col):
                            dataset = dataset.drop(feat, axis=1)  
                
            else:
                if (cancerType == 'Overall'):
                    dataset = cancer[pancancer_features]
                    dataset = process_clean(dataset)
                    
                else:
                    dataset = cancer.copy()
                    dataset = process_clean(dataset)
                    dataset = cancerTypeDataset(dataset, cancerType)
                    dataset = process_clean_cancerType(dataset, cancerType)

            dataset = pre_process(dataset)

            if (featureEngineering):
                print("STARTING FEATURE ENGINEERING")
                dataset = create_sumFeatures(dataset)
                dataset = process_logNormalize(dataset)
                keepImportantUnknown=True
            else:
                keepImportantUnknown=False        

            if (featureselection):
                print("STARTING FEATURE SELECTION")
                dataset = featureSelection(dataset, chosen_class, normalize=False, dummyCat=True, remove_FP=remove_FP,user_Features_Include= user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold=binary_class_threshold, keepImportantUnknown=keepImportantUnknown)

            df=dataset.copy()
            df.to_csv("csv/{}".format(dataset_name), index=False)
         

        print("STARTING OPTIMIZED TRAINING")
        gs = optimize_classifier(df, chosen_class, chosen_algorithm, normalize=False, dummyCat=True, remove_FP=remove_FP, user_Features_Include=user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold=binary_class_threshold, keepImportantUnknown = keepImportantUnknown)    
        with open("gs/{}".format(filename), 'wb') as gs_file:
            pickle.dump(gs, gs_file, pickle.HIGHEST_PROTOCOL)
        
    
    print(gs.best_estimator_)
    print("Accuracy:", gs.best_score_)
    index = np.where(gs.cv_results_['mean_test_Accuracy']==gs.cv_results_['mean_test_Accuracy'].max())
    #if isinstance(chosen_class, str) and (chosen_class != 'Immune_evasion'):
    print("AUC:", gs.cv_results_['mean_test_AUC'][index])
    precision = gs.cv_results_['mean_test_Precision'][index][0]
    recall = gs.cv_results_['mean_test_Recall'][index][0]
    precision_std = gs.cv_results_['mean_test_Precision_std'][index][0]
    recall_std = gs.cv_results_['mean_test_Recall_std'][index][0]
    if (chosen_class=="multilabel" or chosen_class=="multiclass"):
        print("Precision: %0.2f (+/- %0.2f)" % (precision, precision_std * 2 ))
        print("Recall: %0.2f (+/- %0.2f)" % (recall, recall_std * 2 ))
    else:
        print("Precision: %0.2f" % precision)
        print("Recall: %0.2f" % recall)  
    
def process_clean(dataset):
    
    df=dataset.copy()
    
    #Remove irregularies in database
    #Continuous variables should be with nan for later missing processing
    df['leucocyte_fraction'] = df['leucocyte_fraction'].replace('                  ', np.nan)
    df['Study Abbreviation'] = df['Study Abbreviation'].replace('    ', 'Unknown')
    df['cause_of_death'] = df['cause_of_death'].replace(['               ','[Not Available]','[Unknown]      '], 'Unknown')
    df['cause_of_death'] = df['cause_of_death'].replace(['[Not Applicable', '#N/A           '], 'NA')
    df['vital_status'] = df['vital_status'].replace(['     ','[Disc'], 'Unknown')
    df['histological_type'] = df['histological_type'].replace(['                                                   ','[Discrepancy]                                      ','[Not Available]                                    '], 'Unknown')
    df['histological_type'] = df['histological_type'].replace('#N/A                                               ', 'NA')
    df['histological_grade'] = df['histological_grade'].replace(['[Discrepancy]  ','               ','[Not Available]','[Unknown]      '], 'Unknown')
    df['OS'] = df['OS'].replace('               ', np.nan)
    df['OS'] = df['OS'].replace('0              ', 0)
    df['OS'] = df['OS'].replace('1              ', 1)
    df['OS.time'] = df['OS.time'].replace('[#N/A]', np.nan)
    df['OS.time'] = df['OS.time'].replace('#N/A           ', np.nan)
    df['treatment_outcome_first_course'] = df['treatment_outcome_first_course'].replace(['                           ','[Not Available]            ','[Unknown]                  ','[Discrepancy]              ', '[Not Evaluated]            '], np.nan)
    df['treatment_outcome_first_course'] = df['treatment_outcome_first_course'].replace(['[Not Applicable]           ','#N/A                       '], 'NA')
    df['tumor_status'] = df['tumor_status'].replace('#N/A      ', 'NA')
    df['tumor_status'] = df['tumor_status'].replace(['          ','[Discrepan'], 'Unknown')
    df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace(['[Not Available]','[Unknown]      ','               '], 'Unknown')
    df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace('[Discrepancy]  ', 'Unknown')
    df['ajcc_pathologic_tumor_stage'] = df['ajcc_pathologic_tumor_stage'].replace('[Not Applicable', 'NA')
    df['mDNAsi'] = df['mDNAsi'].replace(0., 0.0)
    df['Subtype_mRNA'] = df['Subtype_mRNA'].replace('NA                                  ', 'NA')
    df['Subtype_mRNA'] = df['Subtype_mRNA'].replace(['                                    ','-                                   '], 'Unknown')
    df['Subtype_DNAmeth'] = df['Subtype_DNAmeth'].replace('NA               ', 'NA')
    df['Subtype_DNAmeth'] = df['Subtype_DNAmeth'].replace(['                 '], 'Unknown')
    df['Subtype_DNAmeth'] = df['Subtype_DNAmeth'].replace(['CpG island-methyl'], 'CpG island methyl')
    df['Subtype_protein'] = df['Subtype_protein'].replace(['  ', '- '], 'Unknown')
    df['Subtype_miRNA'] = df['Subtype_miRNA'].replace(['NA     ','#N/A   '], 'NA')
    df['Subtype_miRNA'] = df['Subtype_miRNA'].replace(['-      ','       '], 'Unknown')
    df['Subtype_CNA'] = df['Subtype_CNA'].replace('NA         ', 'NA')
    df['Subtype_CNA'] = df['Subtype_CNA'].replace('           ', 'Unknown')
    df['Subtype_Integrative'] = df['Subtype_Integrative'].replace('NA  ', 'NA')
    df['Subtype_Integrative'] = df['Subtype_Integrative'].replace('    ', 'Unknown')
    df['Subtype_other'] = df['Subtype_other'].replace('NA ', 'NA')
    df['Subtype_other'] = df['Subtype_other'].replace(['   ','-  '], 'Unknown')
    df['Subtype_Selected'] = df['Subtype_Selected'].replace('                     ', 'Unknown')
    df['race'] = df['race'].replace(['                         ','[Not Available]          ','[Not Evaluated]          ','[Unknown]                '], 'Unknown')
    df['gender'] = df['gender'].replace('      ', 'Unknown')
    df['age_at_initial_pathologic_diagnosis'] = df['age_at_initial_pathologic_diagnosis'].replace(['               ', '#N/A           '], np.nan)
        
    to_clean = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2','Study Abbreviation','cause_of_death','Subtype_Selected','Subtype_other', 'Subtype_Integrative','Subtype_CNA','Subtype_miRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_mRNA', 'ajcc_pathologic_tumor_stage', 'treatment_outcome_first_course', 'tumor_status', 'histological_type', 'histological_grade', 'vital_status', 'gender','race']
    for column in to_clean:
        df = clean_emptyString(df, column, mode='cat')
        
    df['Treg_cluster'] = df['Treg_cluster'].replace('               ', -1)
    df['Treg_cluster'] = df['Treg_cluster'].replace('0              ', 0)
    df['Treg_cluster'] = df['Treg_cluster'].replace('1              ', 1)
    
    df['OS'] = df['OS'].fillna(-1)

    #There were only 2544 non-nulll leucocyte_fraction 
    df = df.drop('leucocyte_fraction', axis=1)
    
    #No need to have both vital_status and OS.time and OS
    #df = df.drop('vital_status', axis=1)
    df = df.drop('OS.time', axis=1)
    #df = df.drop('OS', axis=1)
    
    #toDrop = ['vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'genre', 'race', 'age_at_initial_pathologic_diagnosis', 'Subtype_DNAmeth', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_Selected', 'Subtype_other', 'Subtype_miRNA', 'Subtype_protein', 'Subtype_mRNA', 'histological_type', 'histological_grade']
    toDrop=[]
    for col in toDrop:
        if col in df.columns:
            df = df.drop(col, axis=1)


    
    return df

def pre_process(dataset, dummyCat=True, catAge=True, normalize=False):
    
    pancancer_features = ['sample_code', 'AS', 'LOH_n_seg', 'LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'Study Abbreviation', 'Study Name', 'CNV_segs', 'CNV_frac', 'HRD', 'purity', 'Virus_CMV', 'Virus_EBV', 'Virus_HBV', 'Virus_HCV', 'Virus_HHV', 'Virus_HIV', 'Virus_HPV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV', 'Virus_BK_Polyomavirus', 'Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'leucocyte_fraction', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_miRNA', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0', 'Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils', 'totTCR_reads', 'totTCRa_reads', 'totTCRb_reads', 'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'mDNAsi', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Treg_cluster']
    
    df=dataset.copy()

    to_numeric = ['CNV_frac', 'OS', 'MutationsSilent', 'MutationsNonSilent', 'Virus_HCV','Virus_HPV','Virus_BK_Polyomavirus','Virus_CMV','Virus_EBV','Virus_HBV','Virus_HHV','Virus_HIV','Virus_HTLV','Virus_MCV','Virus_SV','Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation','numberOfBindingExpressedPMHC','Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated','Eosinophils','Neutrophils', 'totTCRa_reads', 'totTCR_reads', 'totTCRb_reads','mDNAsi', 'CNV_segs','Dendritic.cells.resting','Dendritic.cells.activated','Macrophages.M2','Macrophages.M1','Macrophages.M0','Monocytes']
    
    for column in to_numeric:
        if column in df.columns:
            df[column] = df[column].convert_objects(convert_numeric=True)
            df = clean_emptyString(df, column, mode='num')
            df = process_missing(df, column, 'median')
            
    if ('age_at_initial_pathologic_diagnosis' in df.columns):
        if catAge:
            df['age_at_initial_pathologic_diagnosis'] = df['age_at_initial_pathologic_diagnosis'].fillna(-1)
            df['age_at_initial_pathologic_diagnosis'] = df['age_at_initial_pathologic_diagnosis'].convert_objects(convert_numeric=True)
            df = process_categorizeAge(df)
        else:
            df['age_at_initial_pathologic_diagnosis'] = df['age_at_initial_pathologic_diagnosis'].convert_objects(convert_numeric=True)
            df = process_missing(df,'age_at_initial_pathologic_diagnosis', 'median')
  
    if normalize:        
        numerical=df.select_dtypes(include=['int', 'uint8', 'float', 'float64'])
        for feature in list(numerical.columns):
            df=process_normalize(df, feature, "n")


    #We remove the identifiers and some repeated variables
    cols = ['sample_code', 'Study Name', 'totTCR_reads']
    for col in cols:
        if col in df.columns:
            df = df.drop(col, axis=1)    
        
    categorical_variables = df.select_dtypes(include = 'object').columns.tolist()
    if catAge:
        categorical_variables.append('Age_categories')

    if dummyCat:
        for column in df:
            if column in categorical_variables:
                df = create_dummies(df, column)
                df = df.drop(column, axis =1)        
    else:   
        for column in df:
            if column in categorical_variables:
                df = df.drop(column, axis =1)        

    #SWITCH TREG CLUSTER SIGNAL
    if 'Treg_cluster' in df:
        df['Treg_cluster'] = np.where(df['Treg_cluster'] == 1, 0, np.where(df['Treg_cluster'] == 0, 1, -1))
   
    
    return df

def cancerTypeDataset(dataset, cancerType):
    
    pancancer_features = ['sample_code', 'AS', 'LOH_n_seg', 'LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'Study Abbreviation', 'Study Name', 'CNV_segs', 'CNV_frac', 'HRD', 'purity', 'Virus_CMV', 'Virus_EBV', 'Virus_HBV', 'Virus_HCV', 'Virus_HHV', 'Virus_HIV', 'Virus_HPV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV', 'Virus_BK_Polyomavirus', 'Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'leucocyte_fraction', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_miRNA', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0', 'Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils', 'totTCR_reads', 'totTCRa_reads', 'totTCRb_reads', 'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'mDNAsi', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2','Treg_cluster']
    
    df=dataset.copy()
    
    #DROP TRASH COLUMNS
    toDrop = ['abnormality_tested_methodology.1', 'abnormality_tested_results.1', 'b_lymphocyte_genotyping_method.1', 'bone_marrow_involvement.1', 'sample_code_1','sample_code_1_1','sample_code_2','sample_code_3','age_at_diagnosis','ASprime', 'disease_code', 'residual_tumor_1' ]    
    for col in toDrop:
        if col in df.columns:
            df = df.drop([col], axis=1)
 
    #FILTER DATASET ROWS
    df['Study Abbreviation'] = df['Study Abbreviation'].str.strip()
    dfType = df[df['Study Abbreviation']==cancerType]

    #FILTER DATASET COLUMNS
    followup_features = []
    categorical=dfType.select_dtypes(include=['object'], exclude=['float64'])
    #For some reason this one is being added sometimes (??)
    if ('immunohistochemistry_positive_cell_score' in categorical):   
        categorical = categorical.drop(['immunohistochemistry_positive_cell_score'], axis=1)
    for col in dfType.columns:
        if col not in pancancer_features:
            if col in categorical:
                #lower case all columns
                #print("THIS COLUMN", col)
                dfType[col] = dfType[col].str.lower()
                dfType[col] = dfType[col].str.strip()
            #see if there is any patient with a non-null entry in each followup column
            size = dfType[dfType[col].notnull()].shape[0]
            if size > 0:
                followup_features.append(col)
            else:
                dfType = dfType.drop([col], axis=1)
                
    dfType = filterByVariance(dfType, followup_features)  
    dfType = filterByVariance(dfType, pancancer_features)  
    
    return dfType

def filterByVariance(d, columns):
    df = d.copy()
    
    for col in columns:
        if col in df:
            var = df[col].unique()
            length=len(var)
            for value in var:
                if str(value).isspace():
                    length=length-1
            if length<2:
                df = df.drop([col],axis=1)
            
    return df

def clean_emptyString(dataset, column, mode=None):
    df=dataset.copy()
    unique=dataset[column].unique()
    for val in unique:
        if str(val).isspace():
            if (mode=='cat'):
                df[column] = df[column].replace(str(val), 'Unknown')
            if (mode=='num'):
                df[column] = df[column].replace(str(val), np.nan)
                
    df = clean_otherStrings(df, column, mode)
    
    return df

def clean_otherStrings(dataset, column, mode=None):
    
    df=dataset.copy()
    unknown_values=['','[unknown]', '[not available]','[not evaluated]', '[not applicable]' ,'indeterminate','specify','[discrepan     ','#n/a','[not applicable','[no','[not available','[not av','other_ specify','equivocal','other (specify)','[not','other','[unk','indeterminate (','[discrepancy]', 'other (specify)', 'not amplified','>6.0', '<4.0', '>6', '[not', '[not avai', 'other method, specify:', '`', 'other, specify', '[not availabl', 'other (please specify)', '[not avail', '[not evalu', '[un'] 
    unique=dataset[column].unique()
    for val in unique:
        if str(val) in unknown_values:
            if (mode=='cat'):
                df[column] = df[column].replace(str(val), 'Unknown')
            if (mode=='num'):
                df[column] = df[column].replace(str(val), np.nan)
   
    return df

def process_clean_cancerType(dataset, cancerType):
    if (cancerType == 'BRCA'):
        df = process_clean_BRCA(dataset)
    elif (cancerType == 'LUAD'):
        df = process_clean_LUAD(dataset)
    else:
        df = process_clean_anyCancerType(dataset)
    return df

def process_clean_anyCancerType(dataset):
    
    df=dataset.copy()
    
    pancancer_features = ['sample_code', 'AS', 'LOH_n_seg', 'LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'Study Abbreviation', 'Study Name', 'CNV_segs', 'CNV_frac', 'HRD', 'purity', 'Virus_CMV', 'Virus_EBV', 'Virus_HBV', 'Virus_HCV', 'Virus_HHV', 'Virus_HIV', 'Virus_HPV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV', 'Virus_BK_Polyomavirus', 'Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'leucocyte_fraction', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_miRNA', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0', 'Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils', 'totTCR_reads', 'totTCRa_reads', 'totTCRb_reads', 'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'mDNAsi', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Treg_cluster']
      
     #List from BRCA and LUAD
    followup_to_numeric = ['cent17_copy_number', 'death_days_to', 'her2_and_cent17_cells_count', 'her2_copy_number', 'anatomic_neoplasm_subdivision', 'history_immunosuppresive_dx', 'history_of_neoadjuvant_treatment', 'immunohistochemistry_positive_cell_score', 'karnofsky_performance_score', 'last_contact_days_to', 'lymph_nodes_examined_count', 'lymph_nodes_examined_he_count', 'lymph_nodes_examined_ihc_count', 'new_tumor_event_dx_days_to', 'on_haart_therapy_prior_to_cancer_diagnosis', 'performance_status_scale_timing' , 'tissue_retrospective_collection_indicator', 'venous_invasion', 'complete_response_observed', 'fluorescence_in_situ_hybridization_diagnostic_procedure_chromosome_17_signal_result_range', 'her2_erbb_pos_finding_cell_percent_category', 'immunohistochemistry_positive_cell_score', 'last_contact_days_to', 'new_tumor_event_dx_days_to', 'pos_finding_progesterone_receptor_other_measurement_scale_text', 'death_days_to']    
    
    #EXTRACT YEAR
    if ('form_completion_date' in df):
        df = extract_year(df, 'form_completion_date')
        df = df.drop('form_completion_date', axis=1)
    
    #OUTLIERS
    if ('cent17_copy_number' in df):
        df['cent17_copy_number'] = df['cent17_copy_number'].replace('polisomy                                                                                ', np.nan)
    if ('performance_status_scale_timing' in df):
        df['performance_status_scale_timing'] = df['performance_status_scale_timing'].replace('polisomy', np.nan)          
        
    for column in followup_to_numeric:
        if column in df.columns:
            df[column] = df[column].convert_objects(convert_numeric=True)
            df = clean_emptyString(df, column, "num")
            df = process_missing(df, column, 'median')
                
    categorical=df.select_dtypes(include=['object'])
    for cat in categorical:
        if cat not in pancancer_features:
            df = clean_emptyString(df, cat, "cat")
            df = process_missing(df, cat, 'Unknown')
            #df = create_dummies(df, cat)
   
    followup = []
    for col in df.columns:
        if col not in pancancer_features:
            followup.append(col)
            
    df = filterByVariance(df, followup)
    df = filterByVariance(df, pancancer_features)  
    
    return df    


def process_clean_BRCA(dataset):
    
    df=dataset.copy()
    
    pancancer_features = ['sample_code', 'AS', 'LOH_n_seg', 'LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'Study Abbreviation', 'Study Name', 'CNV_segs', 'CNV_frac', 'HRD', 'purity', 'Virus_CMV', 'Virus_EBV', 'Virus_HBV', 'Virus_HCV', 'Virus_HHV', 'Virus_HIV', 'Virus_HPV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV', 'Virus_BK_Polyomavirus', 'Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'leucocyte_fraction', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_miRNA', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0', 'Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils', 'totTCR_reads', 'totTCRa_reads', 'totTCRb_reads', 'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'mDNAsi', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Treg_cluster']
      
    followup_to_numeric = ['cent17_copy_number', 'death_days_to', 'her2_and_cent17_cells_count', 'her2_copy_number', 'anatomic_neoplasm_subdivision', 'history_immunosuppresive_dx', 'history_of_neoadjuvant_treatment', 'immunohistochemistry_positive_cell_score', 'karnofsky_performance_score', 'last_contact_days_to', 'lymph_nodes_examined_count', 'lymph_nodes_examined_he_count', 'lymph_nodes_examined_ihc_count', 'new_tumor_event_dx_days_to', 'on_haart_therapy_prior_to_cancer_diagnosis', 'performance_status_scale_timing' , 'tissue_retrospective_collection_indicator', 'venous_invasion']
    
    
    #EXTRACT YEAR
    if ('form_completion_date' in df.columns):
        df = extract_year(df, 'form_completion_date')
        df = df.drop('form_completion_date', axis=1)
    
    #OUTLIERS
    if ('cent17_copy_number' in df.columns):
        df['cent17_copy_number'] = df['cent17_copy_number'].replace('polisomy                                                                                ', np.nan)
    if ('performance_status_scale_timing' in df.columns):    
        df['performance_status_scale_timing'] = df['performance_status_scale_timing'].replace('polisomy', np.nan)  
    
    for column in followup_to_numeric:
        if column in df.columns:
            df[column] = df[column].convert_objects(convert_numeric=True)
            df = clean_emptyString(df, column, "num")
            df = process_missing(df, column, 'median')
                
    categorical=df.select_dtypes(include=['object'])
    for cat in categorical:
        if cat not in pancancer_features:
            df = clean_emptyString(df, cat, "cat")
            df = process_missing(df, cat, 'Unknown')
            #df = create_dummies(df, cat)
   
    followup = []
    for col in df.columns:
        if col not in pancancer_features:
            followup.append(col)
            
    df = filterByVariance(df, followup)
    df = filterByVariance(df, pancancer_features)  
    
    return df    


def process_clean_LUAD(dataset):
    
    df=dataset.copy()
    
    pancancer_features = ['sample_code', 'AS', 'LOH_n_seg', 'LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'Study Abbreviation', 'Study Name', 'CNV_segs', 'CNV_frac', 'HRD', 'purity', 'Virus_CMV', 'Virus_EBV', 'Virus_HBV', 'Virus_HCV', 'Virus_HHV', 'Virus_HIV', 'Virus_HPV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV', 'Virus_BK_Polyomavirus', 'Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'leucocyte_fraction', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_miRNA', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0', 'Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils', 'totTCR_reads', 'totTCRa_reads', 'totTCRb_reads', 'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'mDNAsi', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Treg_cluster']
      
        
    followup_to_numeric = ['complete_response_observed', 'fluorescence_in_situ_hybridization_diagnostic_procedure_chromosome_17_signal_result_range', 'her2_erbb_pos_finding_cell_percent_category', 'immunohistochemistry_positive_cell_score', 'last_contact_days_to', 'new_tumor_event_dx_days_to', 'pos_finding_progesterone_receptor_other_measurement_scale_text', 'death_days_to']
    

    #OUTLIERS
    if ('type' in df):
        df = df.drop(['type'], axis=1)
    
    for column in followup_to_numeric:
        if column in df.columns:
            df[column] = df[column].convert_objects(convert_numeric=True)
            df = clean_emptyString(df, column, "num")
            df = process_missing(df, column, 'median')
                
    categorical=df.select_dtypes(include=['object'])
    for cat in categorical:
        if cat not in pancancer_features:
            df = clean_emptyString(df, cat, "cat")
            df = process_missing(df, cat, 'Unknown')
   
    followup = []
    for col in df.columns:
        if col not in pancancer_features:
            followup.append(col)
            
    df = filterByVariance(df, followup)
    df = filterByVariance(df, pancancer_features)  
    
    return df    


def extract_year(dataset, column):
    df=dataset.copy()
    
    year="{}_year".format(column)
    
    df[year] = df[column].str.split('-').str[0]
    
    return df

def process_missing(dataset, column, mode):
    """Handle various missing values from the data set
    """
    df=dataset.copy()
    if column in dataset.columns:
        if (mode=="zero"):
            df[column] = df[column].fillna(0)
        if (mode=="mean"):
            df[column] = df[column].fillna(df[column].mean())
        if (mode=="median"):
            df[column] = df[column].fillna(df[column].median())        
        if (mode=="Unknown"):
            df[column] = df[column].fillna("Unknown")
    return df


def process_categorizeAge(dataset):
    """Process the Age column into pre-defined 'bins' 
    """
    df=dataset.copy()
    cut_points = [-10,5,18,35,65,100]
    label_names = ["Unknown","Child","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["age_at_initial_pathologic_diagnosis"],cut_points,labels=label_names)
    df = df.drop("age_at_initial_pathologic_diagnosis", axis=1)
    return df


def create_dummies(dataset,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    Usage
    ------

    train = create_dummies(train,"Age")
    """
    df=dataset.copy()
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

def process_logNormalize(dataset, chosen_column=None):
    
    df = dataset.copy()

    if (chosen_column == None):
        features_to_logNormalize = ['LOH_n_seg','LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'CNV_segs', 'CNV_frac', 'HRD', 'Virus_HCV', 'Virus_HPV', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'T.cells.CD4.memory.resting','totTCRa_reads', 'totTCRb_reads', 'mDNAsi']

        for column in features_to_logNormalize:
            df = process_normalize(df,column,"log")
    else:
        df = process_normalize(df,chosen_column,"log")
        log="{}_log".format(chosen_column)
        df[chosen_column] = df[log]
        df = df.drop([log], axis=1)
        
    return df


def process_removeUnknown(dataset, keepImportantUnknown):
    
    df=dataset.copy()
    Unknown_cols = [col for col in df.columns if 'Unknown' in col]

    if keepImportantUnknown:   
        Unknown_cols = [col for col in df.columns if 'Unknown' in col]
        #we choose those that may add some info, not all Unknown cols
        important_unknown_cols = ['vital_status_Unknown', 'tumor_status_Unknown', 'tumor_status_NA']
        for col in Unknown_cols:
            if col not in important_unknown_cols:
                df = df.drop(col, axis=1)                   
    else:
        for column in Unknown_cols:            
            df = df.drop(column, axis=1)   
        
    return df

def process_normalize(dataset,column_name, mode):
    """Normalize Continuous Columns 
     
    mode n => normal normalization
    mode z => z normalization
    
    """
    df=dataset.copy()
    
    if (mode=="n"):
        column=df[column_name]
        new="{}_norm".format(column_name)
        df[new]=(column-column.min())/(column.max()-column.min())
        df[column_name]=df[new]
        df=df.drop([new], axis=1)
    if (mode=="log"):
        column=df[column_name]
        new="{}_plusone".format(column_name)
        log="{}_log".format(column_name)
        df[new]=1+column
        df[log]=np.log(df[new])
        #df[column_name]=df[log]
        df=df.drop([new], axis=1)
        #df=df.drop([log], axis=1)
        
    if (mode=="z"):
        column=df[column_name]
        new="{}_norm".format(column_name)
        avg=column.mean()
        var=df.loc[:,column_name].var()
        df[new]=(column-avg)/var
        df[column_name]=df[new]
        df=df.drop([new], axis=1)
        
    return df


    
def plot_confusion_matrix(cnf_matrix, classesNames, normalize=False,
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)

    if normalize:
        soma = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / soma
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix, without normalization'

    print(cm)

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classesNames))
    plt.xticks(tick_marks, classesNames, rotation=45)
    plt.yticks(tick_marks, classesNames)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def create_class(dataset):
    df=dataset.copy()
    df["Neoantigens_total"]=df['numberOfBindingExpressedPMHC']+df['Neoantigen_num']
    df['Neoantigens_total'] = df['Neoantigens_total'].convert_objects(convert_numeric=True)

    return df

def create_multiclass(dataset, normalize=False):
    df=dataset.copy()
    #df['Immune-evasion'] = np.where(df['Neoantigens_total'] == 1, 'Neoantigens_total', np.where(df['Treg_cluster'] == 1, 1, 0))
    
    print("STARTING CLASS BINARIZATION OF DATASET WITH SIZE:", df.shape[0], df.shape[1])
    print()          
    
    print("CLASS: Treg_cluster") 
    if (normalize):
        df = df[df['Treg_cluster'] != 0]
        df['Treg_cluster'] = np.where(df['Treg_cluster'] == 0.5, 0, 1)            
    else:            
        df = df[df['Treg_cluster'] != -1]
    print("IMPORTANT: Number of rows shortened to", df.shape[0], "due to Unknown entries in the chosen class")
    print()
    
    print("CLASS: Neoantigens_total")  
    #Balanced cut
    print("IMPORTANT: Classe balanced: 50% of samples with higher values labeled 1 and 50% of samples with lower value labeled 0")
    print()
    df["temp"] = pd.qcut(df['Neoantigens_total'],2, labels=[0,1])
    df['Neoantigens_total'] = np.where(df["temp"] == 0, 0, 1)
    df = df.drop(["temp"], axis=1)
    
    col1 = 'Neoantigens_total'
    col2 = 'Treg_cluster'
    conditions = [ (df[col1] == 0) & (df[col2] == 1), df[col1] == 0, df[col2] == 1 ]
    choices = [ "Neoantigens depletion &  IS abundance", "Neoantigens depletion" , "IS abundance"]
    #choices = [ 0, 1 , -1]
    df["Immune_evasion"] = np.select(conditions, choices, default="None")

    print("Immune Evasion unique values")
    print(df["Immune_evasion"].value_counts())
    print()
         
    return df


def create_sumFeatures(dataset):
    df=dataset.copy()
    
    df["Mutations_total"]=df['numberOfNonSynonymousSNP']+df['MutationsSilent']+df['MutationsNonSilent']+df['Indel_num']
    df["Virus_total"]= df['Virus_HCV']+df['Virus_HPV']+df['Virus_BK_Polyomavirus']+df['Virus_CMV']+df['Virus_EBV']+df['Virus_HBV']+df['Virus_HHV'] +df['Virus_HIV']+df['Virus_HTLV']+df['Virus_MCV']+df['Virus_SV']+df['Virus_Saimiriine_Herpesvirus']             
    df["Immune_cells_total"] = df['B.cells.memory']+df['B.cells.naive']+df['B.cells.naive']+df['T.cells.CD4.memory.resting']+df['Dendritic.cells.activated']+df['Dendritic.cells.resting']+df['Eosinophils']+df['Macrophages.M0']+df['Macrophages.M1']+df['Macrophages.M2']+df['Mast.cells.resting']+df['Mast.cells.activated']+df['Monocytes']+df['NK.cells.activated']+df['NK.cells.resting']+df['Neutrophils']+df['Plasma.cells']+df['T.cells.CD4.memory.activated']+df['T.cells.CD4.memory.resting']+df['T.cells.CD4.naive']+df['T.cells.CD8']+df['T.cells.follicular.helper']+df['T.cells.gamma.delta']+df['T.cells.regulatory..Tregs.']
                                                                             
    return df


def getDataCls(dataset, chosen_class, normalize=False, dummyCat=True, remove_FP = True, user_Features_Include = None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False):
    
    warnings=[]
    
    print("CHOSEN CLASS:", chosen_class)
    print()
    df = dataset.copy()
    X = dataset.copy() 
    
    if (chosen_class=="IS"):
        chosen_class = 'Treg_cluster'
        try:
            y = df[chosen_class]
        except:
            print("ERROR: chosen class cannot be predicted by chosen population.")
            sys.exit(1)
            
    if (chosen_class=="HA"):
        chosen_class = 'Neoantigens_total'
        try:
            X = create_class(X)
        except:
            print("ERROR: chosen class cannot be predicted by chosen population.")
            sys.exit(1)
            
    if (chosen_class=="multilabel"):
        chosen_class = ['Neoantigens_total', 'Treg_cluster']
        try:
            X = create_class(X)
        except:
            print("ERROR: chosen class cannot be predicted by chosen population.")
            sys.exit(1)
            
    if (chosen_class=="multiclass"):
        chosen_class = 'Immune_evasion'
        try:
            X = create_class(X)
            X = create_multiclass(X, normalize) 
        except:
            print("ERROR: chosen class cannot be predicted by chosen population.")
            sys.exit(1)

    #Considering any attribute a possible class
    if remove_FP:
        if (chosen_class == "Immune_evasion"):
            X, warning1 = remove_fake_predictors(X,'Neoantigens_total')
            X, warning2 = remove_fake_predictors(X,'Treg_cluster')
            X = X.drop(['Neoantigens_total'], axis=1)
            X = X.drop(['Treg_cluster'], axis=1)
            for warning in [warning1, warning2]:
                if isinstance(warning, str):
                    warnings.append(warning)
                else:
                    for w in warning:
                        warnings.append(w)
        else:
            X, warning = remove_fake_predictors(X, chosen_class)
            if isinstance(warning, str):
                warnings.append(warning)
            else:
                for w in warning:
                    warnings.append(w)

    
    if isinstance(chosen_class, str):
        if (chosen_class != "Immune_evasion"):
            X, warning = make_binary_class(X, chosen_class, normalize=normalize, binary_class_threshold = binary_class_threshold)
        y = X[chosen_class].copy()
        X = X.drop([chosen_class], axis=1)
        warnings.append(warning)
    else:
        for col in chosen_class:
            X, warning = make_binary_class(X, col, normalize=normalize, binary_class_threshold = binary_class_threshold)    
            warnings.append(warning)
        y = X[chosen_class].copy()     
        X = X.drop(chosen_class, axis=1)


    #We remove unknown categories
    X = process_removeUnknown(X, keepImportantUnknown)
    
    return X, y, warnings

def getDataReg(dataset, chosen_class, normalize=False, dummyCat=True, remove_FP = True, user_Features_Include=None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False, visualize=False):
    
    X = dataset.copy()
    
    y = ""
    cls = ""
    
    if (chosen_class=="IS"):
        chosen_class = 'Treg_cluster'     
        
    if (chosen_class=="HA"):
        chosen_class = 'Neoantigens_total'
        X = create_class(X)
         
    #Considering any attribute a possible class
    if remove_FP:
        X = remove_fake_predictors(X,chosen_class) 
    y = X[chosen_class].copy()
    X = X.drop([chosen_class], axis=1)

    #We remove unknown categories
    X = process_removeUnknown(X, keepImportantUnknown)

    return (X, y)


def train_auto_classifier(dataset, chosen_class, normalize=False, dummyCat=True, boxplot=False, user_FP = None, remove_FP = True, user_Features_Include=None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False, visualize=False):
    
    X, y , warnings= getDataCls(dataset, chosen_class, normalize, dummyCat, remove_FP, user_Features_Include, user_Features_Exclude, binary_class_threshold, keepImportantUnknown)
    
    labels = y.unique()
    trX, tsX, trY, tsY = train_test_split(X, y, train_size=0.7, stratify=y)  
    
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
    )

    automl.fit(trX.copy(), trY.copy())
    automl.refit(trX.copy(), trY.copy())
    
    print(automl.show_models())

    predictions = automl.predict(tsX)
    print("Accuracy score", accuracy_score(tsY, predictions))

def train_regressor(dataset, chosen_class, normalize=False, dummyCat=True, boxplot=False, remove_FP = True, user_Features_Include=None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False, visualize=False):
    X = dataset.copy()
    
    y = ""
    cls = ""
    
    if (chosen_class=="IS"):
        chosen_class = 'Treg_cluster'     
        
    if (chosen_class=="HA"):
        chosen_class = 'Neoantigens_total'
        X = create_class(X)
         
    #Considering any attribute a possible class
    if remove_FP:
        X = remove_fake_predictors(X,chosen_class) 
    y = X[chosen_class].copy()
    X = X.drop([chosen_class], axis=1)

    #We remove unknown categories
    X = process_removeUnknown(X, keepImportantUnknown)
    
    models=[]
    models.append(('Random Forest', RandomForestRegressor(random_state=123)))
    models.append(('LR', LinearRegression()))
    models.append(('DT', DecisionTreeRegressor(random_state=123)))
    
    results=[]
    names=[]
    #evaluate each model in turn
    for name, model in models:
    
        #scoring = {'prec_macro': 'precision_macro', 'rec_macro': 'recall_macro'}   

        scores = cross_validate(model, X, y, cv=10)

        #To print
        individual_results = scores['test_score']
        
        print("MSE of ",name ,": %0.2f (+/- %0.2f)" % (individual_results.mean(), individual_results.std() * 2 ))

        #Tu use in boxplot

        results.append(individual_results)
        names.append(name)

        
    
#FOR DEVELOPMENT
def train_classifier(dataset, chosen_class, classifier="None", normalize=False, dummyCat=True, boxplot=False, user_FP = None, remove_FP = True, user_Features_Include=None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False, visualize=False):
    
    X, y , warnings = getDataCls(dataset, chosen_class, normalize, dummyCat, remove_FP, user_Features_Include, user_Features_Exclude, binary_class_threshold, keepImportantUnknown)

     
    #only for development
    if (classifier == "None"): 
        # prepare models
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('CART', DecisionTreeClassifier(random_state=123)))
        models.append(('RandomForest', RandomForestClassifier(random_state=123)))
        models.append(('Naive Bayes Gaussian', GaussianNB()))
        if dummyCat:
            models.append(('Naive Bayes Bernoulli', BernoulliNB()))
                  

        # variables to store
        results_acc = []
        results_aucroc = []
        results_prec = []
        results_rec = []
        results_aucroc = []
        names = []
        final_acc=[]
        final_prec=[]
        final_rec=[]
        final_aucroc=[]
        final_results=[]
        scoring = {'acc': 'accuracy','prec_macro': 'precision_macro', 'rec_micro': 'recall_macro', 'roc_auc': 'roc_auc'}   
        
        #evaluate each model in turn
        for name, model in models:
            
            categorical_variables = ['Study Abbreviation', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_miRNA','Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'Age_categories','Subtype_protein', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2']  
            #categorical_variables = X.select_dtypes(include = 'object').columns

            X_cat_columns=[]
            
            for original_col in categorical_variables:
                for col in X.columns:
                    if original_col in col:
                        X_cat_columns.append(col)
                               
            X_cat = X[X_cat_columns]                          
            X_num = X.copy().drop(X_cat_columns, axis=1)
            
            if (name=="Naive Bayes Gaussian"):
                scores = cross_validate(model, X_num, y, scoring=scoring, cv=10, return_train_score=True)

            elif (name=="Naive Bayes Bernoulli"):
                scores = cross_validate(model, X_cat, y, scoring=scoring, cv=10, return_train_score=True)

            else:
                scores = cross_validate(model, X, y, scoring=scoring, cv=10, return_train_score=True)
            
            #To print
            acc_results = scores['test_acc']
            prec_results = scores['test_prec_macro']
            rec_results = scores['test_rec_micro']
            aucroc_results = scores['test_roc_auc']
            
            #Tu use in boxplot
            results_acc.append(acc_results)
            results_prec.append(prec_results)
            results_rec.append(rec_results)
            results_aucroc.append(aucroc_results)
            names.append(name)
            
            acc = "%s, %s: %f (%f)" % (name, "Accuracy", acc_results.mean(), acc_results.std())
            print(acc)
            prec = "%s, %s: %f (%f)" % (name, "Precision", prec_results.mean(), prec_results.std())
            print(prec)
            rec = "%s, %s: %f (%f)" % (name, "Recall", rec_results.mean(), rec_results.std())
            print(rec)
            aucroc = "%s, %s: %f (%f)" % (name, "ROC-AUC", aucroc_results.mean(), aucroc_results.std())
            print(aucroc)
            
            intermediate_results=[acc_results.mean(),prec_results.mean(),rec_results.mean(),aucroc_results.mean()]
            final_results.append(intermediate_results)
        
        print()
        print("Results by measure: acc, prec, rec, aucroc (horizontal) VS LR , CART , RF, NB (vertical)")

        for result in final_results:
            print(result)
        
        if (boxplot):
            # boxplot algorithm comparison
            fig = plt.figure()
            fig.suptitle('Algorithm Accuracy Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(results_acc)
            ax.set_xticklabels(names)
            #ax.set_yticks(np.arange(0, 1, step=0.2))

            plt.show()     
      
    else:
        if (classifier=="RF"):

            cls = RandomForestClassifier(random_state=123)
            train_classifier_testSplit(X, y, cls, classifier,visualize=visualize)

        if (classifier=="LR"):
            cls = LogisticRegression()
            train_classifier_testSplit(X, y, cls, classifier,visualize=visualize)

        if (classifier=="DT"):
            cls = DecisionTreeClassifier() 
            train_classifier_testSplit(X, y, cls, classifier,visualize=visualize)

        if (classifier=="NB"):
            
            categorical_variables = ['Study Abbreviation', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_miRNA','Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'Age_categories','Subtype_protein', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Treg_cluster']  
            X_cat_columns=[]
            for original_col in categorical_variables:
                for col in X.columns:
                    if original_col in col:
                        X_cat_columns.append(col)
                               
            X_cat = X[X_cat_columns]                          
            X_num = X.copy().drop(X_cat_columns, axis=1)
            
            cls_num = GaussianNB()  #for continuous variables
            cls_cat = BernoulliNB()
            
            print("CATEGORICAL NAIVE BAYES: Training with ", X_cat.shape[1], " columns")
            print()
            train_classifier_testSplit(X_cat, y, cls_cat, "NBB", visualize=visualize)
            print("NUMERICAL NAIVE BAYES: Training with ", X_num.shape[1], " columns")
            print()
            train_classifier_testSplit(X_num, y, cls_num, "NBG", visualize=visualize)

def tree_depth(tree):   
   
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    depth = np.amax(node_depth)
    #("The binary tree structure has {} nodes with {} levels".format(str(n_nodes), str(depth)))
    #return Tree
    return (n_nodes, depth)
    #print("The binary tree structure has ", str(n_nodes)," nodes with ",str(depth)," levels.")
    
def prune_minsamplesleaf(decisiontree, min_samples_leaf = 1):
    if decisiontree.min_samples_leaf >= min_samples_leaf:
        raise Exception('Tree already more pruned')
    else:
        decisiontree.min_samples_leaf = min_samples_leaf
        tree = decisiontree.tree_
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            if n_samples <= min_samples_leaf:
                tree.children_left[i]=-1
                tree.children_right[i]=-1

def prune_depth(tree, depth):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    
    to_erase = []
    i=0
    while (i<len(node_depth)):
        if (node_depth[i]>depth):
            to_erase.append(i)
        i+=1
    for n in to_erase:
        children_left[n] =-1
        children_right[n] =-1
        
def visualize_tree(model, X, y):
    estimator = model.estimators_

    from sklearn.tree import export_graphviz
    # Export as dot file
    from subprocess import call
    import os
    import six
    import pydot
    from sklearn.tree import export_graphviz
    from sklearn import tree
    from IPython.display import Image
   
    i_tree = 0
    for tree_in_forest in estimator:
        export_graphviz(tree_in_forest,out_file='tree.dot',
        feature_names=X.columns.tolist(), class_names = str(y.unique()),rounded=True, filled=True)
        (graph,) = pydot.graph_from_dot_file('tree.dot')
        name = 'tree' + str(i_tree)
        graph.write_png(name+  '.png')
        os.system('dot -Tpng tree.dot -o tree.png')
        i_tree +=1
     
    
def train_classifier_testSplit(features, classes, cls, classifier, visualize=False):
    
    X = features.copy()
    y = classes.copy()
    
    if (len(classes.shape)==1):
        labels = y.unique()
    trX, tsX, trY, tsY = train_test_split(X, y, train_size=0.7, stratify=y)    
    model = cls.fit(trX, trY)
    predY = model.predict(tsX)
    accuracy = accuracy_score(tsY, predY)
    
    if visualize:
        if (classifier == "RF"):
            visualize_tree(model, X, y)
        else:
            estimator=model
            n, d=tree_depth(estimator)
            print("The tree structure has ", n," nodes with ",d," levels.")
            graph = Source(export_graphviz(estimator, feature_names=X.columns.tolist(), class_names=[str(x) for x in  estimator.classes_.tolist()] , out_file=None, filled = True))
            display(SVG(graph.pipe(format='svg')))
            graph.format = 'png'
            save_name = "Tree_NoFPremoval"
            graph.render(save_name, view=False)
    #Calculate AUROC
    #Predict class probability, outputs array of shape = [n_samples, n_classes]
    prob_y= cls.predict_proba(X)
    
    
    if (len(classes.shape)==1):
        #Keep only positive class
        prob_y = [p[1] for p in prob_y]
        
        cnf_matrix = confusion_matrix(tsY, predY, labels)
        plot_confusion_matrix(cnf_matrix, labels) 
        
        if (len(classes.unique()) < 3):
            auroc = roc_auc_score(y, prob_y)
   
    else:
        #Keep only positive class on both classe        
        auroc =[]
        i=0
        for col in y.columns:
            prob_y_i = [p[1] for p in prob_y[i]]
            auroc.append(roc_auc_score(y[col], prob_y_i))
            i+=1             
    print()
    print("Accuracy:", accuracy)
    print()
    if (len(classes.unique()) < 3):
        print("AUCROC:", auroc)
        print()

    if ((classifier=="RF") or (classifier=="DT")):
        print("Features Importance")
        feature_importances = pd.DataFrame(model.feature_importances_, index = trX.columns, columns=['importance']).sort_values('importance', ascending=False)
        print(feature_importances)
        df_filtered = feature_importances[feature_importances['importance'] != 0]
        print(df_filtered.shape[0],df_filtered.shape[1])
        print(df_filtered)
        
    if (classifier=="LR"):
        feature_importance = model.coef_[0]
        feature_importance = 100.0 * (abs(feature_importance) / abs(feature_importance.max()))
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        featfig = plt.figure(figsize=(10,30))
        featax = featfig.add_subplot(1, 1, 1)
        featax.barh(pos, feature_importance[sorted_idx], align='center')
        featax.set_yticks(pos)
        featax.set_yticklabels(np.array(trX.columns)[sorted_idx], fontsize=8)
        featax.set_xlabel('Relative Feature Importance')

        plt.tight_layout()   
        plt.show()
        
    if (classifier=="NBB"):
        print(model.feature_log_prob_)
    if (classifier=="NBG"):
        print(model.theta_)
        print(model.sigma_)

    
def make_binary_class(dataset, chosen_column, normalize=False, binary_class_threshold = None):
    df=dataset.copy()
    warning=""
    print("STARTING CLASS BINARIZATION OF DATASET WITH SIZE:", df.shape[0], df.shape[1])
    print()          
    new="{}_bin".format(chosen_column)
    
    # TO DO
    domain_knowledge_thresholds=[]
   
    data = {'First' : 0., 'Second' : 1, 'Third' : 2.}
    s = pd.Series(data,index=['First','Second','Third'])
    
    if (binary_class_threshold == None):
        
        ###PRINT
        print("CLASS:", chosen_column) 
        #number of underscores in feature name
        underscores = chosen_column.count("_")
        #split all parts of feature name and join all but last
        original_chosen_column = "_".join(chosen_column.split("_", underscores)[:underscores])
        #create column with Unknown as last word after underscore
        unknown_chosen_column = "{}_Unknown".format(original_chosen_column)
        NA_chosen_column = "{}_NA".format(original_chosen_column)
        
        #This is made before removing all Unknown columns of the dataset
        if unknown_chosen_column in df.columns:
            df = df[df[unknown_chosen_column] == 0]
            df = df[df[NA_chosen_column] == 0]
            df[new] = np.where(df[chosen_column] == 1, 1, 0)
            print("IMPORTANT: Number of rows shortened to", df.shape[0], "due to Unknown entries in the chosen class")
            print()
            warning = "IMPORTANT: Number of rows shortened to {} due to Unknown entries in the chosen class".format(df.shape[0])

            df = balance_dataset(df, new)
        
        elif chosen_column in domain_knowledge_thresholds:
            if (normalize):
                cut = []
                #df[new] = ... search threshold in data dictionary
                #df = balance_dataset(df, chosen_column)
        elif (chosen_column =='Virus_HPV' ):
            #Balanced cut
            print("IMPORTANT: Classe balanced: 50% of samples with higher values labeled 1 and 50% of samples with lower value labeled 0")
            print()
            warning="IMPORTANT: Classe balanced: 50% of samples with higher values labeled 1 and 50% of samples with lower value labeled 0"
            df["temp"] = pd.qcut(df[chosen_column],2, labels=[0,1])
            df[new] = np.where(df["temp"] == 0, 0, 1)
            df = df.drop(["temp"], axis=1)
            
        elif 'Virus' in chosen_column:
            df[new] = np.where(df[chosen_column] == 0, 0, 1)
            df = balance_dataset(df, new)
            
        elif (chosen_column == 'OS'):
            if normalize:
                df = df[df['OS'] != 0]
                df[new] = np.where(df[chosen_column] == 0.5, 0, 1)            
            else:            
                df = df[df['OS'] != -1]
                df[new] = np.where(df[chosen_column] != -1, df[chosen_column], np.nan)
            print("IMPORTANT: Number of rows shortened to", df.shape[0], "due to Unknown entries in the chosen class")
            print()
            warning = "IMPORTANT: Number of rows shortened to {} due to Unknown entries in the chosen class".format(df.shape[0])

            df = balance_dataset(df, new)
        
        elif (chosen_column == 'Treg_cluster'):

            if (normalize):
                df = df[df['Treg_cluster'] != 0]
                df[new] = np.where(df[chosen_column] == 0.5, 0, 1)            
            else:            
                df = df[df['Treg_cluster']!=-1]
                df[new] = np.where(df[chosen_column] != -1, df[chosen_column], np.nan)
            print("IMPORTANT: Number of rows shortened to", df.shape[0], "due to Unknown entries in the chosen class")
            print()
            warning = "IMPORTANT: Number of rows shortened to {} due to Unknown entries in the chosen class".format(df.shape[0])

            df = balance_dataset(df, new)
            
        elif (chosen_column == 'mDNAsi'):
            df[new] = np.where(df[chosen_column] > 0.35, 1, 0)            
            print("IMPORTANT: Number of rows shortened to", df.shape[0], "due to Unknown entries in the chosen class")
            print()
            warning = "IMPORTANT: Number of rows shortened to {} due to Unknown entries in the chosen class".format(df.shape[0])

            df = balance_dataset(df, new)
            
        elif (df[chosen_column].unique().shape[0] == 2):
            df = df[df[chosen_column].notnull()]
            df[new] = df[chosen_column]
            print("IMPORTANT: Number of rows shortened to", df.shape[0], "due to Unknown entries in the chosen class")
            print()
            warning = "IMPORTANT: Number of rows shortened to {} due to Unknown entries in the chosen class".format(df.shape[0])
            df = balance_dataset(df, new)              
               
              
        else:            
            #Balanced cut
            print("IMPORTANT: Classe balanced: 50% of samples with higher values labeled 1 and 50% of samples with lower value labeled 0")
            print()
            warning="IMPORTANT: Classe balanced: 50% of samples with higher values labeled 1 and 50% of samples with lower value labeled 0"
            df["temp"] = pd.qcut(df[chosen_column],2, labels=[0,1])
            df[new] = np.where(df["temp"] == 0, 0, 1)
            df = df.drop(["temp"], axis=1)
    else:
        cut = []
        #df[new] = ...
             
    df[chosen_column]=df[new]
    df=df.drop([new], axis=1)
    
    return df, warning  

def balance_dataset(dataset, chosen_column):
    df=dataset.copy()
    df_zero = df[df[chosen_column]==0]
    df_one = df[df[chosen_column]==1]
    print("SHAPES 0 and 1:", df_zero.shape[0], df_one.shape[0])
        
    if (df_zero.shape[0] > 1.5*df_one.shape[0]):   
        df = upSample_minority(df, 1, chosen_column)
    elif (df_one.shape[0] > 1.5*df_zero.shape[0]):
        df = upSample_minority(df, 0, chosen_column)
    else:
        print("NO BALANCING NEEDED FOR CLASS: ", chosen_column)
        print(df.info())
    return df
        

def upSample_minority(dataset, minority_class, chosen_column):
        
    df=dataset.copy()
    print("STARTING UPSAMPLING OF DATASET WITH SIZE:", df.shape[0], df.shape[1])
    print()
    print("THIS IS MINORITY CLASS:", minority_class)
    if (minority_class==0):
        majority_class=1
    else:
        majority_class=0
           
    df_majority = df[df[chosen_column]==majority_class]
    df_minority = df[df[chosen_column]==minority_class]
    print("SHAPES maj and min:", df_majority.shape[0], df_minority.shape[0])

    n_samples= df_majority.shape[0]
    
    #Upsample minority
    df_minority_upsampled = resample(df_minority, replace=True, n_samples = n_samples, random_state=123)
    print("SHAPES min upsampled:", df_minority_upsampled.shape[0])
    
    #Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
   
    print("IMPORTANT: Number of rows increased to", df_upsampled.shape[0], "due to balancement techniques")
    print()
    
    return df_upsampled

def remove_fake_predictors(dataset, chosen_column):
    df=dataset.copy()
           
    if isinstance(chosen_column, str):  
        df, warning = remove_fake_predictors_aux(df, chosen_column)
    else:
        warning=[]
        for col in chosen_column:
            df, w = remove_fake_predictors_aux(df, col) 
            warning.append(w)    

    return df, warning

def remove_fake_predictors_aux(dataset, chosen_column):
    df =dataset.copy()
    
    mutations_columns = ['Immunogenic_indel_num','numberOfImmunogenicMutation', 'numberOfNonSynonymousSNP', 'Indel_num', 'MutationsNonSilent', 'MutationsSilent', 'Neoantigens_total', 'numberOfBindingExpressedPMHC', 'Neoantigen_num', 'Immunogenic_indel_num_log','numberOfImmunogenicMutation_log', 'numberOfNonSynonymousSNP_log', 'Indel_num_log', 'MutationsNonSilent_log', 'MutationsSilent_log', 'Neoantigens_total_log', 'numberOfBindingExpressedPMHC_log', 'Neoantigen_num_log', 'Mutations_total']    
    immune_cells_columns = ['Immune_cells_total', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0','Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils']        

    virus_columns = ['Virus_HCV','Virus_HPV', 'Virus_BK_Polyomavirus','Virus_CMV','Virus_EBV', 'Virus_HBV','Virus_HHV','Virus_HIV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV' ,'Virus_Saimiirine_Herpesvirus','Virus_HCV_log','Virus_HPV_log', 'Virus_total']

    survival_columns = ['OS', 'vital_status_Alive', 'vital_status_Dead', 'vital_status_Unknown', 'vital_status_NA', 'tumor_status_TUMOR FREE', 'tumor_status_WITH TUMOR', 'tumor_status_Unknown']
    

    correlations_data = df.corr()[chosen_column]
    cols = correlations_data.index.tolist()
    correlated = []
    for col in cols:
        if (col != chosen_column):
            if (correlations_data[col] > 0.75):
                correlated.append(col)

    if chosen_column in mutations_columns:
        for column in mutations_columns:
            if (column != chosen_column):
                if column not in correlated:
                    correlated.append(column)
    
    if chosen_column in mutations_columns:
        column = 'Mutations_total'            
        if (column != chosen_column):
            if column not in correlated:
                correlated.append(column)
    
    if chosen_column in virus_columns:
        for column in virus_columns:
            if (column != chosen_column):
                if column not in correlated:
                    correlated.append(column)
                    
    if chosen_column in virus_columns:
        column = 'Virus_total'
        if (column != chosen_column):
            if column not in correlated:
                correlated.append(column)
                
    if chosen_column in survival_columns:
        for column in survival_columns:
            if (column != chosen_column):
                if column not in correlated:
                    correlated.append(column)

    if chosen_column in immune_cells_columns:
        column = 'Immune_cells_total'
        if (column != chosen_column):
            if column not in correlated:
                correlated.append(column)
                
    for column in correlated:
        if column in df.columns:
            df = df.drop([column], axis=1)        

    print()
    print("CORRELATED VARIABLES REMOVED:", correlated)
    warning="CORRELATED VARIABLES REMOVED: {}".format(correlated)
    print()
    
    return df, warning

def importantFeatures(X, y, chosen_class, mode):
    
    if (mode == "LR"):       
        lr = LogisticRegression()
        lr.fit(X,y)

        coefficients = lr.coef_

        feature_importance = pd.Series(coefficients[0], index=X.columns)
        ordered_feature_importance = feature_importance.abs().sort_values()
        ordered_feature_importance = ordered_feature_importance[ordered_feature_importance > 0.2]
        return ordered_feature_importance
    
    if (mode == "PCA"):
        for feature in list(X.columns):
            if isinstance(chosen_class, str):
                if (chosen_class == 'Immune_evasion'):
                    chosen_class = ['Neoantigens_total', 'Treg_cluster']
                    for col in chosen_class:
                        if (feature!=col):
                            XZ=process_normalize(X, feature, "z")
                elif (feature!=chosen_class):
                    XZ=process_normalize(X, feature, "z")
            else:
                for col in chosen_class:
                    if (feature!=col):
                        XZ=process_normalize(X, feature, "z")
                        
        pca_trafo = PCA().fit(XZ);
        pca_data=pca_trafo.transform(XZ)
        feats = get_important_features_PCA(pca_data, pca_trafo.components_, XZ.columns.values)        
        
        return feats
    
    if (mode=="RFECV"):
        clf = LogisticRegression()
        #clf = RidgeClassifierCV()
        selector = RFECV(clf,cv=10)
        selector.fit(X,y)

        optimized_columns = X.columns[selector.support_]
        
        return optimized_columns

def get_important_features_PCA(transformed_features, components_, columns):
    num_columns = len(columns)
    # Scale the principal components by the max value in the transformed set
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])
    # Sort each original column by it's length
    impt_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    impt_features = sorted(zip(impt_features.values(),impt_features.keys()),reverse=True)
    feats=[]
    for feat in impt_features:
        if (feat[0]>0.5):
            feats.append(feat[1])
            
    return feats

def featureSelection(dataset, chosen_class, normalize=False, dummyCat=True, remove_FP = True, user_Features_Include = None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False):

    #Create X and y just to obtain list of selected features but return full dataset with only droped columns, no getDataCls changes
    X, y, warnings= getDataCls(dataset, chosen_class, normalize=normalize, dummyCat=dummyCat, remove_FP = remove_FP, user_Features_Include = user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = binary_class_threshold, keepImportantUnknown=keepImportantUnknown)
    
    if (chosen_class != 'multilabel'):
        IF_logRegression=importantFeatures(X, y, chosen_class, "LR")
        IF_PCA=importantFeatures(X, y, chosen_class, "PCA")    
        IF_RFECV=importantFeatures(X, y, chosen_class, "RFECV")

        df=dataset.copy()
        for column in X.columns:
            if column not in IF_logRegression:
                if column not in IF_PCA:
                    if column not in IF_RFECV:
                        if column not in ['Treg_cluster','Neoantigen_num','numberOfBindingExpressedPMHC']:
                            if (column != chosen_class):
                                df=df.drop([column], axis=1)
    else:
        IF_PCA=importantFeatures(X, y, chosen_class, "PCA")    
        df=dataset.copy()
        for column in X.columns:
            if column not in IF_PCA:
                if column not in ['Treg_cluster','Neoantigen_num','numberOfBindingExpressedPMHC']:
                    if (column != chosen_class):
                        df=df.drop([column], axis=1)
    return df      

      
def train_XGBoost(dataset, chosen_class, prediction="classification", train="cv", tunning=False, normalize=False, dummyCat=True, boxplot=False, remove_FP = True, user_Features_Include=None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False):
    
    df = dataset.copy()
    #Clean data for XGBoost because of error: feature_names may not contain [, ] or <
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]
    
    if (prediction == "regression"):
        X, y = getDataReg(df, chosen_class, normalize, dummyCat, remove_FP, user_Features_Include, user_Features_Exclude, binary_class_threshold, keepImportantUnknown, visualize)
    if (prediction == "classification"): 
        X, y, warnings= getDataCls(df, chosen_class, normalize, dummyCat, remove_FP, user_Features_Include, user_Features_Exclude, binary_class_threshold, keepImportantUnknown)
        
        
    #data_dmatrix = xgb.DMatrix(data=X,label=y)
    
    
    if (train=="split"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        
        if tunning:
            xgb_model = tuneXGB(X, y, chosen_class, prediction, train)
        else:
            if (prediction == "regression"):
                xgb_model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
            if (prediction == "classification"):
                xgb_model = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

        xgb_model.fit(X_train,y_train)
        preds = xg_model.predict(X_test)    

        if (prediction == "regression"):
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            print("RMSE: %f" % (rmse))
        if (prediction == "classification"): 
            accuracy = accuracy_score(y_test, preds)
            prob_y= xg_model.predict_proba(X)
            prob_y = [p[1] for p in prob_y]
            aucroc = roc_auc_score(y, prob_y) 
            print("Accuracy: %f" % (accuracy))
            print("AUC: %f" % (aucroc))
    
    if (train=="cv"):
        if (prediction=="regression"):
            if tunning:
                tuneXGB(X, y, chosen_class, prediction, train)
            else:
                params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}
                cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50,early_stopping_rounds=10,metrics=['auc', 'rmse'], as_pandas=True, seed=123)            
                print((cv_results["test-rmse-mean"]).tail(1))
                print((cv_results["test-auc-mean"]).tail(1))

        if (prediction=="classification"):
            if tunning:
                model = tuneXGB(X, y, chosen_class, prediction, train)
            else:
                params = {"objective":"reg:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}
                cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50,early_stopping_rounds=10,metrics=['auc'], as_pandas=True, seed=123)
                print("AUC:", (cv_results["test-auc-mean"]).tail(1))
                
                model = xgb.XGBClassifier(objective="reg:logistic",colsample_bytree=0.3,learning_rate=0.1, max_depth=5, alpha=10, n_estimators = 10)
                kfold = StratifiedKFold(n_splits=10, random_state=123)
                results = cross_val_score(model, X, y, cv=kfold)
                print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
            
            return model
        
        
def visualizeXGB(xgb_model, num_trees):

    xgb.plot_tree(xgb_model,num_trees=num_trees)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.show()
    
    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()

    
def tuneXGB(X, y, chosen_class, prediction, train):
    if (train=="cv"):
        if (prediction=="regression"):
            xgb_model = xgb.XGBRegressor()

        if (prediction=="classification"):            
            xgb_model = xgb.XGBClassifier()
       
        if (chosen_class == 'Immune_evasion'):
            enc = LabelEncoder()
            y = enc.fit_transform(y)
            #y[chosen_class] = mlb.fit_transform(y[chosen_class])
            print(y) 

            xgb_model = OneVsOneClassifier(xgb.XGBClassifier(n_jobs=-1))    
      
            #kfold = KFold(n_splits=10, shuffle=True, random_state=123)

            #score=cross_val_score(clf, X, y, cv=kfold, n_jobs=-1, scoring='accuracy')
            
        tuned_xgb_model = randomTuneXGB(X, y, chosen_class, xgb_model)
        
        return tuned_xgb_model
        
        
def randomTuneXGB(X, y, chosen_class, xgb_model):            
    params = {
            "colsample_bytree": uniform(0.7, 0.3),
            "gamma": uniform(0, 0.5),
            "learning_rate": uniform(0.03, 0.3), # default 0.1 
            "max_depth": randint(2, 6), # default 3
            "n_estimators": randint(100, 150), # default 100
            "subsample": uniform(0.6, 0.4)
    }          
        
    #scoring to be saved, refit will choose final criteria
    if (chosen_class == 'multiclass'):
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(my_custom_auc_score_func, greater_is_better=True), 'Precision_std':make_scorer(my_custom_precstd_multiclass_score_func, greater_is_better=False), 'Recall_std':make_scorer(my_custom_recstd_multiclass_score_func, greater_is_better=False)}
    elif (chosen_class=="multilabel"):
        #scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(roc_auc_score, average='micro')}
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(average_precision_score, average='micro'), 'Precision_std':make_scorer(my_custom_precstd_multilabel_score_func, greater_is_better=False), 'Recall_std':make_scorer(my_custom_recstd_multilabel_score_func, greater_is_better=False)}   
    else:
        #being other class rather than multiclass
        scoring = {'Precision': 'precision', 'Recall':'recall','AUC': 'roc_auc', 'Accuracy': 'accuracy'}   

    search = RandomizedSearchCV(xgb_model, param_distributions=params, scoring=scoring, random_state=123, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True, refit='Accuracy')

    search.fit(X, y)

    return search 

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")            


def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

def my_custom_auc_score_func(y_test, y_pred):
    
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    roc = average_precision_score(y_test, y_pred, average="micro")
    
    return roc

def my_custom_precstd_multilabel_score_func(y_test, y_pred):
    
    prec = precision_score(y_test, y_pred, average=None)
    
    return prec.std()

def my_custom_precstd_multiclass_score_func(y_test, y_pred):
    
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    prec = precision_score(y_test, y_pred, average=None)
    
    return prec.std()

def my_custom_recstd_multilabel_score_func(y_test, y_pred):
    
    rec = recall_score(y_test, y_pred, average=None)
    
    return rec.std()

def my_custom_recstd_multiclass_score_func(y_test, y_pred):
    
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    rec = recall_score(y_test, y_pred, average=None)
    
    return rec.std()

def randomTune(X, y, chosen_class, classifier, normalize=False, dummyCat=True, boxplot=False, remove_FP = True, user_FP=None, user_Features_Include = None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False, visualize=False):        
                  
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start = 2, stop = 1002, num = 50)]
    n_estimators = [10,100,500]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(2, 102, num = 10)]
    max_depth = [10, 100,500]
    max_depth.append(None)    
    # Minimum number of samples required to split a node
    min_samples_split = [2, 20, 40]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 10, 20]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    if (classifier=="RF"):
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    elif (classifier== "DT"):
        # Create the random grid
        random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

    #scoring to be saved, refit will choose final criteria
    if (chosen_class == 'multiclass'):
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(my_custom_auc_score_func, greater_is_better=True), 'Precision_std':make_scorer(my_custom_precstd_multiclass_score_func, greater_is_better=False), 'Recall_std':make_scorer(my_custom_recstd_multiclass_score_func, greater_is_better=False)}
    elif (chosen_class=="multilabel"):
        #scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(roc_auc_score, average='micro')}
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(average_precision_score, average='micro'), 'Precision_std':make_scorer(my_custom_precstd_multilabel_score_func, greater_is_better=False), 'Recall_std':make_scorer(my_custom_recstd_multilabel_score_func, greater_is_better=False)}   
    else:
        #being other class rather than multiclass
        scoring = {'Precision': 'precision', 'Recall':'recall','AUC': 'roc_auc', 'Accuracy': 'accuracy'}   

    # Create a based model
    if (classifier ==  "RF"):
        cls = RandomForestClassifier(random_state=123)
    elif (classifier == "DT"):
        cls = DecisionTreeClassifier(random_state=123)
        
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    random = RandomizedSearchCV(estimator = cls, param_distributions = random_grid, n_iter = 100, scoring=scoring, cv = 3, verbose=2, refit='Accuracy', random_state=123, n_jobs = -1)
    # Fit the random search model
    random.fit(X, y)
    
    return random.best_params_




def gridSearch(X, y, chosen_class, classifier, n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth, bootstrap, normalize=False, dummyCat=True, boxplot=False, remove_FP = True, user_Features_Include=None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False, visualize=False):
#def gridSearch_RandomForest(dataset, chosen_class, categorical="None"):
    
    
    if (min_samples_leaf==1):
        min_samples_leaf+=1  
    
    if (min_samples_split==2):
        min_samples_split+=2
    
    if (n_estimators==1):
        n_estimators+=1   
   
        
    # Create the parameter grid based on the results of random search 
    if (classifier=="RF"):
        if (max_depth==None):
            param_grid = {
                'bootstrap': [bootstrap],
                'max_depth': [max_depth],
                'max_features': [max_features],
                'min_samples_leaf': [min_samples_leaf-1, min_samples_leaf, min_samples_leaf+1,min_samples_leaf+2],
                'min_samples_split': [min_samples_split-2, min_samples_split, min_samples_split+2],
                'n_estimators': [n_estimators-1, n_estimators, n_estimators+1, n_estimators+2]
            }
        else:
            param_grid = {
                'bootstrap': [bootstrap],
                'max_depth': [max_depth-8, max_depth-5,max_depth, max_depth+10],
                'max_features': [max_features],
                'min_samples_leaf': [min_samples_leaf-1, min_samples_leaf, min_samples_leaf+1,min_samples_leaf+2],
                'min_samples_split': [min_samples_split-2, min_samples_split, min_samples_split+2],
                'n_estimators': [n_estimators-1, n_estimators, n_estimators+1, n_estimators+2]
            } 
    elif (classifier== "DT"):
        if (max_depth==None):
            param_grid = {
                'max_depth': [max_depth],
                'max_features': [max_features],
                'min_samples_leaf': [min_samples_leaf-1, min_samples_leaf, min_samples_leaf+1,min_samples_leaf+2],
                'min_samples_split': [min_samples_split-2, min_samples_split, min_samples_split+2],
            }
        else:
            param_grid = {
                'max_depth': [max_depth-2, max_depth-1,max_depth, max_depth+1],
                'max_features': [max_features],
                'min_samples_leaf': [min_samples_leaf-1, min_samples_leaf, min_samples_leaf+1,min_samples_leaf+2],
                'min_samples_split': [min_samples_split-2, min_samples_split, min_samples_split+2],
            } 
        
    # Create a based model
    if (classifier ==  "RF"):
        cls = RandomForestClassifier(random_state=123)
    elif (classifier == "DT"):
        cls = DecisionTreeClassifier(random_state=123)
    
    #scoring to be saved, refit will choose final criteria
    if (chosen_class == 'multiclass'):
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(my_custom_auc_score_func, greater_is_better=True), 'Precision_std':make_scorer(my_custom_precstd_multiclass_score_func, greater_is_better=False), 'Recall_std':make_scorer(my_custom_recstd_multiclass_score_func, greater_is_better=False)}
    elif (chosen_class=="multilabel"):
        #scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(roc_auc_score, average='micro')}
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(average_precision_score, average='micro'), 'Precision_std':make_scorer(my_custom_precstd_multilabel_score_func, greater_is_better=False), 'Recall_std':make_scorer(my_custom_recstd_multilabel_score_func, greater_is_better=False)}   
    else:
        #being other class rather than multiclass
        scoring = {'Precision': 'precision', 'Recall':'recall','AUC': 'roc_auc', 'Accuracy': 'accuracy'}   

    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    # That estimator is made available at ``gs.best_estimator_`` along with
    # parameters like ``gs.best_score_``, ``gs.best_params_`` and
    # ``gs.best_index_``
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = cls, param_grid = param_grid, scoring=scoring, cv = 3, n_jobs = -1, verbose = 2, refit='Accuracy', return_train_score=True)

    # Fit the grid search to the data
    gs = grid_search.fit(X, y)

    return gs

def optimize_classifier(dataset, chosen_class, classifier="RF", normalize=False, dummyCat=True, boxplot=False, remove_FP = True, user_Features_Include = None, user_Features_Exclude=None, binary_class_threshold = None, keepImportantUnknown=False, visualize=False):
    
    if (classifier=="XGB"):
        model = train_XGBoost(dataset, chosen_class, prediction="classification", train="cv", tunning=True, normalize=normalize, dummyCat=dummyCat, boxplot=False, remove_FP = remove_FP, user_Features_Include=user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = binary_class_threshold, keepImportantUnknown=keepImportantUnknown)
        return model
    
    else:
        X, y, warnings = getDataCls(dataset, chosen_class, normalize, dummyCat, remove_FP, user_Features_Include, user_Features_Exclude, binary_class_threshold, keepImportantUnknown)

        random_best_params = randomTune(X, y, chosen_class, classifier=classifier, normalize=normalize, dummyCat=dummyCat, boxplot=boxplot, remove_FP = remove_FP, user_Features_Include=user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = binary_class_threshold, keepImportantUnknown=keepImportantUnknown, visualize=visualize)

        #random_best_params = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': True}

        if (classifier=="RF"):
            n_estimators = random_best_params['n_estimators']
            bootstrap = random_best_params['bootstrap']
        elif (classifier=="DT"):
            n_estimators=None
            bootstrap=None
        min_samples_split = random_best_params['min_samples_split']
        min_samples_leaf = random_best_params['min_samples_leaf']
        max_features = random_best_params['max_features']
        max_depth = random_best_params['max_depth']


        gs_results = gridSearch(X, y, chosen_class, classifier, n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth, bootstrap, normalize=normalize, dummyCat=dummyCat, boxplot=boxplot, remove_FP = remove_FP, user_Features_Include=user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = binary_class_threshold, keepImportantUnknown=keepImportantUnknown, visualize=visualize)

        if (chosen_class != 'multilabel'):
            cv = StratifiedKFold(10)
            scores, permutation_scores, pvalues = permutation_test_score(gs_results.best_estimator_, X, y, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=-1)
            print()
            print("Re-running model Score and p-value:")
            print(scores, pvalues)

        return gs_results
       
             
                     
def plot_correlation_heatmap(df):
    corr = df.corr()
    
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)


    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
        
