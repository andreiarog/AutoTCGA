import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import PIL
import io
import re 
import xgboost as xgb
from sklearn.metrics import auc

from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, average_precision_score
import math
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
 
from sklearn.tree import export_graphviz
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score

import pickle
import os.path
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelBinarizer,LabelEncoder, MultiLabelBinarizer
from sklearn.multiclass import OneVsOneClassifier

#def check_input_parameters (chosen_class, preprocessing = 5, prediction="classification", predictor="RF", metric="accuracy", tunning=False, remove_FP=True, user_Features=None, binary_class_threshold = None, featureselection=False, featureEngineering=False, visualization=True, cls = None, cancerType = 'Overall'):
    
    
def previewApp(chosen_class, preprocessing = 5, prediction="classification", predictor="RF", metric="accuracy", remove_FP=True, user_Features_Include=None,user_Features_Exclude=None, binary_class_threshold = None, featureselection=False, featureEngineering=False, visualization=True, cls = None, cancerType = 'Overall', features_source='TCGA', inputDf=None):

    preprocessing = int(preprocessing)

    if (preprocessing != 5) or (prediction != "classification") or (remove_FP == False) or (metric!= "accuracy") or (user_Features_Include is not None) or (user_Features_Exclude is not None) or (features_source != 'TCGA'):
        
        if (user_Features_Include is None):
            user_Features_Include_None = "None"
        else:
            firsts = [w[0] for w in user_Features_Include]
            user_Features_Include_None = firsts
        
        if (user_Features_Exclude is None):
            user_Features_Exclude_None = "None"
        else:
            firsts = [w[0] for w in user_Features-Exclude]
            user_Features_Exclude_None = firsts
        

        dataset_name="{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(cancerType, chosen_class, featureselection, featureEngineering, preprocessing, metric, remove_FP, user_Features_Include_None, user_Features_Exclude_None, features_source)
        filename="gs_System_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(cancerType, chosen_class, predictor, featureselection, featureEngineering, preprocessing, prediction, metric, remove_FP, user_FeaturesInclude_None, user_Features_Exclude_None, features_source)

    else:
        dataset_name="{}_{}_{}_{}.csv".format(cancerType, chosen_class, featureselection, featureEngineering)
        filename="gs_System_{}_{}_{}_{}_{}".format(cancerType, chosen_class, predictor, featureselection, featureEngineering)
                
    if (preprocessing == 1):        
        dummyCat=False 
        catAge=False 
        normalize=False
    elif (preprocessing == 2):    
        dummyCat=True
        catAge=False
        normalize=False
    elif (preprocessing == 3):    
        dummyCat=False
        catAge=False
        normalize=True
    elif (preprocessing == 4):    
        dummyCat=True
        catAge=False
        normalize=True
    elif (preprocessing == 5):    
        dummyCat=True
        catAge=True
        normalize=False
    elif (preprocessing == 6):    
        dummyCat=True
        catAge=True
        normalize=True
    else:
        error = "ERROR: Parameter preprocessing mode must be between 1 and 6."
        return error
    
    # IF DATASET EXISTS
    if (os.path.isfile("csv/{}".format(dataset_name))):
        print("File ", dataset_name, " found. Loading.")
        dataset=pd.read_csv("csv/{}".format(dataset_name))
        keepImportantUnknown=featureEngineering

    else:
        if (features_source == 'TCGA'):
            cancer = pd.read_csv("cancer.csv")

            pancancer_features = ['sample_code', 'AS', 'LOH_n_seg', 'LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'Study Abbreviation', 'Study Name', 'CNV_segs', 'CNV_frac', 'HRD', 'purity', 'Virus_CMV', 'Virus_EBV', 'Virus_HBV', 'Virus_HCV', 'Virus_HHV', 'Virus_HIV', 'Virus_HPV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV', 'Virus_BK_Polyomavirus', 'Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'leucocyte_fraction', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_miRNA', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0', 'Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils', 'totTCR_reads', 'totTCRa_reads', 'totTCRb_reads', 'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'mDNAsi', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Treg_cluster']
            
            
            if (cancerType == 'Overall'):
                dataset = cancer[pancancer_features]
                dataset = f.process_clean(dataset)

            else:
                dataset = cancer.copy()
                dataset = f.process_clean(dataset)
                dataset = f.cancerTypeDataset(dataset, cancerType)
                print("IMPORTANT: Number of rows shortened to ", dataset.shape[0], " due to cancer type filter.")         
                dataset = f.process_clean_cancerType(dataset, cancerType)                
            
            #Add user input columns
            if (inputDf is not None):
                print("IMPORTANT: new columns added by user with shape ", inputDf.shape)
                print()
                #TO DO => check if input is ok (numeric, not null, if join is not empty...)
                dataset = pd.concat([dataset, inputDf], axis=1, join='inner')

                #return dataset
            
            #Filter by user_Features        
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
                    print("COLUMN IS THIS", col)
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
            
            #PRE-PROCESS => sample-code is droped
            dataset = f.pre_process(dataset, dummyCat=dummyCat, catAge=catAge, normalize=normalize)

            #FEATURE ENGINEERING   
            if (featureEngineering ):
                keepImportantUnknown=True
                dataset=f.create_sumFeatures(dataset) 
                dataset=f.process_logNormalize(dataset)
            else:
                keepImportantUnknown=False

            #FEATURE SELECTION                   
            if (featureselection):
                dataset = f.featureSelection(dataset, chosen_class, normalize, dummyCat, remove_FP, user_Features_Include, user_Features_Exclude, binary_class_threshold, keepImportantUnknown)
                
        else:
            keepImportantUnknown=False
            dataset, warnings_toAdd = get_dataset(chosen_class, dummyCat=dummyCat, normalize=normalize, user_Features_Include=user_Features_Include,user_Features_Exclude=user_Features_Exclude, binary_class_threshold = binary_class_threshold, cancerType=cancerType, features_source=features_source, inputDf=inputDf)
            #FEATURE SELECTION                   
            if (featureselection):
                dataset = f.featureSelection(dataset, chosen_class, normalize, dummyCat, binary_class_threshold, keepImportantUnknown)    
            
            print("DATASET TO BE USED: ", features_source, "with shape:", dataset.shape)
            print() 
        
        if (dataset.shape[0] == 0):
            error="ERROR: Dataset empty after pre-processing. "
            return error
        else:
            df=dataset.copy()
            df.to_csv("csv/{}".format(dataset_name), index=False)

    head = dataset.head()
    
    return head       

    
def createModelApp(chosen_class, preprocessing = 5, prediction="classification", predictor="RF", metric="accuracy", remove_FP=True, user_Features_Include = None, user_Features_Exclude=None, binary_class_threshold = None, featureselection=False, featureEngineering=False, visualization=True, cancerType = 'Overall', features_source='TCGA', inputDf=None):

    preprocessing = int(preprocessing)
    warnings = []
    
    for var in [featureEngineering, featureselection]:
        if (var == "True"):
            var=True
        if (var == "False"):
            var=False
    
    if (preprocessing != 5) or (prediction != "classification") or (remove_FP == False) or (metric!= "accuracy") or (user_Features_Include is not None) or (user_Features_Exclude is not None) or (features_source != 'TCGA'):
        
        if (user_Features_Include is None):
            user_Features_Include_None = "None"
        else:
            firsts = [w[0] for w in user_Features_Include]
            user_Features_Include_None = firsts

        if (user_Features_Exclude is None):
            user_Features_Include_None = "None"
        else:
            firsts = [w[0] for w in user_Features_Exclude]
            user_Features_Exclude_None=firsts
            
        dataset_name="{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(cancerType, chosen_class, featureselection, featureEngineering, preprocessing, metric, remove_FP, user_Features_None, features_source)
        filename="gs_System_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(cancerType, chosen_class, predictor, featureselection, featureEngineering, preprocessing, prediction, metric, remove_FP, user_Features_Include_None, user_Features_Exclude_None, features_source)

    else:
        dataset_name="{}_{}_{}_{}.csv".format(cancerType, chosen_class, featureselection, featureEngineering)
        filename="gs_System_{}_{}_{}_{}_{}".format(cancerType, chosen_class, predictor, featureselection, featureEngineering)
                
    if (preprocessing == 1):        
        dummyCat=False 
        catAge=False 
        normalize=False
    elif (preprocessing == 2):    
        dummyCat=True
        catAge=False
        normalize=False
    elif (preprocessing == 3):    
        dummyCat=False
        catAge=False
        normalize=True
    elif (preprocessing == 4):    
        dummyCat=True
        catAge=False
        normalize=True
    elif (preprocessing == 5):    
        dummyCat=True
        catAge=True
        normalize=False
    elif (preprocessing == 6):    
        dummyCat=True
        catAge=True
        normalize=True
    else:
        error = "ERROR: Parameter preprocessing mode must be between 1 and 6."
        return error
    
    # IF DATASET EXISTS
    if (os.path.isfile("csv/{}".format(dataset_name))):
        print("File ", dataset_name, " found. Loading.")
        dataset=pd.read_csv("csv/{}".format(dataset_name))
        keepImportantUnknown=featureEngineering

    else:
        if (features_source == 'TCGA'):
            cancer = pd.read_csv("cancer.csv")

            pancancer_features = ['sample_code', 'AS', 'LOH_n_seg', 'LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'Study Abbreviation', 'Study Name', 'CNV_segs', 'CNV_frac', 'HRD', 'purity', 'Virus_CMV', 'Virus_EBV', 'Virus_HBV', 'Virus_HCV', 'Virus_HHV', 'Virus_HIV', 'Virus_HPV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV', 'Virus_BK_Polyomavirus', 'Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'leucocyte_fraction', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_miRNA', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0', 'Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils', 'totTCR_reads', 'totTCRa_reads', 'totTCRb_reads', 'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'mDNAsi', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Treg_cluster']
            
            
            if (cancerType == 'Overall'):
                dataset = cancer[pancancer_features]
                dataset = f.process_clean(dataset)

            else:
                dataset = cancer.copy()
                dataset = f.process_clean(dataset)
                dataset = f.cancerTypeDataset(dataset, cancerType)
                warning="IMPORTANT: Number of rows shortened to {} due to cancer type filter.".format(dataset.shape[0])         
                warnings.append(warning)
                dataset = f.process_clean_cancerType(dataset, cancerType)                
            
            #Add user input columns
            if (inputDf is not None):
                warning = "IMPORTANT: new columns added by user with shape {}.".format(inputDf.shape)
                warnings.append(warning)
                #TO DO => check if input is ok (numeric, not null, if join is not empty...)
                dataset = pd.concat([dataset, inputDf], axis=1, join='inner')

                #return dataset
            
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
    
            
            #PRE-PROCESS => sample-code is droped
            dataset = f.pre_process(dataset, dummyCat=dummyCat, catAge=catAge, normalize=normalize)

            #FEATURE ENGINEERING   
            if (featureEngineering ):
                keepImportantUnknown=True
                dataset=f.create_sumFeatures(dataset) 
                dataset=f.process_logNormalize(dataset)
            else:
                keepImportantUnknown=False

            #FEATURE SELECTION                   
            if (featureselection):
                dataset = f.featureSelection(dataset, chosen_class, normalize, dummyCat, remove_FP, user_Features_Include, user_Features_Exclude, binary_class_threshold, keepImportantUnknown)
                
        else:
            keepImportantUnknown=False
            dataset, warnings_toAdd = get_dataset(chosen_class, dummyCat=dummyCat, normalize=normalize, user_Features_Include = user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = binary_class_threshold, cancerType=cancerType, features_source=features_source, inputDf=inputDf)
            for w in warnings_toAdd:
                warnings.append(w)

            #FEATURE SELECTION                   
            if (featureselection):
                dataset = f.featureSelection(dataset, chosen_class, normalize, dummyCat, binary_class_threshold, keepImportantUnknown)    
            
            warning="DATASET TO BE USED: {} with shape: {}".format(features_source, dataset.shape)
            warnings.append(warning) 
        
        if (dataset.shape[0] == 0):
            error="ERROR: Dataset empty after pre-processing. "
            return error, error, error
        else:
            df=dataset.copy()
            df.to_csv("csv/{}".format(dataset_name), index=False)
            
    #PREPARE DATASET FOR PREDICTION => always necessary to create X and y even with dataset already created
    X, y, warnings_Data = f.getDataCls(dataset, chosen_class, normalize=normalize, dummyCat=dummyCat, remove_FP = remove_FP, binary_class_threshold = binary_class_threshold, keepImportantUnknown = keepImportantUnknown)

    for warning in warnings_Data:
        warnings.append(warning)
        
    #IF MODEL ALREADY EXISTS
    if (os.path.isfile("gs/{}".format(filename))):
        print("File ", filename, " found. Loading.")
        print()
        with open("gs/{}".format(filename), 'rb') as gs_file:
            gs = pickle.load(gs_file)
        
        print()
        evaluation=[]
        evaluation.append("Model Created:")
        evaluation.append(gs.best_estimator_)
        evaluation.append("Accuracy:")
        evaluation.append(gs.best_score_)
        index = np.where(gs.cv_results_['mean_test_Accuracy']==gs.cv_results_['mean_test_Accuracy'].max())
        #if isinstance(chosen_class, str) and (chosen_class != 'Immune_evasion'):
        evaluation.append("AUC:")
        evaluation.append(gs.cv_results_['mean_test_AUC'][index][0])
        
        model = gs.best_estimator_
    
    else:
        #PREDICTION
        if (prediction == "regression"):
            model = fitRegressor(X, y, chosen_class, regressor=predictor, metric=metric, tunning = True, normalize=normalize, dummyCat=dummyCat, remove_FP=remove_FP, binary_class_threshold = binary_class_threshold, featureselection = featureselection, featureEngineering = featureEngineering, visualization=visualization, cancerType = cancerType)
        elif (prediction == "classification"):
            model, evaluation = fitClassifier(X, y, chosen_class, classifier=predictor, metric=metric, tunning = True, normalize=normalize, dummyCat=dummyCat, remove_FP=remove_FP, user_Features_Include=user_Features_Include,user_Features_Exclude=user_Features_Exclude , binary_class_threshold = binary_class_threshold, featureselection = featureselection, featureEngineering = featureEngineering, visualization=visualization, cancerType = cancerType, filename=filename)
        else:
            error = "ERROR: Parameter prediction mode must be either 'regression' or 'classification'."
            return error, error, error 

    #VISUALIZATION
    if (visualization):
        result = []
        
        if (predictor=="XGB"):
            f.visualizeXGB(model, len(model.estimators_))

        if (predictor == "RF"):
            estimators=model.estimators_
            i=0
            for tree in estimators:
                tree_text1 = "Tree no {}".format(i)
                n, d =f.tree_depth(tree)
                tree_text2 = "The tree structure has {} nodes with {} levels".format(n, d)
                
                if (chosen_class == 'multilabel'):                    
                    graph = Source(export_graphviz(tree, feature_names=X.columns.tolist(), class_names=[str(x) for x in  tree.classes_], out_file=None, filled = True))
                else:
                    graph = Source(export_graphviz(tree, feature_names=X.columns.tolist(), class_names=[str(x) for x in  tree.classes_.tolist()], out_file=None, filled = True))
                    
                #display(SVG(graph.pipe(format='svg')))
                
                graphPrint = graph.pipe(format='png')
                graphPrint = base64.b64encode(graphPrint).decode('utf-8')
                
                #Save
                #graph.format = 'png'
                #save_name = "Tree_{}_{}".format(i, filename)
                #graph.render(save_name, view=False)

                #Feature Importance List
                #a,b,c,d,boxplot, a and d are titles, b is text with non-null, c is list
                features_importance = printFeatureImportanceApp(X, y, tree, chosen_class)
                

                #RESULT
                result.extend(( ('text', tree_text1), ('text', tree_text2) , ('tree', graphPrint),('title',features_importance[0]), ('text', features_importance[1]) ))
                for feat in features_importance[2]:
                    result.append( ('text', feat) )

                if (chosen_class != 'multilabel'):                
                    result.append( ('title', features_importance[3]) )
                    for b in features_importance[4]:
                        result.append( ('boxplot', b) )                   
                i+=1

        else:
            estimator=model
            n, d=f.tree_depth(estimator)
            tree_text = "The tree structure has {} nodes with {} levels".format(n, d)
            if (chosen_class == 'multilabel'):                    
                graph = Source(export_graphviz(estimator, feature_names=X.columns.tolist(), class_names=[str(x) for x in  estimator.classes_] , out_file=None, filled = True))
            else:
                graph = Source(export_graphviz(estimator, feature_names=X.columns.tolist(), class_names=[str(x) for x in  estimator.classes_.tolist()] , out_file=None, filled = True))
            
            #graphData = Source(export_graphviz(estimator, feature_names=X.columns.tolist(), class_names=[str(x) for x in  estimator.classes_.tolist()] , out_file=data, filled = True))
            #graph_data = pydotplus.graph_from_dot_data(data.getvalue())
            #display(SVG(graph.pipe(format='svg')))

            graphPrint = graph.pipe(format='png')
            graphPrint = base64.b64encode(graphPrint).decode('utf-8')

            #Save
            #graph.format = 'png'
            #save_name = "Tree_{}".format(filename)
            #graph.render(save_name, view=False)
            
            #Feature Importance List
            #a,b,c,d,boxplot, a and d are titles, b is text with non-null, c is list
            features_importance = printFeatureImportanceApp(X, y, estimator, chosen_class)
            

            #RESULT
            result.extend(( ('text', tree_text) , ('tree', graphPrint),('title',features_importance[0]), ('text', features_importance[1]) ))
            for feat in features_importance[2]:
                result.append( ('text', feat) )

            if (chosen_class != 'multilabel'):                
                result.append( ('title', features_importance[3]) )
                for b in features_importance[4]:
                    result.append( ('boxplot', b) )                   
        
            
    return result, warnings, evaluation

        
    
def createModel(chosen_class, preprocessing = 5, prediction="classification", predictor="RF", metric="accuracy", tunning=False, remove_FP=True, user_Features_Include=None,user_Features_Exclude=None, binary_class_threshold = None, featureselection=False, featureEngineering=False, visualization=True, cls = None, cancerType = 'Overall', features_source='TCGA', inputDf=None):
# it needs to have X and y in order to call the fitClassifier and fitRegressor which are personalized here

#dataset: dataframe with features and class and all records
#pre-processing: number of pre-processing approach from 1 to 6                
#chosen_class: column name or abbreviation of exists
#prediction: classification or regression
#remove_FP: remove fake predictors based on correlation values
    
    #check_input_parameters(chosen_class, preprocessing = 5, prediction="classification", predictor="RF", metric="accuracy", tunning=False, remove_FP=True, binary_class_threshold = None, featureselection=False, featureEngineering=False, visualization=True, cls = None, cancerType = 'Overall')
    
    if (preprocessing != 5) or (prediction != "classification") or (remove_FP == False) or (metric!= "accuracy") or (user_Features_Include is not None) or (user_Features_Exclude is not None) or (features_source != 'TCGA'):
        
        if (user_Features_Include is None):
            user_Features_Include_None = "None"
        else:
            firsts = [w[0] for w in user_Features]
            user_Features_None = firsts
        
        if (user_Features_Exclude is None):
            user_Features_Exclude_None = "None"
        else:
            firsts = [w[0] for w in user_Features-Exclude]
            user_Features_Exclude_None = firsts
        

        dataset_name="{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(cancerType, chosen_class, featureselection, featureEngineering, preprocessing, metric, remove_FP, user_Features_Include_None, user_Features_Exclude_None, features_source)
        filename="gs_System_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(cancerType, chosen_class, predictor, featureselection, featureEngineering, preprocessing, prediction, metric, remove_FP, user_FeaturesInclude_None, user_Features_Exclude_None, features_source)

    else:
        dataset_name="{}_{}_{}_{}.csv".format(cancerType, chosen_class, featureselection, featureEngineering)
        filename="gs_System_{}_{}_{}_{}_{}".format(cancerType, chosen_class, predictor, featureselection, featureEngineering)
                
    if (preprocessing == 1):        
        dummyCat=False 
        catAge=False 
        normalize=False
    elif (preprocessing == 2):    
        dummyCat=True
        catAge=False
        normalize=False
    elif (preprocessing == 3):    
        dummyCat=False
        catAge=False
        normalize=True
    elif (preprocessing == 4):    
        dummyCat=True
        catAge=False
        normalize=True
    elif (preprocessing == 5):    
        dummyCat=True
        catAge=True
        normalize=False
    elif (preprocessing == 6):    
        dummyCat=True
        catAge=True
        normalize=True
    else:
        error = "ERROR: Parameter preprocessing mode must be between 1 and 6."
        return error
    
    # IF DATASET EXISTS
    if (os.path.isfile("csv/{}".format(dataset_name))):
        print("File ", dataset_name, " found. Loading.")
        dataset=pd.read_csv("csv/{}".format(dataset_name))
        keepImportantUnknown=featureEngineering

    else:
        if (features_source == 'TCGA'):
            cancer = pd.read_csv("cancer.csv")

            pancancer_features = ['sample_code', 'AS', 'LOH_n_seg', 'LOH_frac_altered', 'MutationsSilent', 'MutationsNonSilent', 'Study Abbreviation', 'Study Name', 'CNV_segs', 'CNV_frac', 'HRD', 'purity', 'Virus_CMV', 'Virus_EBV', 'Virus_HBV', 'Virus_HCV', 'Virus_HHV', 'Virus_HIV', 'Virus_HPV', 'Virus_HTLV', 'Virus_MCV', 'Virus_SV', 'Virus_BK_Polyomavirus', 'Virus_Saimiriine_Herpesvirus', 'numberOfNonSynonymousSNP', 'numberOfImmunogenicMutation', 'numberOfBindingExpressedPMHC', 'Indel_num', 'Immunogenic_indel_num', 'Neoantigen_num', 'leucocyte_fraction', 'Subtype_mRNA', 'Subtype_DNAmeth', 'Subtype_protein', 'Subtype_miRNA', 'Subtype_CNA', 'Subtype_Integrative', 'Subtype_other', 'Subtype_Selected', 'B.cells.naive', 'B.cells.memory', 'Plasma.cells', 'T.cells.CD8', 'T.cells.CD4.naive', 'T.cells.CD4.memory.resting', 'T.cells.CD4.memory.activated', 'T.cells.follicular.helper', 'T.cells.regulatory..Tregs.', 'T.cells.gamma.delta', 'NK.cells.resting', 'NK.cells.activated', 'Monocytes', 'Macrophages.M0', 'Macrophages.M1', 'Macrophages.M2', 'Dendritic.cells.resting', 'Dendritic.cells.activated', 'Mast.cells.resting', 'Mast.cells.activated', 'Eosinophils', 'Neutrophils', 'totTCR_reads', 'totTCRa_reads', 'totTCRb_reads', 'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'histological_type', 'histological_grade', 'vital_status', 'tumor_status', 'cause_of_death', 'treatment_outcome_first_course', 'OS', 'OS.time', 'mDNAsi', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Treg_cluster']
            
            
            if (cancerType == 'Overall'):
                dataset = cancer[pancancer_features]
                dataset = f.process_clean(dataset)

            else:
                dataset = cancer.copy()
                dataset = f.process_clean(dataset)
                dataset = f.cancerTypeDataset(dataset, cancerType)
                print("IMPORTANT: Number of rows shortened to ", dataset.shape[0], " due to cancer type filter.")         
                dataset = f.process_clean_cancerType(dataset, cancerType)                
            
            #Add user input columns
            if (inputDf is not None):
                print("IMPORTANT: new columns added by user with shape ", inputDf.shape)
                print()
                #TO DO => check if input is ok (numeric, not null, if join is not empty...)
                dataset = pd.concat([dataset, inputDf], axis=1, join='inner')

                #return dataset
            
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

            
            #PRE-PROCESS => sample-code is droped
            dataset = f.pre_process(dataset, dummyCat=dummyCat, catAge=catAge, normalize=normalize)

            #FEATURE ENGINEERING   
            if (featureEngineering ):
                keepImportantUnknown=True
                dataset=f.create_sumFeatures(dataset) 
                dataset=f.process_logNormalize(dataset)
            else:
                keepImportantUnknown=False

            #FEATURE SELECTION                   
            if (featureselection):
                dataset = f.featureSelection(dataset, chosen_class, normalize, dummyCat, remove_FP, user_Features_Include, user_Features_Exclude, binary_class_threshold, keepImportantUnknown)
                
        else:
            keepImportantUnknown=False
            dataset, warnings_toAdd = get_dataset(chosen_class, dummyCat=dummyCat, normalize=normalize, user_Features_Include=user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = binary_class_threshold, cancerType=cancerType, features_source=features_source, inputDf=inputDf)
            #FEATURE SELECTION                   
            if (featureselection):
                dataset = f.featureSelection(dataset, chosen_class, normalize, dummyCat, binary_class_threshold, keepImportantUnknown)    
            
            print("DATASET TO BE USED: ", features_source, "with shape:", dataset.shape)
            print() 
        
        if (dataset.shape[0] == 0):
            error="ERROR: Dataset empty after pre-processing. "
            return error
        else:
            df=dataset.copy()
            df.to_csv("csv/{}".format(dataset_name), index=False)
            
    #PREPARE DATASET FOR PREDICTION => always necessary to create X and y even with dataset already created           
    if (predictor=="XGB"):
        #Clean data for XGBoost because of error: feature_names may not contain [, ] or <
        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        dataset.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in dataset.columns.values]
    if (prediction=="classification"):
        X, y, warnings = f.getDataCls(dataset, chosen_class, normalize=normalize, dummyCat=dummyCat, remove_FP = remove_FP, user_Features_Include=user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = binary_class_threshold, keepImportantUnknown = keepImportantUnknown)
        print("INPUT GIVEN", X.shape, y.shape)

    if (prediction == "regression"):
        X, y = f.getDataReg(dataset, chosen_class, normalize, dummyCat, remove_FP, user_Features_Include, binary_class_threshold, keepImportantUnknown, visualize)
            
            
    #IF MODEL ALREADY EXISTS
    if (os.path.isfile("gs/{}".format(filename))):
        print("File ", filename, " found. Loading.")
        print()
        with open("gs/{}".format(filename), 'rb') as gs_file:
            gs = pickle.load(gs_file)
        
        print()
        print("MODELS CREATED:")
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

        model = gs.best_estimator_
    
    else:
        #PREDICTION
        if (prediction == "regression"):
            model = fitRegressor(X, y, chosen_class, regressor=predictor, metric=metric, tunning = tunning, normalize=normalize, dummyCat=dummyCat, remove_FP=remove_FP, binary_class_threshold = binary_class_threshold, featureselection = featureselection, featureEngineering = featureEngineering, visualization=visualization, cancerType = cancerType)
        elif (prediction == "classification"):
            model, evaluation = fitClassifier(X, y, chosen_class, classifier=predictor, metric=metric, tunning = tunning, normalize=normalize, dummyCat=dummyCat, remove_FP=remove_FP, user_Features_Include=user_Features_Include, user_Features_Exclude=user_Features_Exclude, binary_class_threshold = binary_class_threshold, featureselection = featureselection, featureEngineering = featureEngineering, visualization=visualization, cancerType = cancerType, filename=filename)
        else:
            error = "ERROR: Parameter prediction mode must be either 'regression' or 'classification'."
            return error

    #VISUALIZATION
    if (visualization):
        
        if (predictor=="XGB"):
            #f.visualizeXGB(model, len(model.estimators_))
            print("No visualization available")
        elif (predictor == "RF"):
            estimators=model.estimators_
            i=0
            for tree in estimators:
                print("Tree no", i)
                n, d =f.tree_depth(tree)
                print("The tree structure has ", n," nodes with ",d," levels.")
                
                if (chosen_class == 'multilabel'):                    
                    graph = Source(export_graphviz(tree, feature_names=X.columns.tolist(), class_names=[str(x) for x in  tree.classes_], out_file=None, filled = True))
                else:
                    graph = Source(export_graphviz(tree, feature_names=X.columns.tolist(), class_names=[str(x) for x in  tree.classes_.tolist()], out_file=None, filled = True))

                display(SVG(graph.pipe(format='svg')))
                #Save
                graph.format = 'png'
                save_name = "Tree_{}_{}".format(i, filename)
                graph.render(save_name, view=False)
                #Feature Importance List
                printFeatureImportance(X, y, tree, chosen_class)

                i+=1

        else:
            estimator=model
            n, d=f.tree_depth(estimator)
            print("The tree structure has ", n," nodes with ",d," levels.")
            if (chosen_class == 'multilabel'):                    
                graph = Source(export_graphviz(estimator, feature_names=X.columns.tolist(), class_names=[str(x) for x in  estimator.classes_] , out_file=None, filled = True))
            else:
                graph = Source(export_graphviz(estimator, feature_names=X.columns.tolist(), class_names=[str(x) for x in  estimator.classes_.tolist()] , out_file=None, filled = True))
            display(SVG(graph.pipe(format='svg')))
            #Save
            graph.format = 'png'
            save_name = "Tree_{}".format(filename)
            graph.render(save_name, view=False)
            #Feature Importance List
            printFeatureImportance(X, y, estimator, chosen_class)

            
def get_dataset(chosen_class, dummyCat=True, normalize=False, user_Features_Include = None, user_Features_Exclude=None, binary_class_threshold = None, cancerType='Overall', features_source=None, inputDf=None):
    
    warnings=[]

    if (features_source == 'msignaturedb'):
        dataset = pd.read_csv("Mutational.Sign.H1.csv",index_col=False)
        
    if (features_source == 'immunodulator'):
        dataset = pd.read_csv('Immunodulator.expression.15TCGA.csv',index_col=False)

    if (features_source == 'ntai'):
        dataset=pd.read_csv('NtAI.H1.csv',index_col=False)
        
    dataset = process_dataset(dataset, features_source = features_source)
    

    #Add user input columns
    if (inputDf is not None):
        #TO DO => check if input is ok (numeric, not null, if join is not empty...)
        dataset = pd.concat([dataset, inputDf], axis=1, join='inner')
        print("IMPORTANT: new columns added by user with shape ", inputDf.shape)
        print()
        warnings.append("IMPORTANT: new columns added by user with shape {}".format(inputDf.shape))
        #return dataset 

    #FILTER DATASET ROWS
    if (cancerType != 'Overall'):
        dataset = dataset[dataset['Study Abbreviation']==cancerType]
        print("IMPORTANT: Number of rows shortened to ", dataset.shape[0], " due to cancer type filter.")
        warnings.append("IMPORTANT: Number of rows shortened to {} due to cancer type filter.".format(dataset.shape[0]))

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
    

    #PRE-PROCESS => sample-code is droped
    toDrop = ['Study Abbreviation', 'sample_code', 'Sampletype']
    for col in toDrop:
        if col in dataset:
            dataset = dataset.drop(col, axis=1)
    
    
    return dataset, warnings

def process_dataset(dataset, features_source):
    
    df=dataset.copy()
    
    if (features_source == 'msignaturedb') or (features_source == 'immunomodulator'):
        for column in df.columns:
            if column not in ['Cohort', 'H1.Status', 'PatientID', 'Sampletype','SampleID']:
                df[column] = df[column].convert_objects(convert_numeric=True)
                df = f.clean_emptyString(df, column, mode='num')
                df = f.process_missing(df, column, 'median')
                
    if (features_source == 'msignaturedb'):
        df = df.rename({'Cohort': 'Study Abbreviation'}, axis=1)
        df['PatientID'] = df['PatientID'].apply(lambda x: x.replace(".", "_"))
        df = df.rename({'PatientID': 'sample_code'}, axis=1)

    if (features_source == 'immunodulator'):
        df = df.rename({'Cohort': 'Study Abbreviation'}, axis=1)
        df['SampleID'] = df['SampleID'].apply(lambda x: x[:12])
        df['SampleID'] = df['SampleID'].apply(lambda x: x.replace(".", "_"))
        df = df.rename({'SampleID': 'sample_code'}, axis=1)
    
    df = df.drop('H1.Status', axis=1)
            
    return df       
            
def fitClassifier(X, y, chosen_class, classifier="RF", metric="Accuracy", tunning = False, normalize=False, dummyCat=True, remove_FP=True, user_Features_Include = None, user_Features_Exclude=None, binary_class_threshold = None, featureselection=False, featureEngineering=False, visualization=True, cls = None, cancerType = 'Overall', filename=None):
    
    evaluation = []
    print("INPUT GIVEN", X.shape, y.shape)
    if (tunning):
        if (classifier=="XGB"):
            gs = train_XGBoost(X, y, chosen_class, prediction="classification", train="cv")            
        else:
            gs = optimizeClassifier(X, y, chosen_class, metric, classifier)
                
        with open("gs/{}".format(filename), 'wb') as gs_file:
            pickle.dump(gs, gs_file, pickle.HIGHEST_PROTOCOL)

        print()
        print("MODELS CREATED:")
        print(gs.best_estimator_)
        print("Accuracy:", gs.best_score_)
        index = np.where(gs.cv_results_['mean_test_Accuracy']==gs.cv_results_['mean_test_Accuracy'].max())
        #if isinstance(chosen_class, str) and (chosen_class != 'Immune_evasion'):
        print("AUC:", gs.cv_results_['mean_test_AUC'][index][0])
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
            
        evaluation.append("Model Created:")
        evaluation.append(gs.best_estimator_)
        evaluation.append("Accuracy:")
        evaluation.append(gs.best_score_)
        evaluation.append("AUC:")
        evaluation.append(gs.cv_results_['mean_test_AUC'][index][0])
        evaluation.append("Precision")
        evaluation.append(precision)
        evaluation.append("Recall")
        evaluation.append(recall)
        model = gs.best_estimator_
            
    else:
        if (classifier=="RF"):
            cls = RandomForestClassifier(n_estimators = 10, max_depth = 5, random_state=123)
        if (classifier=="DT"):
            cls = DecisionTreeClassifier( max_depth = 5, random_state=123) 
        
        #Create model
        model = cls.fit(X, y)
            
        print()
        if (classifier=="RF"):
            print("MODELS CREATED:")
            for estimator in model.estimators_:
                print(estimator)
                print()
        if (classifier=="DT"):
            print("MODEL CREATED:")
            print(model) 
        
        #scoring to be saved, refit will choose final criteria
        if (chosen_class == 'multiclass'):
            scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(f.my_custom_auc_score_func, greater_is_better=True), 'Precision_std':make_scorer(f.my_custom_precstd_multiclass_score_func, greater_is_better=False), 'Recall_std':make_scorer(f.my_custom_recstd_multiclass_score_func, greater_is_better=False)}
        elif (chosen_class=="multilabel"):
            #scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(roc_auc_score, average='micro')}
            scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(average_precision_score, average='micro'), 'Precision_std':make_scorer(f.my_custom_precstd_multilabel_score_func, greater_is_better=False), 'Recall_std':make_scorer(f.my_custom_recstd_multilabel_score_func, greater_is_better=False)}   
        else:
            #being other class rather than multiclass
            scoring = {'Precision': 'precision', 'Recall':'recall','AUC': 'roc_auc', 'Accuracy': 'accuracy'}   

                
        #Evaluate model
        
        if (chosen_class != 'multiclass'):
            scores = cross_validate(cls, X, y, scoring=scoring, cv=10, return_train_score=True)
            #To print
            acc_results = scores['test_acc']
            prec_results = scores['test_prec_macro']
            rec_results = scores['test_rec_micro']
            aucroc_results = scores['test_roc_auc']

            name=classifier

            acc = "%s, %s: %f (%f)" % (name, "Accuracy", acc_results.mean(), acc_results.std())
            print(acc)
            prec = "%s, %s: %f (%f)" % (name, "Precision", prec_results.mean(), prec_results.std())
            print(prec)
            rec = "%s, %s: %f (%f)" % (name, "Recall", rec_results.mean(), rec_results.std())
            print(rec)
            aucroc = "%s, %s: %f (%f)" % (name, "ROC-AUC", aucroc_results.mean(), aucroc_results.std())
            print(aucroc)
        else:
            scores = cross_validate(cls, X, y, cv=10, return_train_score=True)
            score = scores['test_score']
            acc = "%s: %f (%f)" % ("Accuracy", score.mean(), score.std())
            print(acc)
            
    if (chosen_class != 'multilabel'):
        printpvalue = pvalueModel(X, y, chosen_class, model, metric)
        print(printpvalue)
        evaluation.append(printpvalue)

    return model, evaluation

    
def fitRegressor(X, y, chosen_class, regressor="RF", metric="Accuracy", tunning = False, normalize=False, dummyCat=True, remove_FP=True, user_Features_Include=None, user_Features_Exclude=None, binary_class_threshold = None, featureselection=False, featureEngineering=False, visualization=True, cls = None, cancerType = 'Overall'):
        
    if (regressor=="RF"):
        reg = RandomForestClassifier(n_estimators = 10, max_depth = 5)
    if (regressor=="DT"):
        reg = DecisionTreeClassifier(max_depth = 5) 
    
    scores = cross_validate(reg, X, y, cv=10)

    #To print
    individual_results = scores['test_score']

    print("MSE of ", regressor ,": %0.2f (+/- %0.2f)" % (individual_results.mean(), individual_results.std() * 2 ))
    
    
def optimizeClassifier(X, y, chosen_class, metric="Accuracy", classifier="RF"):
      
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    n_estimators = [3,5]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', None]
    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth = [3,5]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 10, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 6, 10]
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
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(f.my_custom_auc_score_func, greater_is_better=True), 'Precision_std':make_scorer(f.my_custom_precstd_multiclass_score_func, greater_is_better=False), 'Recall_std':make_scorer(f.my_custom_recstd_multiclass_score_func, greater_is_better=False)}
    elif (chosen_class=="multilabel"):
        #scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(roc_auc_score, average='micro')}
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(average_precision_score, average='micro'), 'Precision_std':make_scorer(f.my_custom_precstd_multilabel_score_func, greater_is_better=False), 'Recall_std':make_scorer(f.my_custom_recstd_multilabel_score_func, greater_is_better=False)}   
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
    random = RandomizedSearchCV(estimator = cls, param_distributions = random_grid, n_iter = 100, scoring=scoring, cv = 3, verbose=2, refit="Accuracy", random_state=123, n_jobs = -1)
    # Fit the random search model
    random.fit(X, y)

    random_best_params = random.best_params_ 

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

    if (min_samples_leaf==1):
        min_samples_leaf+=1  

    if (min_samples_split==2):
        min_samples_split+=2

    if (n_estimators==5):
        n_estimators-=2    

    if (n_estimators==4):
        n_estimators-=1


    # Create the parameter grid based on the results of random search 
    if (classifier=="RF"):

        param_grid = {
            'bootstrap': [bootstrap],
            'max_depth': [max_depth-2, max_depth-1,max_depth, max_depth+1],
            'max_features': [max_features],
            'min_samples_leaf': [min_samples_leaf-1, min_samples_leaf, min_samples_leaf+1],
            'min_samples_split': [min_samples_split-2, min_samples_split, min_samples_split+2],
            'n_estimators': [n_estimators-1, n_estimators, n_estimators+1, n_estimators+2]
        } 
    elif (classifier == "DT"):

        param_grid = {
            'max_depth': [max_depth-2, max_depth-1,max_depth, max_depth+1],
            'max_features': [max_features],
            'min_samples_leaf': [min_samples_leaf-1, min_samples_leaf, min_samples_leaf+1],
            'min_samples_split': [min_samples_split-2, min_samples_split, min_samples_split+2],
        } 

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = cls, param_grid = param_grid, scoring=scoring, cv = 3, n_jobs = -1, verbose = 2, refit="Accuracy", return_train_score=True)

    # Fit the grid search to the data
    gs = grid_search.fit(X, y)

    return gs
       
def train_XGBoost(X, y, chosen_class, prediction="classification", train="cv"):
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
            "n_estimators": randint(1, 10), # default 100
            "subsample": uniform(0.6, 0.4)
    }          
        
    #scoring = {'Precision': 'precision', 'Recall':'recall','AUC': 'roc_auc', 'Accuracy': 'accuracy'}
    #scoring to be saved, refit will choose final criteria
    if (chosen_class == 'multiclass'):
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(f.my_custom_auc_score_func, greater_is_better=True), 'Precision_std':make_scorer(f.my_custom_precstd_multiclass_score_func, greater_is_better=False), 'Recall_std':make_scorer(f.my_custom_recstd_multiclass_score_func, greater_is_better=False)}
    elif (chosen_class=="multilabel"):
        #scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(roc_auc_score, average='micro')}
        scoring = {'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'Accuracy': 'accuracy', 'AUC': make_scorer(average_precision_score, average='micro'), 'Precision_std':make_scorer(f.my_custom_precstd_multilabel_score_func, greater_is_better=False), 'Recall_std':make_scorer(f.my_custom_recstd_multilabel_score_func, greater_is_better=False)}   
    else:
        #being other class rather than multiclass
        scoring = {'Precision': 'precision', 'Recall':'recall','AUC': 'roc_auc', 'Accuracy': 'accuracy'}   

    search = RandomizedSearchCV(xgb_model, param_distributions=params, scoring=scoring, random_state=123, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True, refit='Accuracy')
    
    print("INPUT GIVEN:", X.shape, y.shape)
    search.fit(X, y)

    return search 

def pvalueModel(X, y, chosen_class, model, metric):
    to_print = []

    cv = StratifiedKFold(10)
    scores, permutation_scores, pvalues = permutation_test_score(model, X, y, scoring=metric, cv=cv, n_permutations=100, n_jobs=-1)
    print()
    warning = "Re-running model Score and p-value:"
    print(warning)
    print(scores, pvalues)

    to_print.append(warning)
    to_print.append(scores)
    to_print.append(pvalues)

    return to_print    


             
def printFeatureImportance(X, y, model, chosen_class):
    print("Features Importance")
    feature_importances = pd.DataFrame(model.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)
    #print(feature_importances)
    df_filtered = feature_importances[feature_importances['importance'] != 0]
    print("Non-zero importance features listed: ",df_filtered.shape[0])
    print(df_filtered)
    
    if (chosen_class!='multilabel'):
        print("Distribution of most important features")
        dataset=pd.concat([X, y], axis=1)
        feat = df_filtered.reset_index()
        df=dataset.copy()
        j=0
        for i in feat['index'].unique():
            if j<5:
                dfTemp = df[[i, y.name]]
                boxplot = dfTemp.boxplot(by=y.name)
                plt.suptitle("")
            j+=1

def printFeatureImportanceApp(X, y, model, chosen_class):
    result=[]
    a = "Features Importance"
    feature_importances = pd.DataFrame(model.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)
    #print(feature_importances)
    df_filtered = feature_importances[feature_importances['importance'] != 0]
    b = "Non-zero importance features listed:{} ".format(df_filtered.shape[0])
    df_print = df_filtered.reset_index()
    df_print = [tuple(x) for x in df_print.values]
    c = df_print
    
    result =[a,b,c]    
    if (chosen_class!='multilabel'):
        boxplots=[]

        d = "Distribution of most important features"
        result.append(d)
        
        dataset=pd.concat([X, y], axis=1)
        feat = df_filtered.reset_index()
        df=dataset.copy()
        j=0
        for i in feat['index'].unique():
            if j<5:
                dfTemp = df[[i, y.name]]
                boxplot = dfTemp.boxplot(by=y.name)
                plt.suptitle("")
                fname="boxplot_{}.png".format(i)
                plt.savefig(fname)
                
                # load the image
                image = PIL.Image.open(fname)

                imgByteArr = io.BytesIO()
                image.save(imgByteArr, format='PNG')
                imgByteArr = imgByteArr.getvalue()
                image = base64.b64encode(imgByteArr).decode('utf-8')
                boxplots.append(image)
                os.remove(fname)
            j+=1
            

        result.append(boxplots)
            
    return result











