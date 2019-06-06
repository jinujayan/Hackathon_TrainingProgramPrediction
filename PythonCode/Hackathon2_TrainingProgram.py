from flask import Flask, render_template, request, jsonify
import pandas as pd
##from werkzeug import secure_filename
import numpy as np
#import seaborn as sns
#from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
import json
#import os.path
from pathlib import Path

app = Flask(__name__)

@app.route("/")
def index():
    return "Index!"
    
@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        print("Inside post...show json")
        print (request.is_json)
        content = request.get_json()
        print("Get json complete ....display json now..")
        #print(content['canvas_course_name'])
        model = Path("RFModel_Hackathon2.pickle")
        print(model)
        if model.is_file():
            print("Model available, use it for inference....")
            jout = inference(content)
            out = json.dumps(jout).replace('\\"',"\"").replace('"[', '[').replace(']"', ']')
            response = jsonify(jout)
            response.set_data(out)
            #print(f"jsonify complete {response.get_json}")
            response.headers.add('Access-Control-Allow-Origin', '*')
            
        else:
            print("Build model....")
            modelBuild(content)
        return response


def modelBuild(jsondata):
    print("Inside the build and predict...show content")
    
    skill_program_df = pd.read_excel("TrainingProgram.xlsx", sheet_name = 'Completion Data Clean', header=0)
    userinfo_app_demo_df = pd.read_excel("TrainingProgram.xlsx", sheet_name = 'App & Demo Data ', header=0)
    essay_df = pd.read_excel("TrainingProgram.xlsx", sheet_name = 'Essay Scores & Distance', header=0)
    
    skill_program_df.columns = [c.lower().replace(' ','_').replace('_/_', '-').replace('(', '').replace(')', '').replace('_-_', '_') for c in skill_program_df.columns.tolist() ]
    userinfo_app_demo_df.columns = [c.lower().replace(' ','_').replace('_/_', '-').replace('(', '').replace(')', '').replace('_-_', '_') for c in userinfo_app_demo_df.columns.tolist() ]
    essay_df.columns = [c.lower().replace(' ','_').replace('_/_', '-').replace('(', '').replace(')', '').replace('_-_', '_') for c in essay_df.columns.tolist() ]
    skill_program_df = skill_program_df.astype({'canvas_course_name':'category', 'unit':'category','record_type':'category','course_name':'category','section_name':'category'})

    merged_df = pd.merge(skill_program_df, userinfo_app_demo_df, how='left', left_on=['fake_applicant_id', 'course_name','section_name'], right_on=['fake_applicant_id', 'course_name','section_name'])

    cols_fillna = ['essay_1_score_1','essay_1_score_2','essay_1_score_3','essay_2_score_1','essay_2_score_2','essay_2_score_3','essay_3_score_1','essay_3_score_2','essay_3_score_3']
    essay_df[cols_fillna]=essay_df[cols_fillna].fillna(-1)
    essay_df['essay_percentage'] = essay_df.apply(evaluateEssayScore, axis=1)

    ###Total merge
    merged_total_df = pd.merge(merged_df, essay_df[['course','fake_email','essay_percentage']], how='left', left_on=['course_name','fake_email_address'], right_on=['course','fake_email'])
    merged_total_df['maxlearner_test_score_percent'] = merged_total_df['maxlearner_test_score']/merged_total_df['points_possible_on_learner_test']
    merged_total_df.rename(columns={'completed?_y':'completed', 'enrollment_status_y':'enrollment_status'}, inplace=True)

    merged_total_df = merged_total_df.dropna(subset=['maxlearner_test_score'])
    merged_total_df = merged_total_df.dropna(subset=['record_type'])
    merged_total_df = merged_total_df.dropna(subset=['education'])

    drop_cols = ['day_of_start_at','title','day_of_submitted_at','day_of_due_at','grade','day_of_course_start_date_x','date_difference','day_of_course_start_date_y','day_of_applied_to_course_date','fake_first_name-given_name_y','fake_last_name-surname_','zip_code','essay_1_score_1','essay_1_score_2','essay_1_score_3','essay_2_score_1','essay_2_score_2','essay_2_score_3','essay_3_score_1','essay_3_score_2','essay_3_score_3','max_essay_score','fake_email','average_essay_score','course']
    drop_cols2 = ['attendance','excused','submitted?','major','essay_percentage']
    drop_cols3 = ['canvas_user_id','course_id','fake_first_name-given_name_x','section_canvas_course_id']
    drop_cols4 = ['maxlearner_test_score','points_possible_on_learner_test']
    drop_cols5 = ['promise_zone?','completed?_x','enrollment_status_x']
    drop_cols6= ['course_name','section_name','fake_applicant_id','fake_email_address']

    merged_total_df = merged_total_df.drop(drop_cols, axis=1)
    merged_total_df = merged_total_df.drop(drop_cols2, axis=1)
    merged_total_df = merged_total_df.drop(drop_cols3, axis=1)
    merged_total_df = merged_total_df.drop(drop_cols4, axis=1)
    merged_total_df = merged_total_df.drop(drop_cols5, axis=1)

    #transform columns
    code = {'Interested in seeking an apprenticeship/job in tech':'Apprentice_job','Interested in pursuing further education':'education' ,'General interest in coding':'coding','Professional development for my current job':'development'}
    promise_zone_code = {'No':0, 'Promise Zone':1}
    include_code = {'Yes':1, 'No':0}

    merged_total_df['primary_interest_in_course'] = merged_total_df['primary_interest_in_course'].map(code)
    merged_total_df['promise_zone_indicator'] = merged_total_df['promise_zone_indicator'].map(promise_zone_code)
    merged_total_df['include?'] = merged_total_df['include?'].map(include_code)
    one_hot_cols = ['canvas_course_name','title_clean','record_type', 'unit', 'include?','enrollment_status','race','gender','income','education', 'city', 'state','primary_interest_in_course', 'hours_coded','how_many_hours_a_week_can_you_commit_to_class']
    #print(jsondata)
    merged_total_df_dummies = pd.get_dummies( merged_total_df, columns = one_hot_cols )

    Y = merged_total_df_dummies['completed'].values
    merged_total_df_dummies = merged_total_df_dummies.drop('completed', axis = 1)
    X = merged_total_df_dummies.values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    with open('RFModel_'+'.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 

def inference(jsondata):
    print("Inside the inference method....show data")
    print(jsondata)
    #json_str = json.dumps(jsondata)
    #print("Converted json string")
    #print(json_str)
    infer_df = pd.DataFrame(jsondata['students'], index=list(range(len(jsondata['students']))))
    #print(f"print full infer -> {infer_df}")
    print(infer_df.head())
    output_df = infer_df[['course_name','section_name','fake_applicant_id','fake_email_address']].copy()
    drop_cols6= ['course_name','section_name','fake_applicant_id','fake_email_address']
    infer_df = infer_df.drop(drop_cols6, axis=1)
    print(f"The loaded and cleaned data shape is -> {infer_df.shape}")
    ###Load the saved full dataset
    Interim_load_df = pd.read_csv("ProcessedTotalFile_Interim.csv", header=0)
    dataset_all = pd.concat(objs=[Interim_load_df, infer_df], axis=0,sort=False)

    one_hot_cols2 = ['canvas_course_name','title_clean','record_type', 'unit','enrollment_status','race','gender','income','education', 'city', 'state','primary_interest_in_course', 'hours_coded','how_many_hours_a_week_can_you_commit_to_class']
    dataset_all_dummies = pd.get_dummies( dataset_all, columns = one_hot_cols2 )
    print(f"The OHE shape is {dataset_all_dummies.shape}")
    #Y = merged_total_df_dummies['completed'].values
    #merged_total_df_dummies = merged_total_df_dummies.drop('completed', axis = 1)
    X = dataset_all_dummies.values

    savedModel = 'RFModel_Hackathon2.pickle'
    with open(savedModel, 'rb') as handle:
        rf_clf_loaded = pickle.load(handle)
        print("model loaded.....")
    y_pred=rf_clf_loaded.predict(X)
    output_df['Prediction'] = y_pred[-output_df.shape[0]:]

    jsn = output_df.to_json(orient='records')
    #jsnobj = json.loads(jsn)
    feature_imp = pd.DataFrame({'FeatureName' : dataset_all_dummies.columns,
                            'FeatureImportance' : rf_clf_loaded.feature_importances_}).sort_values(by = 'FeatureImportance',ascending=False)
    feature_json= feature_imp.iloc[:10].to_json(orient='records')
    full_out = {'students':jsn, 'features':feature_json}
    #jsnobj.append(dict_features)
    print("Show the dict obj...")
    print(full_out)
    #print(json.dumps(full_out).replace('\\"',"\"").replace('"[', '[').replace(']"', ']'))
    #return json.dumps(full_out).replace('\\"',"\"").replace('"[', '[').replace(']"', ']')
    return full_out

def evaluateEssayScore(row):
    score_list = [row.essay_1_score_1,row.essay_1_score_2,row.essay_1_score_3,row.essay_2_score_1,row.essay_2_score_2,row.essay_2_score_3,row.essay_3_score_1,row.essay_3_score_2,row.essay_3_score_3]
    percentage = (sum(x for x in score_list if x>=0)/(sum(1 for x in score_list if x>=0)*row.max_score + 0.001))*100
    return percentage
if __name__ == '__main__':
    app.run()
    #app.run(host='0.0.0.0', port=5009)
    #app.run(debug=True, port=5009)