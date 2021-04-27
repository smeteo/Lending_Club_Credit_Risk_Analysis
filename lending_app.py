import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import webbrowser

st.sidebar.title("Lending Club Loan Prediction")

im = Image.open("image.jpg")
st.image(im, width=500)


#xg_model = pickle.load(open("XGBoost","rb"))
xgb_model = pickle.load(open("XGBoost.pkl","rb"))
rf_model = pickle.load(open("RF.pkl","rb"))



sub_grades = ['A2','A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4' ,'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5']
home_owns = ['Own','Rent']
purposes = ['Credit Card','Debt Consolidaton','Educational','Home Improvement','House','Medical','Major Purcase','Moving','Vacation','Wedding','Other']


st.sidebar.header("Configure the Borrower Features:")
sub_grade = st.sidebar.selectbox("Grade of Borrower?",(sub_grades))
home_own = st.sidebar.selectbox("Borrower Home Ownership?",(home_owns))
purpose = st.sidebar.selectbox("Stated Purpose of Loan?",(purposes))

loan_amnt = st.sidebar.slider("What is loan amount?",1000,50000,10000,step=1)
interest_rate = st.sidebar.slider("Interest Rate?",0,25,12, step=0.1)
annual_income= st.sidebar.slider("Annual Income?",10000,250000,50000, step=10000)
issue_date= st.sidebar.slider("Loan Issue Date?",2012,2020,2015, step=1)
term = st.sidebar.slider("What is term 36/60 months?",36,60,36)
last_pay = st.sidebar.slider("What is Last Payment Amount?",100,15000,7000,step=100)

#subgrade_encode={  'A2':0,'A3':1, 'A4':2, 'A5':3, 'B1':4, 'B2':5, 'B3':6, 'B4':7, 'B5':8, 'C1':9, 'C2':10, 'C3':11, 'C4':12, 'C5':13, 'D1':14, 'D2':15, 'D3':16, 'D4':17,'D5':18, 'E1':19, 'E2':20, 'E3':21, 'E4':22, 'E5':23, 'F1':24, 'F2':25, 'F3':26, 'F4':27, 'F5':28, 'G1':29, 'G2':30, 'G3':31, 'G4':32, 'G5':33}

#home_encode = {'Own':0,
#                 'Rent':1}


#purpose_encode = {'Credit Card':0,'Debt Consolidaton':1,'Educational':2,'Home Improvement':3,'House':4,'Medical':5,'Major Purcase':6,'Moving':7,'Vacation':8,'Wedding':9,'Other':10}

my_dict = {
    'loan_amnt':loan_amnt,
 'term':term,
 'int_rate':interest_rate,
 'annual_inc':annual_income,
 'issue_d':issue_date,
 'dti':5,
 'inq_last_6mths':0,
 'open_acc':6,
 'revol_bal':5000,
 'revol_util':20,
 'last_pymnt_amnt':last_pay,
 'mort_acc':2,
 'pub_rec_bankruptcies'0,
 'earliest_cr_year':2000,
 'fico_range':600,
    'grade':'B',
 'sub_grade'sub_grade,
    'home_ownership':home_own,
    'verification_status':'Verified',
    'purpose':purpose,
    'initial_list_status':'w',
    'application_type':'Joint App'   
    
}

columns =['loan_amnt',
 'term',
 'int_rate',
 'annual_inc',
 'issue_d',
 'dti',
 'inq_last_6mths',
 'open_acc',
 'revol_bal',
 'revol_util',
 'last_pymnt_amnt',
 'mort_acc',
 'pub_rec_bankruptcies',
 'earliest_cr_year',
 'fico_range','grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G',
 'sub_grade_A2',
 'sub_grade_A3',
 'sub_grade_A4',
 'sub_grade_A5',
 'sub_grade_B1',
 'sub_grade_B2',
 'sub_grade_B3',
 'sub_grade_B4',
 'sub_grade_B5',
 'sub_grade_C1',
 'sub_grade_C2',
 'sub_grade_C3',
 'sub_grade_C4',
 'sub_grade_C5',
 'sub_grade_D1',
 'sub_grade_D2',
 'sub_grade_D3',
 'sub_grade_D4',
 'sub_grade_D5',
 'sub_grade_E1',
 'sub_grade_E2',
 'sub_grade_E3',
 'sub_grade_E4',
 'sub_grade_E5',
 'sub_grade_F1',
 'sub_grade_F2',
 'sub_grade_F3',
 'sub_grade_F4',
 'sub_grade_F5',
 'sub_grade_G1',
 'sub_grade_G2',
 'sub_grade_G3',
 'sub_grade_G4',
 'sub_grade_G5',
 'home_ownership_OWN',
 'home_ownership_RENT',
 'verification_status_Source Verified',
 'verification_status_Verified',
 'purpose_credit_card',
 'purpose_debt_consolidation',
 'purpose_educational',
 'purpose_home_improvement',
 'purpose_house',
 'purpose_major_purchase',
 'purpose_medical',
 'purpose_moving',
 'purpose_other',
 'purpose_renewable_energy',
 'purpose_small_business',
 'purpose_vacation',
 'purpose_wedding',
 'initial_list_status_w',
 'application_type_Joint App']

my_dict = {'satisfaction_level':satisfaction/100, 
           'last_evaluation':evaluation/100, 
           'number_project':project_count,
           'average_montly_hours':montly_hours, 
           'time_spend_company':spend_time, 
           'Work_accident':yes_no_encode[work_accident],
           'promotion_last_5years':yes_no_encode[promotion], 
           'Departments':department_encode[department], 
           'salary':salary_encode[salary],
            }

df = pd.DataFrame.from_dict([my_dict])
st.write('')
#st.dataframe(data=df, width=500, height=400)


st.subheader("1.Select features of an employee from left sidebar")

st.image('left.png', width=100)


st.write('')

st.subheader("2.Choose a Machine Learning Model:")
model = st.radio('',[
    #'XGBoost Classifier', 
                     'Random Forest Classifier',
                     'XGBOOST Classifier'
                    ])


# Button
if st.button("Predict"):
    import time
    with st.spinner("ML Model is loading..."):
        my_bar=st.progress(0)
        for p in range(0,101,10):
            my_bar.progress(p)
            time.sleep(0.1)
    
        if model=='Random Forest Classifier':
            churn_probability = rf_model.predict_proba(df)
            is_churn= rf_model.predict(df)
            
#        elif model=='XGBoost Classifier':
#            churn_probability= xg_model.predict_proba(df)
#            is_churn= xg_model.predict(df)
            
        elif model=='XGBOOST Classifier':
            churn_probability= xgb_model.predict_proba(df)
            is_churn= xgb_model.predict(df)
        
           
    
        st.success(f'The Probability of the Employee Churn (leave) is %{round(churn_probability[0][1]*100,1)}')
        
        if is_churn[0]:
            st.warning("The Employee will leave")
        else:
            st.success("The Employee will not leave")
            
st.subheader("Description")            
st.markdown("""

This is a Human Recources Analytics Project based on real data set. Dataset consist of 15.000 sample employee information such as the attributes on left side of the page.


Open source dataset can be found on Kaggle. 
""")





#if st.button('Click here to see datasource'):
    #webbrowser.open_new_tab('https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Employee+Churn+in+Python/HR_comma_sep.csv')
st.write("Click here to see real data source [Link](https://www.kaggle.com/c/employee-churn-prediction/data)")
st.write("Click here to download csv file [Link](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Employee+Churn+in+Python/HR_comma_sep.csv)")

    
    
#https://www.kaggle.com/c/employee-churn-prediction/data
    
st.markdown("""

The purpose of that project is to predict, based on the information submitted, whether the employee will going to leave (churn) the company or not in near future.  

Supervised Machine Learning, Random Forest and XGBOOST Classification algoritms used in that model. Change parameters on the left, you'll realize the effect on outcome.

""")


st.write(" ")

st.markdown("Prepared by: XXxxx")
#if st.button('LinkedIn'):
    #webbrowser.open_new_tab('https://www.linkedin.com/xxxx')


#if st.button('GitHub'):
    #webbrowser.open_new_tab('https://github.com/xxxxx')
    
#if st.button('Tableau'):
    #webbrowser.open_new_tab('https://public.tableau.com/profile/xxxx/')
    
#[['this is an image link']('linkedin.png')]('https://streamlit.io')
#['this is a text link']('https://streamlit.io')

st.write("Find me on [LinkedIn](https://www.linkedin.com/in/xxxxx/) / [GitHub](https://github.com/xxxx) / [Tableau](https://public.tableau.com/profile/xxxxx) ")
st.write("Find me on [LinkedIn](https://www.linkedin.com/in/xxxxx/) / [GitHub](https://github.com/xxxx) / [Tableau](https://public.tableau.com/profile/xxxxx) ")


#st.markdown('[![](left.jpg)](site.com)')
