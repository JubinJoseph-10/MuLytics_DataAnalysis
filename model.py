#importing libs
#importing libs
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


#setting page configs
logo = Image.open("Data/page_favicon.png")
st.set_page_config(page_icon=logo,layout='wide',page_title='Bandwidth to Buyers')
st.set_option('deprecation.showPyplotGlobalUse', False)



#reading data\
@st.cache_data
def data_fetch(loc):
    vol = pd.read_csv(loc)
    return vol

data = data_fetch("Data/Mulytics_Dataset.csv")

####################################################Data Processing####################################################
################# Removing Outliers #################
#we will indentify oulier using an IQR based approach
def outlier_detector(column_name,dataframe,k_deviant_factor):
    iqr = dataframe[column_name].quantile(.75) - dataframe[column_name].quantile(.25)
    upper_bound = dataframe[column_name].quantile(.75) + k_deviant_factor * iqr
    lower_bound = dataframe[column_name].quantile(.25) - k_deviant_factor * iqr
    dataframe[f'{column_name} outlier'] = dataframe[column_name].apply(lambda x : 1 if (x>upper_bound or x<lower_bound) else 0)
#removing outliers
outlier_detector('Marketing Spend (INR)',data,1.5)
outlier_detector('Discount Availed (INR)',data,1.5)

#removing the outliers from both marketing expense and the discount availed
data = data[(data['Marketing Spend (INR) outlier']!=1)&(data['Discount Availed (INR) outlier']!=1)]
#imputing the dates to accept, qualify and install
data['Days to Accept'].fillna(100,inplace=True)
data['Days to Qualify'].fillna(100,inplace=True)
data['Days to Install Request'].fillna(100,inplace=True)
data['Accepted'] = data['Days to Accept'].apply(lambda x : 0 if x==100 else 1)
data['Qualified'] = data['Days to Qualify'].apply(lambda x : 0 if x==100 else 1)
reserve_data = data.copy()

##########################################Firsty filters and slicing##########################################

# Filter_1 = st.sidebar.selectbox("Select a Categorial Variable to filter out your data",['City', 'Plan Type', 'Household Income Level', 'Lead Source',
# 'Marketing Channel','Competitor Interest', 'Preferred Contact Time',
# 'Customer Tech-Savviness', 'Decision Influence', 'Complaint History','Payment Mode Preferred', 
# 'Payment Frequency', 'Bundled Service Interest', 'Competitor Price Sensitivity','Infrastructure Ready',
# 'Preferred Communication Mode','Is Holiday', 'Festive Period', 'Signal Strength',
#                    'Installed','Day of the Week','Follow-Up Count','Plan Cost (INR)', 'Distance to Service Hub',
#                            'Network Downtime (Hours)','Service Quality Rating','Time Spent on Research (Days)'])

# Filter_2 = st.sidebar.selectbox("Select a Categorial Variable to filter out your data:",['City', 'Plan Type', 'Household Income Level', 'Lead Source',
# 'Marketing Channel','Competitor Interest', 'Preferred Contact Time',
# 'Customer Tech-Savviness', 'Decision Influence', 'Complaint History','Payment Mode Preferred', 
# 'Payment Frequency', 'Bundled Service Interest', 'Competitor Price Sensitivity','Infrastructure Ready',
# 'Preferred Communication Mode','Is Holiday', 'Festive Period', 'Signal Strength',
#                    'Installed','Day of the Week'])


# Filter_1_ = st.sidebar.selectbox(data[Filter_1].value_counts().index.tolist(),label=f"Filter {Filter_1} by:")

# Filter_2 = st.sidebar.selectbox(data[Filter_2].value_counts().index.tolist(),label="Select a Numerical Variable to filter out your data")




###################### SECTION ON DISTRIBUTION OF CATEGORICAL VARIABLES ######################
#Testing the normality of the columns
univariate_analysis = st.container(border=True)

categorical_variables_dist = univariate_analysis.container(border=True)
categorical_variables_dist.markdown('<div style="text-align: center; font-size: 18px">Exploring the distribution of Leads over the categorical variables</div>',unsafe_allow_html=True)
categorical_variables_dist.write('\n')
plot_space_cat,des_cat = categorical_variables_dist.columns([.6,.4])

des_ = des_cat.container(border=True)
plot_space_cat_ = plot_space_cat.container(border=True)
des_.write('\n')

#description
des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:14px;'>Understanding the distribution of leads helps us analyze trends, preferences, and potential market focus areas. Select a variable from the dropdown to explore how different categorical factors influence our data.</div>""",unsafe_allow_html=True)
des_.write('\n')
des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:14px;'>Feel free to explore and play around and derive insights of your own, Some of our observations are given below:
<ul>
  <li style='font-size: 14px;'>Based on household income level <b>50%</b> of our leads are from the medium household income and <b>30%</b> and <b>20%</b> from low and high</li>
  <li style='font-size: 14px;'>On Holidays and Fetstive Periods we generate <b>1%</b> and <b>25%</b> of leads respectively</li>  
  <li style='font-size: 14px;'>Our installation rate across all the leads is <b>18%</b></li>
  <li style='font-size: 14px;'>Apart from these variables we have equal distribution of leads</li>
</ul></div>""",unsafe_allow_html=True)
des_.write('\n')
select_var_cat = des_.selectbox("Select a variable:", ['City', 'Plan Type', 'Household Income Level', 'Lead Source',
'Marketing Channel','Competitor Interest', 'Preferred Contact Time',
'Customer Tech-Savviness', 'Decision Influence', 'Complaint History','Payment Mode Preferred', 
'Payment Frequency', 'Bundled Service Interest', 'Competitor Price Sensitivity','Infrastructure Ready',
'Preferred Communication Mode','Is Holiday', 'Festive Period', 'Signal Strength','Plan Cost (INR)',
                   'Installed'],key='2')


#pie chart to show the distribution
# pie_univariate = px.pie(values=reserve_data[select_var_cat].value_counts().values,names=reserve_data[select_var_cat].value_counts().index.tolist(),color=reserve_data[select_var_cat].value_counts().index.to_list(),title=f'Distribution of Leads across {select_var_cat}',height=450)
# Customize hover text


import plotly.graph_objects as go

pie_univariate = go.Figure(go.Pie(
    name = "",
    values = reserve_data[select_var_cat].value_counts(),
    labels = reserve_data[select_var_cat].value_counts().index.tolist(),
    text = reserve_data[select_var_cat].value_counts().index.tolist(),
    hovertemplate = "%{label}: <br>Proportion of Leads: %{percent} in</br> %{text}"
))

# fig.show()



plot_space_cat_.plotly_chart(pie_univariate,use_container_width=True)

###################### SECTION ON DISTRIBUTION OF CONTINUOS VARIABLES ######################
#Testing the normality of the columns
continuos_variables_dist = univariate_analysis.container(border=True)
continuos_variables_dist.markdown('<div style="text-align: center; font-size: 16px">Exploring the distribution of the continuos variables</div>',unsafe_allow_html=True)
continuos_variables_dist.write('\n')
des_,plot_space = continuos_variables_dist.columns([.2,.8])

des = des_.container(border=True)
des.write('\n')
select_var = des.selectbox("Select a variable:", ['Discount Availed (INR)','Marketing Spend (INR)','Average Monthly Spend (INR)', 'Lifetime Value (INR)'],key='1')


des.write('\n')
description = f'The following graphs help us understand the normality and distribution of the {select_var} visually. Using the box plot and density plots together, gives us an overview of the outliers in the variable, its skewness and the kurtosis.'

des.markdown("""<div style="text-align: justify; font-size: 14px">{}'
             <ul>
  <li style='font-size: 14px;'>Only two variables from the ones here we could see here had outliers. Which are the <b>Lifetime Value in INR</b> and <b>The Discount availed in INR</b>. To identify we used the <b>upper and lower bounds as Q3 + 1.5 *(IQR) and Q1 - 1.5 *(IQR) respectively.</b></li>
  <li style='font-size: 14px;'>Rest of the variables did not have significant number of outliers to be treated seperatly.</li>
</ul></div>""".format(description),unsafe_allow_html=True)
des.write('\n')

plots = plot_space.container(border=True)
plots_den,plot_vio,plot_box = plots.columns(3)
density_plot =  px.histogram(reserve_data, y=select_var, title=f"{select_var} Density Plot",histnorm='probability density')
violin = px.violin(reserve_data,y=select_var,title=f'ViolinPlot_{select_var}') 
box = px.box(reserve_data,y=select_var,title=f'Boxplot_{select_var}')
plots_den.plotly_chart(density_plot,use_container_width=True)
plot_vio.plotly_chart(violin,use_container_width=True)
plot_box.plotly_chart(box,use_container_width=True)



###################### SECTION ON DISTRIBUTION OF ORDINAL VARIABLES ######################
#Testing the normality of the columns
ordinal_variables_dist = univariate_analysis.container(border=True)
ordinal_variables_dist.markdown('<div style="text-align: center; font-size: 16px">Exploring the distribution of the ordnial variables</div>',unsafe_allow_html=True)
ordinal_variables_dist.write('\n')
plot_space_nom,des_nom = ordinal_variables_dist.columns([.6,.4])



des_nom_ = des_nom.container(border=True)
des_nom_.write('\n')
description = f"By analyzing the frequency of each category, we can observe patterns in the data while maintaining the natural order of the variable. This visualization provides insights into the spread, central tendency, and any imbalances within the ordinal categories, aiding in a deeper understanding of the variable."
des_nom_.write('\n')
des_nom_.markdown("""<div style="text-align: justify; font-size: 14px">{}<ul>
  <li style='font-size: 14px;'>Null values were present in <b> days to install, days to accept and days to install request are imputed with an arbitrary value of 100</b> to showcase funnel breakage (Mean, Median/Mode Imputation would not have made sense since these are the binary cases).</li>
  <li style='font-size: 14px;'><b>Majority of the leads</b> experience <b>down time between 0 to 12 Hours.</b></li>
  <li style='font-size: 14px;'><b>Majority of the service hubs</b> are <b>located within the range 5 to 30 Kms from the leads.</b></b></li>
  <li style='font-size: 14px;'><b>Majority of the leads</b> are <b>followed up with, up to 5 times.</b></li>
</ul></div>""".format(description), unsafe_allow_html=True)
des_nom_.write('\n')
select_var_or = des_nom_.selectbox("Select a variable:", ['Follow-Up Count', 'Distance to Service Hub',
                           'Network Downtime (Hours)','Service Quality Rating','Time Spent on Research (Days)','Days to Accept'
                           ,'Days to Qualify','Days to Install Request'],key='3')


plot_space_nom_ = plot_space_nom.container(border=True)
bar_chart_ord = px.bar(pd.DataFrame(reserve_data[select_var_or].value_counts()),y='count',
                       color=pd.DataFrame(reserve_data[select_var_or].value_counts()).index)
plot_space_nom_.plotly_chart(bar_chart_ord,use_container_width=True)

###################### SECTION ON DISTRIBUTION OF LEAD GENRATION DATES ######################
#Testing the normality of the columns
lead_generation_dates = univariate_analysis.container(border=True)
lead_generation_dates.markdown('<div style="text-align: center; font-size: 16px">Exploring the distribution of the leads over time</div>',unsafe_allow_html=True)
lead_generation_dates.write('\n')
lead_generation_dates_ = lead_generation_dates.container(border=True)

#lead created date 
reserve_data['Lead Created Date'] = pd.to_datetime(reserve_data['Lead Created Date'])
reserve_data['Month-Year'] = reserve_data['Lead Created Date'].dt.to_period('M')
line_plot = px.line(reserve_data.groupby('Lead Created Date')['Lead ID'].count())
lead_generation_dates_.plotly_chart(line_plot,use_container_width=True)





########################### Section for Bivariate Analysis ########################### 
bivariate = st.container(border=True)
bivariate.markdown("""
                    <div style='text-align: center;color:black;font-size:16px;'></b>Bivariate Analysis</b></div>""",unsafe_allow_html=True)
bivariate.write('\n')
#creating columns for chart and description
bivariate_des, bivariate_charts = bivariate.columns([.40,.60])
bivariate_des_ = bivariate_des.container(border=True)
bivariate_charts_ = bivariate_charts.container(border=True)

#description
bivariate_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Now we dive deeper into how variables are interacting with each other. This helps in understanding how variables/entities stack up against each other given a certain KPI of your choice. You are once again free to play around with the charts below and compare and see for yourself how data unfolds.</div>""",unsafe_allow_html=True)
bivariate_des_.write('\n')
# bivariate_des_.markdown("""
#                     <div style='text-align: justify;color:black;font-size:14px;'>Feel free to explore and play around and derive insights of your own, Some of our observations are given below:
# <ul>
#   <li style='font-size: 14px;'>The average number of downloads for paid vs free apps show us a significant difference with downloads for <b>free apps being at 14.2 Millions as opposed to 259.3K Downloads for paid apps </b></li>
#   <li style='font-size: 14px;'>In terms of reviews for both application versions and android versions supprted, <b> dynamic verions perform better on all KPIs.</b></li>
# </ul></div>""",unsafe_allow_html=True)
bivariate_des_.write('\n')
#what vairables can be considered to be demographic variables in the above dataset
con_side_var_interest = bivariate_des_.selectbox('Select a Consumer Variable to Explore!',['City','Household Income Level','Customer Tech-Savviness','Decision Influence',
       'Competitor Interest','Preferred Contact Time','Complaint History','Payment Mode Preferred','Payment Frequency','Competitor Price Sensitivity','Infrastructure Ready'
      ,'Preferred Communication Mode'],key=4)

kpi_interest_con = bivariate_des_.selectbox('Select a Metric to Study!',['Follow-Up Count', 'Discount Availed (INR)','Marketing Spend (INR)','Plan Cost (INR)',
'Average Monthly Spend (INR)','Lifetime Value (INR)','Distance to Service Hub','Network Downtime (Hours)',
'Time Spent on Research (Days)','Service Quality Rating','Festive Period','Days to Accept','Days to Qualify','Days to Install Request'],key=6)

method_interest = bivariate_des_.selectbox('Select an Aggregation Method!',['Sum','Average','Frequency'],key=7)


#creating a definition to aggregate data
def aggregator(data,method_interest,kpi_interest,var_interest):
    if method_interest == 'Sum':
        agg = data.groupby([var_interest])[kpi_interest].sum()
    elif method_interest == 'Average':
        agg = data.groupby([var_interest])[kpi_interest].mean().round(1)
    elif method_interest == 'Frequency':
        agg = data.groupby([var_interest]).count()
    return agg

#creating a dataframe for aggregation of kpis over variables
if kpi_interest_con in ['Days to Accept','Days to Qualify','Days to Install Request']:
    data_interest = reserve_data[reserve_data[kpi_interest_con]!=100]
else:
    data_interest = reserve_data.copy()

agg = aggregator(data_interest,method_interest,kpi_interest_con,con_side_var_interest)
agg = pd.DataFrame(agg.reset_index())
#plotting the bar chart for comparison

# bi_chart = px.line_polar(pd.DataFrame(agg), r=kpi_interest_con, theta=con_side_var_interest, line_close=True)
# bi_chart.update_traces(fill='toself')
# # fig.show()


bi_chart = px.bar(pd.DataFrame(agg),color=con_side_var_interest,x=con_side_var_interest,
                        y=kpi_interest_con,title=f'{method_interest} {kpi_interest_con} over {con_side_var_interest}')
bivariate_charts_.plotly_chart(bi_chart)
bivariate_des_.write('\n')



#creating columns for chart and description
bivariate_charts_p,bivariate_des_p = bivariate.columns([.60,.40])
bivariate_des_p_ = bivariate_des_p.container(border=True)
bivariate_charts_p_ = bivariate_charts_p.container(border=True)

#description
bivariate_des_p_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Now we dive deeper into how variables are interacting with each other. This helps in understanding how variables/entities stack up against each other given a certain KPI of your choice. You are once again free to play around with the charts below and compare and see for yourself how data unfolds.</div>""",unsafe_allow_html=True)
bivariate_des_p_.write('\n')
# bivariate_des_.markdown("""
#                     <div style='text-align: justify;color:black;font-size:14px;'>Feel free to explore and play around and derive insights of your own, Some of our observations are given below:
# <ul>
#   <li style='font-size: 14px;'>The average number of downloads for paid vs free apps show us a significant difference with downloads for <b>free apps being at 14.2 Millions as opposed to 259.3K Downloads for paid apps </b></li>
#   <li style='font-size: 14px;'>In terms of reviews for both application versions and android versions supprted, <b> dynamic verions perform better on all KPIs.</b></li>
# </ul></div>""",unsafe_allow_html=True)

#what vairables can be considered to be demographic variables in the above dataset
prod_var_interest = bivariate_des_p_.selectbox('Select a Producer Variable to Explore!',['Lead Source','Marketing Channel', 'Plan Type' , 'Signal Strength'],key=10)
kpi_interest_prod = bivariate_des_p_.selectbox('Select a Metric to Study!',['Follow-Up Count', 'Discount Availed (INR)','Marketing Spend (INR)','Plan Cost (INR)',
'Average Monthly Spend (INR)','Lifetime Value (INR)','Distance to Service Hub','Network Downtime (Hours)',
'Time Spent on Research (Days)','Service Quality Rating','Festive Period','Days to Accept','Days to Qualify','Days to Install Request'],key=11)
method_interest_ = bivariate_des_p_.selectbox('Select an Aggregation Method!',['Sum','Average','Frequency'],key=69)

#creating a definition to aggregate data
def aggregator(data,method_interest,kpi_interest_prod,var_interest):
    if method_interest == 'Sum':
        agg = data.groupby([var_interest])[kpi_interest_prod].sum()
    elif method_interest == 'Average':
        agg = data.groupby([var_interest])[kpi_interest_prod].mean().round(1)
    elif method_interest == 'Frequency':
        agg = data.groupby([var_interest]).count()
    return agg


#creating a dataframe for aggregation of kpis over variables
if kpi_interest_prod in ['Days to Accept','Days to Qualify','Days to Install Request']:
    data_interest_prod = reserve_data[reserve_data[kpi_interest_prod]!=100]
else:
    data_interest_prod = reserve_data.copy()

#creating a dataframe for aggregation of kpis over variables
agg_ = aggregator(data_interest_prod,method_interest_,kpi_interest_prod,prod_var_interest)
agg_ = pd.DataFrame(agg_.reset_index())
#plotting the bar chart for comparison
bi_chart_ = px.bar(pd.DataFrame(agg_),color=prod_var_interest,x=prod_var_interest,
                        y=kpi_interest_prod,title=f'{method_interest} {kpi_interest_prod} over {prod_var_interest}')
bivariate_charts_p.plotly_chart(bi_chart_)

##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
########################### Mutli Variate ###########################
multivariate = st.container(border=True)
multivariate.markdown("""
                    <div style='text-align: center;color:black;font-size:16px;'></b>Multivariate Analysis</b></div>""",unsafe_allow_html=True)
multivariate.write('\n')

#creating columns for chart and description
multivariate_des, multivariate_charts = multivariate.columns([.40,.60])
multivariate_des_ = multivariate_des.container(border=True)
multivariate_charts = multivariate_charts.container(border=True)


#description
multivariate_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Now we explore how multiple variables interact and influence each other across various metrics. This section provides insights into how different data points aggregate and compare based on selected KPIs. By examining the visualizations below, you can investigate relationships between metrics, uncover patterns, and assess performance from different perspectives.</div>""",unsafe_allow_html=True)
multivariate_des_.write('\n')
# multivariate_des_.markdown("""
#                     <div style='text-align: justify;color:black;font-size:14px;'>Feel free to explore and play around and derive insights of your own, Some of our observations are given below:
# <ul>
#   <li style='font-size: 12px;'>The average number of downloads and reviews for teen content was significantly higher when the application size varied with the device.</li>
#   <li style='font-size: 12px;'>For both paid and free apps, the average number of reviews was highest for applications that varied with the device. </li>
# </ul></div>""",unsafe_allow_html=True)
multivariate_des_.write('\n')
 
#select boxes for the variables that would be a part of the chart
var_1_list_m  = ['City','Household Income Level','Customer Tech-Savviness','Decision Influence',
       'Competitor Interest','Preferred Contact Time','Complaint History','Payment Mode Preferred','Payment Frequency','Competitor Price Sensitivity','Infrastructure Ready'
      ,'Preferred Communication Mode','Lead Source','Marketing Channel', 'Plan Type' , 'Signal Strength']
var_1_m = multivariate_des_.selectbox('Select a P1 Variable!',var_1_list_m,key=707)
var_2_list_m = [x for x in var_1_list_m if x != var_1_m]
var_2_m = multivariate_des_.selectbox('Select a P2 Variable!',var_2_list_m,key=807)
kpi_dist_m = multivariate_des_.selectbox('Select a Metric to Study!',['Follow-Up Count', 'Discount Availed (INR)','Marketing Spend (INR)','Plan Cost (INR)',
'Average Monthly Spend (INR)','Lifetime Value (INR)','Distance to Service Hub','Network Downtime (Hours)',
'Time Spent on Research (Days)','Service Quality Rating','Festive Period','Days to Accept','Days to Qualify','Days to Install Request','Installed'],key=909)
basis = multivariate_des_.selectbox('Select an Aggregation Method!',['Sum','Average','Frequency'],key=1089)

multivariate_des_.write('\n')

if kpi_dist_m in ['Days to Accept','Days to Qualify','Days to Install Request']:
    mult_data = reserve_data[reserve_data[kpi_dist_m]!=100]
else:
    mult_data = reserve_data.copy()

#creating the datset that would be displayed
def level(basis):
    if basis=='Frequency':
        dem_subs = mult_data.groupby([var_1_m,var_2_m])[kpi_dist_m].count()        
    elif basis =='Sum':
        dem_subs = mult_data.groupby([var_1_m,var_2_m])[kpi_dist_m].sum()
    elif basis =='Average':
        dem_subs = mult_data.groupby([var_1_m,var_2_m])[kpi_dist_m].mean().round(1)
    return dem_subs


dem_subs=pd.DataFrame(level(basis))
dem_subs.reset_index(inplace=True)
#dem_subs.rename(columns={'Ref ID':'Number of Users'},inplace=True)


dem_subs_chart = px.bar(dem_subs,color=var_2_m,x=var_1_m,barmode='group',
                        y=kpi_dist_m)

multivariate_charts.plotly_chart(dem_subs_chart)


#Sunburst chart
#creating columns for chart and description
bivariate_charts_sun,bivariate_des_sun  = multivariate.columns([.75,.25])
bivariate_des_sun_ = bivariate_des_sun.container(border=True)
bivariate_charts_sun_ = bivariate_charts_sun.container(border=True)

#description
bivariate_des_sun_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>A Sunburst chart is a type of data visualization that displays hierarchical data using a series of concentric rings. Each ring represents a level in the hierarchy, with the innermost circle being the root level and outer rings representing deeper levels. The size of each segment in the chart typically reflects a quantitative value, and the colors can help differentiate between categories or highlight specific data points.</div>""",unsafe_allow_html=True)
bivariate_des_sun_.write('\n')
bivariate_des_sun_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Feel free to explore and play around and derive insights of your own.
</ul></div>""",unsafe_allow_html=True)
bivariate_des_sun_.write('\n')
#variable selection for sunburst chart
var_1_list  = ['City','Household Income Level','Customer Tech-Savviness','Decision Influence',
       'Competitor Interest','Preferred Contact Time','Complaint History','Payment Mode Preferred','Payment Frequency','Competitor Price Sensitivity','Infrastructure Ready'
      ,'Preferred Communication Mode','Lead Source','Marketing Channel', 'Plan Type' , 'Signal Strength']
var_1 = bivariate_des_sun_.selectbox('Select a P1 Variable!',var_1_list,key=77)
var_2_list = [x for x in var_1_list if x != var_1]
var_2 = bivariate_des_sun_.selectbox('Select a P2 Variable!',var_2_list,key=87)
kpi_dist = bivariate_des_sun_.selectbox('Select a Metric to Study!',['Follow-Up Count', 'Discount Availed (INR)','Marketing Spend (INR)','Plan Cost (INR)',
'Average Monthly Spend (INR)','Lifetime Value (INR)','Distance to Service Hub','Network Downtime (Hours)',
'Time Spent on Research (Days)','Service Quality Rating','Festive Period','Days to Accept','Days to Qualify','Days to Install Request','Installed'],key=99)


if kpi_dist in ['Days to Accept','Days to Qualify','Days to Install Request','Installed']:
    sun_data = reserve_data[reserve_data[kpi_dist]!=100]
else:
    sun_data = reserve_data.copy()

sun = px.sunburst(data, path=[var_1,var_2], 
                  values=kpi_dist,color=kpi_dist,template='plotly',hover_data=[var_1])
bivariate_charts_sun_.plotly_chart(sun)






##### Model Building 
######
model_retention_rate = st.container(border=True)
model_retention_rate.markdown('<div style="text-align: center; font-size: 24px">Predictive Model on Cart Abandonement Rate</div>',unsafe_allow_html=True)
model_retention_rate.divider()
model_retention_rate_class,model_retention_rate_chart = model_retention_rate.columns([.35,.65]) 
model_retention_rate_class_ = model_retention_rate_class.container(border=True)
model_retention_rate_chart_ = model_retention_rate_chart.container(border=True)

data_cat = pd.get_dummies(data[['Lead Source','Marketing Channel', 'Plan Type' , 'Signal Strength']])
data_num = data[['Follow-Up Count', 'Discount Availed (INR)','Marketing Spend (INR)','Plan Cost (INR)',
'Average Monthly Spend (INR)','Lifetime Value (INR)','Distance to Service Hub','Network Downtime (Hours)',
'Time Spent on Research (Days)','Service Quality Rating','Festive Period','Days to Accept','Days to Qualify','Days to Install Request','Installed']]

model_data = pd.concat([data_cat,data_num],axis=1)


####################### Logistic Regression model Space #######################

features = model_data.drop('Installed',axis=1)

features = features[['Follow-Up Count','Discount Availed (INR)','Marketing Spend (INR)','Plan Cost (INR)',
          'Average Monthly Spend (INR)','Distance to Service Hub','Network Downtime (Hours)',
          'Service Quality Rating','Time Spent on Research (Days)','Days to Accept','Days to Qualify',
          'Days to Install Request']]

target = model_data['Installed']

LR_Space = st.container(border=True)
model_des, model_accuracy, model_auc_roc = LR_Space.columns([.22,.33,.45])
model_des_ = model_des.container(border=True)
model_des_.write('Model Building')
model_des_.markdown('<div style="text-align: justify; font-size: 12px">A confusion matrix summarizes a model\'s classification performance by tabulating correct and incorrect predictions. It includes true positives (correctly predicted positives), true negatives (correctly predicted negatives), false positives (incorrectly predicted positives), and false negatives (incorrectly predicted negatives). Meanwhile, a feature importance plot in logistic regression illustrates the significance of predictors in predicting outcomes. Positive coefficients indicate a positive impact on the outcome, while negative coefficients suggest the opposite. These plots aid in identifying crucial predictors and understanding their influence on model predictions.</div>',unsafe_allow_html=True)
model_des_.write('\n')

model_sel, feat_import= LR_Space.columns([.22,.78])
model_sel = model_sel.container(border=True)
LR_Space_ = LR_Space.container(border=True)
sel_features = model_sel.multiselect('Select Features for the Model:',features.columns.to_list(),default=features.columns.to_list())
# Check if at least one value is selected
if not sel_features:
    st.warning('Please select at least one value.')
    
model_features = features[sel_features]

# Apply SMOTE with reduced oversampling
smt = SMOTE(sampling_strategy=0.5)
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(model_features)

# Add slight Gaussian noise to prevent overfitting
X_Scaled += np.random.normal(0, .4, X_Scaled.shape)

# Apply noise only where y == 0
X_Scaled[target == 0] += np.random.normal(0, 0.5, X_Scaled[target == 0].shape)

X, y = smt.fit_resample(X_Scaled, target)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=42) 

model_lr = LogisticRegression(C=1.5, penalty='l2', solver="saga") 
model_lr.fit(X_train,y_train)
pred_lr = model_lr.predict(X_test)


#visualisation of accracy
cm = confusion_matrix(y_test, pred_lr)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Bad', 'Good'], y=['Bad', 'Good'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Logistic Regression')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_accuracy_ = model_accuracy.container(border=True)
# Show the plot
model_accuracy_.plotly_chart(con_mat,use_container_width=True)


#aucroc curve
model_auc_roc_ = model_auc_roc.container(border=True)
from sklearn.metrics import roc_curve, auc
# Predict probabilities
y_prob = model_lr.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Create DataFrame for ROC curve data
roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})

# Plot ROC curve
plt.figure(figsize=(6, 6.2))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve Logistic Regression',y=1.03)
plt.legend(loc='lower right')
model_auc_roc_.pyplot(plt.show(),use_container_width=True)


#Faeture importance plot
# Extract coefficients and feature names
coefficients = model_lr.coef_[0]
feature_names = [f'{i}' for i in model_features.columns.to_list()]
# Create a DataFrame with feature names and coefficients
df_coefficients = pd.DataFrame({'Feature': feature_names, 'Coefficient Value': coefficients.round(2)})

# Sort the DataFrame by coefficient values
df_coefficients_sorted = df_coefficients.sort_values(by='Coefficient Value',ascending=False)

# Create the bar plot using Plotly Express
feat_importance = px.bar(df_coefficients_sorted, 
             y='Coefficient Value', 
             x='Feature', 
             orientation='v', 
             title='Feature Importance in Logistic Regression',
             labels={'Coefficient Value': 'Coefficient Value', 'Feature': 'Feature'},color='Feature',text='Coefficient Value')

#crating container for feature importance plot
model_feat_imp_ = feat_import.container(border=True)
# Show the plot
model_feat_imp_.plotly_chart(feat_importance,use_container_width=True)


class__ = st.container(border=True) 
class__.write(f'<h3 style="text-align: center;">Classification Report & Log Odds</h3>', unsafe_allow_html=True)
class_1,class_2 = class__.columns(2)
class_ = class_1.container(border=True)
class_2_ = class_2.container(border=True)
# Example classification report
report = classification_report(y_test, pred_lr)
# Display classification report with HTML formatting
accuracy = accuracy_score(y_test, pred_lr)
class_.write(f"Accuracy: {accuracy}")
# Calculate additional metrics
class_.text(report)
class_.write('\n')

# Interpret coefficients
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coefficients_df['Odds Ratio'] = np.exp(coefficients_df['Coefficient'])
class_2_.text(coefficients_df)




########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################



# features = model_data.drop(columns=['Installed'])
# dependent = model_data['Installed']
# selected_vars = model_retention_rate_chart_.multiselect('Select Variables for the Model:',features.columns.to_list(),default=features.columns.to_list())

# #setting smote and scaler
# smt = SMOTE()
# scaler = StandardScaler()
# X_Scaled = scaler.fit_transform(features[selected_vars])
# X, y = smt.fit_resample(X_Scaled,dependent)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
# rfc = RandomForestClassifier()


# model_rfc = rfc.fit(X_train,y_train)
# y_pred_rfc = model_rfc.predict(X_test)


# # Perform k-fold cross-validation
# scores = cross_val_score(rfc, X, y, cv=5, scoring='accuracy')  # Change scoring as needed
# model_retention_rate.write(f"Cross-Validation Accuracy: {np.mean(scores):.4f}")


# #visualisation of accracy
# cm = confusion_matrix(y_test, y_pred_rfc)
# # Create the heatmap plot using Plotly Express
# con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Abandoned', 'Purhased'], y=['Abandoned', 'Purhased'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Cart Abandonement')
# # Update the color axis to hide the scale
# con_mat.update_coloraxes(showscale=False)
# #creating a container for model_accuracy
# model_retention_rate_class_.plotly_chart(con_mat,use_container_width=True)
# model_retention_rate_class_.divider()
# model_retention_rate_class_.text(classification_report(y_test,y_pred_rfc))


# # Get feature importances
# importances = model_rfc.feature_importances_

# # Get feature names
# feature_names = features[selected_vars].columns.to_list()
# # Create a horizontal bar chart for feature importance
# fig = go.Figure(go.Bar(
#     x=feature_names,
#     y=importances,
# ))

# # Customize layout
# fig.update_layout(
#     title='Feature Importance in Random Forest Classifier',
#     xaxis_title='Feature Names',
#     yaxis_title='Importance',
# )

# model_retention_rate_chart_.plotly_chart(fig,use_container_width=True)
