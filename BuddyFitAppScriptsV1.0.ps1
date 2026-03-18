## order to execut scripts

## Update Main GarminData ###


& C:/Python313/python.exe C:/Python313/Scripts/garmindb_cli.py --all --download --import --analyze --latest

##Stelios laptop - jHeel ##
& C:/Python313/python.exe C:/Users/djsco/AppData/Roaming/Python/Python313/Scripts/garmindb_cli.py --all --download --import --analyze --latest

& C:/Python313/python.exe "C:/smakrykoDev/GitHub_dls/MS-Buddy-Fitness-App/utilities-tools/jHeel_plugin v6.26.py"
& C:/Python313/python.exe "C:/smakrykoDev/GitHub_dls/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/createRunAnalDB - v6.26.py"
& C:/Python313/python.exe c:/smakrykoDev/GitHub_dls/MS-Buddy-Fitness-App/APEX-RunAnalysis/Dev_Scripts/RunningAnalysis_v6.26-Dev.py
& C:/Python313/python.exe c:/smakrykoDev/GitHub_dls/MS-Buddy-Fitness-App/Mercury-HRVAnalysis/Dev_Scripts/HRV_Analytics_v6.25-DEV.py

    #interactive DashBoard - Run Analysis
   
    streamlit run scripts/app.py

    #interactive DashBoard - HRV data analysis
    streamlit run hrv_streamlit_dashboard.py

####################################3

## Run Analysis
& C:/Python313/python.exe "C:/smakrykoDev/GitHub_dls/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/createRunAnalDB.py"
& C:/Python313/python.exe "C:/smakrykoDev/GitHub_dls/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/createRunAnalDB - v6.26.py"




& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/CG_jHeelFitnessProject/GarminParse_PlugIn/jHeel_plugin v5.1.py"

& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/utilities-tools/jHeel_plugin v6.26.py"



## Run Analysis
& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/createRunAnalDB.py"
& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/createRunAnalDB - v6.26.py"

& C://Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/RunningAnalysis_v60.py"

& C://Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/RunningAnalysis_v6.5.py"

& C://Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/RunningAnalysis_v6.26 .py"
    #interactive DashBoard - Run Analysis
   
    streamlit run scripts/app.py

    #interactive DashBoard - HRV data analysis
    streamlit run hrv_streamlit_dashboard.py


## HRV Analysis
    ## HRV Data Warehouse (v1.0 & v2.0)
    ## v1.0 is the 'old' version, v2.0 is the new version with more features
& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Mercury-HRVAnalysis/scripts/HRV_datawarehouse.py"
& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Mercury-HRVAnalysis/scripts/HRV_dataWareHouse_V2.0.py"


& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Mercury-HRVAnalysis/scripts/HRV_Analytics_v3.py"
& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Mercury-HRVAnalysis/scripts/HRV_Analytics_v3.3.py"


    #interactive DashBoard - HRV data analysis
    streamlit run hrv_streamlit_dashboard.py



## GarminHealthData (v1.0 & v2.0)

    #interactive DashBoard - Garmin Health Data
   
    streamlit run app/main.py


