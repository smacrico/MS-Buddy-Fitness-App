## order to execut scripts

## Update Main GarminData ###


& C:/Python313/python.exe C:/Python313/Scripts/garmindb_cli.py --all --download --import --analyze --latest

& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/CG_jHeelFitnessProject/GarminParse_PlugIn/jHeel_plugin v5.1.py"



## Run Analysis
& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/createRunAnalDB.py"

& C://Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Apex-RunAnalysis/scripts/RunningAnalysis_v60.py"


    #interactive DashBoard - Run Analysis
   
    streamlit run scripts/app.py

## HRV Analysis
& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Mercury-HRVAnalysis/scripts/HRV_datawarehouse.py"

& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Mercury-HRVAnalysis/scripts/HRV_Analytics_v3.py"
& C:/Python313/python.exe "C:/smakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/Mercury-HRVAnalysis/scripts/HRV_Analytics_v3.3.py"


    #interactive DashBoard - HRV data analysis
    streamlit run hrv_streamlit_dashboard.py



## GarminHealthData (v1.0 & v2.0)

    #interactive DashBoard - Garmin Health Data
   
    streamlit run app/main.py


