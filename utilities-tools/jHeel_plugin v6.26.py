# this file is used to parse the fit file and store the dev_data in the database
""" stelios (c) steliosmacrico "jHeel 2024 creating plugin"""


######################################
"jHEEL main version"##################
######################################


import sqlite3
import os
import logging
import datetime


# Set up logging
now = datetime.datetime.now()
timestamp = now.strftime('%Y%m%d_%H%M%S')
logging.basicConfig(filename=f'c:/temp/logsFitnessApp/jheel_parse_Fields-v5{timestamp}.log', level=logging.INFO)
logging.info('Starting script...')
print('Starting script...')


def create_table_if_not_exists():
    conn = sqlite3.connect(r'c:/smakrykoDBs/artemis.db')
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS Artemistbl_Fields')
    logging.info('Table dropped successfully.')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Artemistbl_Fields (
            activity_id INT PRIMARY KEY,
            name TEXT,
            timestamp TEXT,
            sport TEXT,
            avg_heart_rate INT,
            max_heart_rate INT,
            total_elapsed_time INT,
            distance REAL,
            hrv INT,
            fat INT,
            total_fat INT,  
            carbs INT,
            total_carbs INT,
            VO2maxSmooth INT,
            VO2maxSession INT,
            CardiacDrift INT,    
            CooperTest INT,
            steps INT,
            stress_hrpa INT,
            HR_RS_Deviation_Index INT,
            hrv_sdrr_f INT,
            hrv_pnn50 INT,                           
            hrv_pnn20 INT,
            rmssd INT,
            aarmssd INT,
            lnrmssd INT,
            sdnn INT,
            aasdnn INT,
            sdsd INT,
            nn50 INT,
            nn20 INT,
            pnn20 INT,
            Long INT,
            Short INT,
            Ectopic_S INT,
            hrv_rmssd INT,
            SD2 INT,
            SD1 INT,
            LF INT,
            HF INT,
            VLF INT,
            pNN50 INT, 
            LFnu INT, 
            HFnu INT,
            MeanHR INT, 
            MeanRR INT, 
            Running_Economy TEXT, 
            aHRV INT, 
            arMSSD INT, 
            aSDNN INT, 
            calories INT,
            total_calories INT,
            total_training_effect REAL,
            recovery_heart_rate INT,
            aerobic_efficiency REAL,
            avg_cadence INT,
            max_cadence INT,
            total_strides INT
        )
    ''')
    logging.info('Table Artemistbl_Fields created successfully.')
    conn.commit()
    conn.close()


from fitparse import FitFile


def parse_all_fit_files_in_folder(folder_path):
    all_session_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.fit'):
            try:
                fit_file_path = os.path.join(folder_path, filename)
                activity_id = os.path.splitext(filename)[0]
                activity_id = activity_id.split('_')[0]
                session_data = parse_fit_file(fit_file_path, activity_id)
                all_session_data.extend(session_data)
            except Exception as e:
                logging.error(f'Error parsing file {filename}: {e}')
                print(f'Error parsing file {filename}: {e}')
                continue
    logging.info('All files parsed successfully.')  
    print('All files parsed successfully.')
    return all_session_data


def parse_fit_file(file_path, activity_id):
    fit_file = FitFile(file_path)
    messages = fit_file.messages
    session_data = []
    name = None

    for msg in messages:
        if msg.name == 'sport':
            fields = msg.fields
            field_dict = {field.name: field.value for field in fields}
            name = field_dict.get('name')

        if msg.name == 'session':
            # Extract all
            fields
            fields = msg.fields
            field_dict = {field.name: field.value for field in fields}

            # TEMP DEBUG: log all session field keys
            logging.info(f"SESSION FIELDS {activity_id}: {list(field_dict.keys())}")

            timestamp = field_dict.get('timestamp')
            sport = field_dict.get('sport')
            avg_heart_rate = field_dict.get('avg_heart_rate')
            max_heart_rate = field_dict.get('max_heart_rate')
            total_elapsed_time = field_dict.get('total_elapsed_time')
            distance = field_dict.get('total_distance')
            hrv = field_dict.get('HRV')
            fat = field_dict.get('Fat')  
            total_fat = field_dict.get('Total Fat')
            carbs = field_dict.get('Carbs')
            total_carbs = field_dict.get('Total Carbs')
            VO2maxSmooth = field_dict.get('VO2maxSmooth')
            VO2maxSession = field_dict.get('VO2maxSession')
            CardiacDrift = field_dict.get('CardiacDrift')
            CooperTest = field_dict.get('CooperTest')
            steps = field_dict.get('Steps') or field_dict.get('steps')
            stress_hrpa = field_dict.get('stress_hrpa')
            HR_RS_Deviation_Index = field_dict.get('HR-RS Deviation Index')
            hrv_sdrr_f = field_dict.get('hrv_sdrr_f')
            hrv_pnn50 = field_dict.get('hrv_pnn50')
            hrv_pnn20 = field_dict.get('hrv_pnn20')
            rmssd = field_dict.get('RMSSD')
            aarmssd = field_dict.get('armssd')
            lnrmssd = field_dict.get('lnRMSSD')
            sdnn = field_dict.get('SDNN')
            aasdnn = field_dict.get('asdnn')
            sdsd = field_dict.get('SDSD')
            nn50 = field_dict.get('NN50')
            nn20 = field_dict.get('NN20')
            pnn20 = field_dict.get('pNN20')
            Long = field_dict.get('Long')
            Short = field_dict.get('Short')
            Ectopic_S = field_dict.get('Ectopic-S')
            hrv_rmssd = field_dict.get('hrv_rmssd')
            SD2 = field_dict.get('SD2')
            SD1 = field_dict.get('SD1')
            LF = field_dict.get('LF')
            HF = field_dict.get('HF')
            VLF = field_dict.get('VLF')
            pNN50 = field_dict.get('pNN50')
            LFnu = field_dict.get('LFnu')
            HFnu = field_dict.get('HFnu')
            MeanHR = field_dict.get('Mean HR')
            MeanRR = field_dict.get('Mean RR')
            Running_Economy = field_dict.get('Running Economy')
            aHRV = field_dict.get('aHRV')
            arMSSD = field_dict.get('arMSSD')
            aSDNN = field_dict.get('aSDNN')
            calories = field_dict.get('calories')

            # NEW FIELDS
            total_calories = field_dict.get('total_calories')
            total_training_effect = field_dict.get('total_training_effect') or field_dict.get('training_effect')
            recovery_heart_rate = field_dict.get('recovery_heart_rate') or field_dict.get('recovery_hr')
            aerobic_efficiency = (field_dict.get('aerobic_efficiency') or 
                                  field_dict.get('Aerobic_efficiency') or 
                                  field_dict.get('Aerobic Efficiency'))
            avg_cadence = (field_dict.get('avg_cadence') or 
                           field_dict.get('avg_cycling_cadence') or 
                           field_dict.get('avg_running_cadence') or
                           field_dict.get('cadence'))
            max_cadence = (field_dict.get('max_cadence') or 
                           field_dict.get('max_cycling_cadence') or 
                           field_dict.get('max_running_cadence'))
            total_strides = field_dict.get('total_strides')

            session_data.append({
                'activity_id': activity_id,
                'name': name,
                'timestamp': timestamp,
                'sport': sport,
                'avg_heart_rate': avg_heart_rate,
                'max_heart_rate': max_heart_rate,
                'total_elapsed_time': total_elapsed_time,
                'distance': distance,
                'hrv': hrv,
                'fat': fat,
                'total_fat': total_fat,
                'carbs': carbs,
                'total_carbs': total_carbs,
                'VO2maxSmooth': VO2maxSmooth,
                'VO2maxSession': VO2maxSession,
                'CardiacDrift': CardiacDrift,
                'CooperTest': CooperTest,
                'steps': steps,
                'stress_hrpa': stress_hrpa,
                'HR_RS_Deviation_Index': HR_RS_Deviation_Index,
                'hrv_sdrr_f': hrv_sdrr_f,
                'hrv_pnn50': hrv_pnn50,
                'hrv_pnn20': hrv_pnn20,
                'rmssd': rmssd,
                'aarmssd': aarmssd,
                'lnrmssd': lnrmssd,
                'sdnn': sdnn,
                'aasdnn': aasdnn,
                'sdsd': sdsd,
                'nn50': nn50,
                'nn20': nn20,
                'pnn20': pnn20,
                'Long': Long,
                'Short': Short,
                'Ectopic_S': Ectopic_S,
                'hrv_rmssd': hrv_rmssd,
                'SD2': SD2,
                'SD1': SD1,
                'LF': LF,
                'HF': HF,
                'VLF': VLF,
                'pNN50': pNN50,
                'LFnu': LFnu,
                'HFnu': HFnu,
                'MeanHR': MeanHR,
                'MeanRR': MeanRR,
                'Running_Economy': Running_Economy,
                'aHRV': aHRV,
                'arMSSD': arMSSD,
                'aSDNN': aSDNN,
                'calories': calories,
                'total_calories': total_calories,
                'total_training_effect': total_training_effect,
                'recovery_heart_rate': recovery_heart_rate,
                'aerobic_efficiency': aerobic_efficiency,
                'avg_cadence': avg_cadence,
                'max_cadence': max_cadence,
                'total_strides': total_strides
            })
            logging.info(f'Parsed session data for activity ID {activity_id}.')
    return session_data


def insert_data_into_db(data):
    conn = sqlite3.connect('c:/smakrykoDBs/artemis.db')
    cursor = conn.cursor()

    # EXACT column order matching table - 58 columns total
    columns = [
        'activity_id', 'name', 'timestamp', 'sport', 'avg_heart_rate', 'max_heart_rate', 
        'total_elapsed_time', 'distance', 'hrv', 'fat', 'total_fat', 'carbs', 'total_carbs', 
        'VO2maxSmooth', 'VO2maxSession', 'CardiacDrift', 'CooperTest', 'steps', 
        'stress_hrpa', 'HR_RS_Deviation_Index', 'hrv_sdrr_f', 'hrv_pnn50', 'hrv_pnn20', 
        'rmssd', 'aarmssd', 'lnrmssd', 'sdnn', 'aasdnn', 'sdsd', 'nn50', 'nn20', 
        'pnn20', 'Long', 'Short', 'Ectopic_S', 'hrv_rmssd', 'SD2', 'SD1', 'LF', 
        'HF', 'VLF', 'pNN50', 'LFnu', 'HFnu', 'MeanHR', 'MeanRR', 'Running_Economy', 
        'aHRV', 'arMSSD', 'aSDNN', 'calories', 'total_calories', 'total_training_effect', 
        'recovery_heart_rate', 'aerobic_efficiency', 'avg_cadence', 'max_cadence', 'total_strides'
    ]
    
    specific_fields = ['fat','total_fat','carbs','total_carbs','VO2maxSmooth','sport',
                      'avg_heart_rate','max_heart_rate','total_elapsed_time','VO2maxSession',
                      'timestamp','CardiacDrift','CooperTest','steps','stress_hrpa',
                      'HR_RS_Deviation_Index','hrv_sdrr_f','hrv_pnn50','hrv_pnn20','rmssd',
                      'aarmssd','lnrmssd','sdnn','aasdnn','sdsd','nn50','nn20','pnn20',
                      'Long','Short','Ectopic_S','hrv_rmssd','SD2','SD1','LF','HF','VLF',
                      'pNN50','LFnu','HFnu','MeanHR','MeanRR','Running_Economy','aHRV',
                      'arMSSD','aSDNN','calories','total_calories','avg_cadence']

    placeholders = ','.join('?' * len(columns))
    column_names = ','.join(columns)

    for session in data:
        if all(session.get(field) is None for field in specific_fields):
            continue

        values = [session.get(col, None) for col in columns]
        
        cursor.execute(f'''
            INSERT OR REPLACE INTO Artemistbl_Fields 
            ({column_names}) VALUES ({placeholders})
        ''', values)

    conn.commit()
    conn.close()


def create_view_if_not_exists():
    conn = sqlite3.connect('c:/smakrykoDBs/artemis.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE VIEW IF NOT EXISTS RunFields_view AS
        SELECT activities.*
        FROM activities
        INNER JOIN Artemistbl_Fields ON activities.activity_id = Artemistbl_Fields.activity_id
        WHERE Artemistbl_Fields.sport = "running" ORDER BY Artemistbl_Fields.timestamp DESC
    ''')
    logging.info('View for Run created successfully.')
    conn.commit()
    conn.close()


if __name__ == "__main__":  
    create_table_if_not_exists()
    create_view_if_not_exists()
    # all_session_data = parse_all_fit_files_in_folder(r'C:/SmakrykoDev/GitHubRepos/MS-Buddy-Fitness-App/utilities-tools/fit_test_files')
    # # all_session_data = parse_all_fit_files_in_folder(r'C:/smakryko/MS-Buddy-Fitness-App/utilities-tools/fit_test_files')
    all_session_data = parse_all_fit_files_in_folder('c:/users/djsco/jheelhealthdatav2/fitfiles/activities')
    insert_data_into_db(all_session_data)
    logging.info('All data inserted successfully.')
    print('All data inserted successfully.')
    logging.info('Script completed successfully.')
    print('Script completed successfully.')
