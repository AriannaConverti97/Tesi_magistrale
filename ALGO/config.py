import itertools

SAMPLE_TIME_H = 4 # hours

n_actions = 7**3
n_cluster_states=660
alive_state=n_cluster_states
died_state=n_cluster_states+1
n_states = n_cluster_states + 2

state_space=['sodium', 'has_dobutamine', 'infection_at_icu_admission', 'has_dopamine', 'adjust_tidal_volume', 'lactate', 'neuroblock', 'bun', 'prone_24h', 'creatinine', 'uo', 
             'opioids', 'potassium', 'los_hospital_day', 'paco2', 'sapsii', 'is_sedated', 'diasbp', 'heartrate', 'bilirubin', 'wbc', 'has_vasopressin', 'resprate', 'sysbp', 
             'meanbp', 'has_epinephrine', 'transfusion_24h', 'ino_24h', 'hospmort', 'admission_category', 'height', 'septic_shock', 'ph', 'tempc', 'admission_type', 
             'weaning_success', 'bmi', 'crrt_24h', 'has_norepinephrine', 'gender', 'platelet', 'icu_outtime', 'age', 'bicarbonate', 'is_intubated', 'diuretic_24h', 'vent_type', 'spo2', 
             'hemoglobin', 'shock_index', 'sedation_type', 'sofa', 'hospmort28day', 'ards', 'los_icu_day', 'ibw', 'comorb_score', 'gcs', 'site_of_infection', 'pao2', 'icumort', 'pao2fio2ratio', 
             'weight', 'transfusion_type', 'steroid_24h', 'vent_mode']

action_space=['adjust_tidal_volume', 'fio2','peep'] 

all_identifiers = ['subject_id','hadm_id','stay_id']
all_demographics=['gender', 'age', 'dod', 'admittime', 'admission_type', 'admission_category', 'dischtime', 'los_hospital_day',
                  'hospmort','hospmort28day', 'hospmort90day', 'icumort', 'icu_intime','icu_outtime', 'los_icu_day', 'weight', 
                  'height', 'bmi', 'comorb_score']
all_vitalsign_vars=['sofa','percentual_missing_value_sofa', 'sapsii','percentual_missing_value_sapsii', 'ards', 'septic_shock',
                     'gcs', 'heartrate', 'sysbp', 'diasbp', 'meanbp', 'resprate', 'tempc', 'spo2', 'shock_index']
all_lab_vars=['ph', 'paco2', 'pao2', 'lactate', 'hemoglobin', 'albumin', 'platelet', 'wbc', 'bilirubin', 'creatinine', 'bun',
               'sodium', 'potassium', 'bicarbonate', 'pao2fio2ratio']
all_treatment_vars=['uo', 'rate_dobutamine', 'rate_dopamine', 'rate_epinephrine', 'rate_norepinephrine', 'rate_vasopressin', 'prone_24h', 
                    'crrt_24h', 'ino_24h', 'diuretic_24h','steroid_24h', 'transfusion_24h', 'transfusion_type', 'neuroblock',
                    'opioids', 'is_sedated', 'sedation_type', 'infection_at_icu_admission', 'site_of_infection']
all_vent_vars=['peep','fio2','tidal_volume']
all_other_vent_vars=['vent_mode', 'vent_type', 'is_intubated', 'intubation_time','extubation_time', 'is_tracheo', 'tracheo_ts', 
                     'end_mech_vent','time_no_mech_vent', 'next_intubation_time', 'weaning_success']

demographics = [ 'gender', 'age', 'admission_type', 'admission_category', 'hospmort90day',  'bmi','ibw', 'comorb_score','icumort', 'hospmort'] #rimosso icumort, hospmort, hospmort28days
vitalsign_vars= ['sofa', 'sapsii','gcs', 'heartrate', 'sysbp', 'diasbp', 'meanbp', 'resprate', 'tempc','spo2', 'shock_index','septic_shock','ards',]
lab_vars=['ph', 'paco2', 'pao2', 'lactate', 'hemoglobin', 'platelet', 'wbc', 'bilirubin', 'creatinine', 'bun', 'sodium', 
          'potassium', 'bicarbonate', 'pao2fio2ratio']
treatment_vars=['uo', 'rate_dobutamine', 'rate_dopamine', 'rate_epinephrine', 'rate_norepinephrine','rate_vasopressin', 'prone_24h', 
                'crrt_24h', 'ino_24h', 'diuretic_24h', 'steroid_24h', 'transfusion_24h', 'transfusion_type', 'neuroblock','opioids', 
                'is_sedated', 'sedation_type', 'infection_at_icu_admission', 'site_of_infection']
vent_vars = ['peep','fio2','tidal_volume']
other_vent_vars=['vent_mode', 'vent_type', 'is_intubated', 'weaning_success']

col_to_remove = ['hadm_id', 'stay_id', 'admittime', 'dischtime', 'icu_intime', 'intubation_time', 'extubation_time', 
                 'end_mech_vent', 'time_no_mech_vent', 'percentual_missing_value_sofa', 'percentual_missing_value_sapsii', 'albumin',
                 'dod', 'tracheo_ts', 'next_intubation_time', 'is_tracheo']



inf = None
ffill_windows_clinical = {
    
    'sofa': 24 / SAMPLE_TIME_H,
    'sapsii': inf,
    
    'gcs': 48 / SAMPLE_TIME_H,
    'heartrate': 48 / SAMPLE_TIME_H,
    'sysbp': 48 / SAMPLE_TIME_H,
    'diasbp': 48 / SAMPLE_TIME_H,
    'meanbp': 48 / SAMPLE_TIME_H,
    'resprate': 48 / SAMPLE_TIME_H,
    'tempc': 48 / SAMPLE_TIME_H,
    'spo2': 48 / SAMPLE_TIME_H,
    'shock_index': 48 / SAMPLE_TIME_H,

    'ph': 48 / SAMPLE_TIME_H,
    'paco2': 48 / SAMPLE_TIME_H,
    'pao2': 48 / SAMPLE_TIME_H,
    'lactate': 48 / SAMPLE_TIME_H,
    'hemoglobin': 48 / SAMPLE_TIME_H,
    #'albumin': 48 / SAMPLE_TIME_H,
    'platelet': 48 / SAMPLE_TIME_H,
    'wbc': 48 / SAMPLE_TIME_H,
    'bilirubin': 48 / SAMPLE_TIME_H,
    'creatinine': 48 / SAMPLE_TIME_H,
    'bun': 48 / SAMPLE_TIME_H,
    'sodium': 48 / SAMPLE_TIME_H,
    'potassium': 48 / SAMPLE_TIME_H,
    'bicarbonate': 48 / SAMPLE_TIME_H    
}

limit={
    'bmi': (10, 60),
    'comorb_score': (0, 24),
    'ph':(6.9,7.7),
    'paco2': (10,130),
    'pao2': (30,630),
    'lactate': (0,20),
    'hemoglobin': (0,20),
    #'albumin': (0,10),
    'platelet': (0,1500),
    'wbc': (0.5,200),
    'bilirubin': (0.1,300),
    'creatinine': (0,25),
    'bun': (5,200),
    'sodium': (110,160),
    'potassium': (2,6),
    'bicarbonate': (6,50),
    'uo': (0,16000),
    'gcs': (3,15),
    'heartrate': (10,250),
    'sysbp': (30,300),
    'diasbp': (10,200),
    #'meanbp': (),
    'resprate': (4,50),
    'tempc': (20,50),
    'spo2': (40,100),
    #'tidal_volume': (),
    'peep': (0,20),
    'fio2': (21,100),
    'sofa': (0,24),
    'sapsii': (0,163),
    'ibw': (25, 400)
}

# Action space bins.
# Table 2a in Supplementary Material
# We assume these ranges are [lower bound, upper bound).
tv_bins = [
    (0, 2.5),
    (2.5, 5),
    (5, 7.5),
    (7.5, 10),
    (10, 12.5),
    (12.5, 15),
    (15, float('inf')),
]
peep_bins = [
    (0, 5),
    (5, 7),
    (7, 9),
    (9, 11),
    (11, 13),
    (13, 15),
    (15, float('inf')),
]
fio2_bins = [
    (20, 30), # NOTE: Peine uses 25 - 30, but this does not cover all of the data
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
    (50, 55),
    (55, float('inf')),
]

action_bin_definition = list(itertools.product(tv_bins, fio2_bins, peep_bins))

fio2_peep_table = (
    (30,  5),
    (40,  5),
    (40,  8),
    (50,  8),
    (50,  10),
    (60,  10),
    (70,  10),
    (70,  12),
    (70,  14),
    (80,  14),
    (90,  14),
    (90,  16),
    (90,  18),
    (100, 18),
    (100, 20),
    (100, 22),
    (100, 24),
)
fio2_min = 30
fio2_max = 100