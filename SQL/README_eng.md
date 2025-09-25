To compile the files, the following queries from MIMIC must be executed:



\- \*\*height\*\*: \[first\_day\_height.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts\_postgres/firstday/first\_day\_height.sql)  

\- \*\*weight\_durations\*\*: \[weight\_durations.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts\_postgres/demographics/weight\_durations.sql)  

\- \*\*antibiotic\*\*: \[antibiotic.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts\_postgres/medication/antibiotic.sql)  

\- \*\*suspicion\_of\_infection\*\*: \[suspicion\_of\_infection.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts\_postgres/sepsis/suspicion\_of\_infection.sql)  

\- \*\*oxygen\_delivery\*\*: \[oxygen\_delivery.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts\_postgres/measurement/oxygen\_delivery.sql)  

\- \*\*ventilator\_settings\*\*: \[ventilator\_setting.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts\_postgres/measurement/ventilator\_setting.sql)  



\### Execution order of queries



1\. \*\*Demographics\*\*: creates the table `tesi.demographics` aggregating demographic and clinical data from `patients`, `admissions`, and `icustays`.  

&nbsp;  - Computes age at admission, hospital and ICU length of stay, mortality indicators (hospital, 28/90 days, in-ICU)  

&nbsp;  - Integrates DNR information extracted from `chartevents`.



2\. \*\*Cohort\*\*: creates the table `tesi.cohort` selecting adult patients (≥18 years) from the first hospital and ICU admission.  

&nbsp;  - Includes only patients with complete mortality information and excludes DNR orders.



3\. \*\*Charlson\_comorbidity\*\*: creates `tesi.charlson` extracting all information to compute the Charlson Comorbidity Index (CCI) considering ICD-9 and ICD-10 codes.



4\. \*\*Bmi\_w\_h\*\*: creates `tesi.bmi` aggregating weight (kg) and height (cm) for each patient from `chartevents`, computing BMI as:  

&nbsp;  `weight / ((height / 100)^2) \[Kg/m²]`.



5\. \*\*Vasopressors\*\*: creates `tesi.vasopressors` containing aggregated doses of main vasopressors administered in the ICU.  

&nbsp;  - Extracts infusions from `inputevents` for norepinephrine (mg), dopamine (mg), vasopressin (unit), dobutamine (mg), and epinephrine (mg), normalizing units.



6\. \*\*UrineOutput\*\*: creates `tesi.getUrineOutput`, aggregating urine output volume per patient, hospital stay (`hadm\_id`), and ICU stay (`stay\_id`) at `charttime`.  

&nbsp;  - Considers multiple types of urine output and treats genito-urinary irrigation volumes as negative for correct net balance.



7\. \*\*AllLabValues\*\*: creates `tesi.getAllLabvalues` containing main lab measurements during ICU stay.  

&nbsp;  - Data from `labevents` and `chartevents`, joined to ICU stays, selecting clinically relevant values (albumin, potassium, sodium, creatinine, bilirubin, etc.).



8\. \*\*AllVitalSigns\*\*: creates `tesi.getAllVitalSigns`, aggregating vital signs and Glasgow Coma Scale (GCS) scores for each ICU patient.  

&nbsp;  - Includes HeartRate, blood pressure, respiration, temperature, SpO₂  

&nbsp;  - Computes GCS components (motor, verbal, eyes) with logic for intubated patients.



9\. \*\*Infection\*\*: creates `tesi.infection` to identify suspected infections at ICU admission, with infection site deduced from `microbiologyevents`.



10\. \*\*Sedatives\*\*: creates `tesi.sedatives`, recording sedative exposure every 4 hours per ICU patient.



11\. \*\*Opioids\*\*: creates `tesi.opioids`, recording opioid exposure every 4 hours.



12\. \*\*Neuroblocks\*\*: creates `tesi.neuroblock`, recording neuromuscular blocker exposure every 4 hours.



13\. \*\*Flags\*\*: creates `tesi.flags`, aggregating treatment and condition indicators every 24 hours, including CRRT, INO, pronation, diuretics, steroids, transfusions.



14\. \*\*Ards\_shock\_admission\*\*: creates `tesi.ards\_shock\_admission`, assigning clinical admission categories in ICU and detecting ARDS and septic shock.



15\. \*\*Flag\_all\*\*: creates `tesi.allFlags`, merging neuroblock, opioids, sedatives, flags, infection, and ARDS/shock data.



16\. \*\*AllLabValues\_times\*\*: creates `getalllabvalues\_time`, mean lab values per patient per 4-hour interval.



17\. \*\*UrineOutput\_times\*\*: creates `geturineoutput\_time`, mean urine output every 4 hours per patient.



18\. \*\*AllVitalSigns\_times\*\*: creates `getallvitalsigns\_time`, mean and maximum of vital signs every 4 hours.



19\. \*\*Vasopressors\_times\*\*: creates `getvasopressor\_time`, mean vasopressor administration rate every 4 hours.



20\. \*\*AllDemoCohort\*\*: creates `alldemocohort`, merging cohort, BMI, and Charlson data.



21\. \*\*Vent\_time\*\*: creates table with invasive ventilation events (start, end, tracheotomy).



22\. \*\*VentParam\*\*: creates `tesi.vent\_param`, aggregating ventilator parameters every 4 hours.



23\. \*\*OverallTable1\*\*: creates `tesi.overalltable1`, aggregating labs, urine output, vasopressors, flags, vital signs, and ventilator parameters.  

&nbsp;  - Computes shock index and PaO₂/FiO₂ ratio.



24\. \*\*Sapsii\*\*: creates `tesi.sapsii`, computing SAPS II score in the first 24 hours of ICU stay.



25\. \*\*Sofa\*\*: creates `tesi.sofa`, computing daily SOFA scores aggregating physiological parameters.



26\. \*\*Extub\_intub\_time\*\*: creates `tesi.prova\_intubation`, marking intubation status every 4 hours with timestamps.



27\. \*\*OverallTable2\*\*: creates `overalltablePROVA`, merging OverallTable1 with intubation, SOFA, and SAPS II.



28\. \*\*Final\*\*: returns the final table, considering patients intubated at least once with at least one pH measurement.



