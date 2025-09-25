Per poter compilare i file bisogna aver eseguito le seguenti query presenti nel MIMIC:

- **height**: [first_day_height.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/firstday/first_day_height.sql)  
- **weight_durations**: [weight_durations.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/demographics/weight_durations.sql)  
- **antibiotic**: [antibiotic.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/medication/antibiotic.sql)  
- **suspicion_of_infection**: [suspicion_of_infection.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/sepsis/suspicion_of_infection.sql)  
- **oxygen_delivery**: [oxygen_delivery.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/measurement/oxygen_delivery.sql)  
- **ventilator_settings**: [ventilator_setting.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts_postgres/measurement/ventilator_setting.sql)  

### Ordine di esecuzione delle query

1. **Demographics**: crea la tabella `tesi.demographics` aggregando dati anagrafici e clinici da `patients`, `admissions` e `icustays`.  
   - Calcola età al momento del ricovero, durata della degenza ospedaliera e ICU, indicatori di mortalità (ospedaliera, 28/90 giorni, in-ICU)  
   - Integra l’informazione DNR estratta da `chartevents`.

2. **Cohort**: crea la tabella `tesi.cohort` selezionando pazienti adulti (≥18 anni) dal primo ricovero ospedaliero e ICU.  
   - Include solo soggetti con informazioni complete sulla mortalità e esclude ordini DNR.

3. **Charlson_comorbidity**: crea `tesi.charlson` estraendo informazioni per calcolare il Charlson Comorbidity Index (CCI) considerando codici ICD-9 e ICD-10.

4. **Bmi_w_h**: crea `tesi.bmi` aggregando peso (Kg) e altezza (cm) per ogni paziente da `chartevents`, calcolando il BMI come:  
   `peso / ((altezza / 100)^2) [Kg/m²]`.

5. **Vasopressors**: crea `tesi.vasopressors` contenente il dosaggio aggregato dei principali vasopressori somministrati in ICU.  
   - Estrae infusioni da `inputevents` per norepinefrina (mg), dopamina (mg), vasopressina (unit), dobutamina (mg) ed epinefrina (mg), normalizzando le unità.

6. **UrineOutput**: crea `tesi.getUrineOutput`, aggregando il volume di output urinario per paziente, ricovero (`hadm_id`) e ICU stay (`stay_id`) a livello di `charttime`.  
   - Considera diversi tipi di output urinario e tratta volumi di irrigazione genito-urinaria come negativi.

7. **AllLabValues**: crea `tesi.getAllLabvalues`, contenente principali valori di laboratorio rilevati in ICU.  
   - Dati da `labevents` e `chartevents`, selezionando valori clinicamente rilevanti (albumina, potassio, sodio, creatinina, bilirubina, ecc.).

8. **AllVitalSigns**: crea `tesi.getAllVitalSigns`, aggregando segni vitali e punteggio GCS per ciascun paziente in ICU.  
   - Include HeartRate, pressione arteriosa, respirazione, temperatura, SpO₂  
   - Calcolo del punteggio GCS (motorio, verbale, occhi), con logica per pazienti intubati.

9. **Infection**: crea `tesi.infection` per identificare infezioni sospette all’ingresso in ICU, con sede dell’infezione dedotta da `microbiologyevents`.

10. **Sedatives**: crea `tesi.sedatives`, registrando esposizione a sedativi ogni 4 ore per paziente ICU.

11. **Opioids**: crea `tesi.opioids`, registrando esposizione a oppioidi ogni 4 ore.

12. **Neuroblocks**: crea `tesi.neuroblock`, registrando esposizione a bloccanti neuromuscolari ogni 4 ore.

13. **Flags**: crea `tesi.flags`, aggregando vari indicatori di trattamento ogni 24 ore, con dettagli su CRRT, INO, pronazione, diuretici, steroidi, trasfusioni.

14. **Ards_shock_admission**: crea `tesi.ards_shock_admission`, assegnando categorie cliniche ICU e rilevando ARDS e shock settico.

15. **Flag_all**: crea `tesi.allFlags`, unendo neuroblock, opioids, sedatives, flags, infection e ARDS/shock.

16. **AllLabValues_times**: crea `getalllabvalues_time`, valori medi di laboratorio per paziente e intervalli di 4 ore.

17. **UrineOutput_times**: crea `geturineoutput_time`, media produzione urinaria ogni 4 ore per paziente.

18. **AllVitalSigns_times**: crea `getallvitalsigns_time`, media e massimo dei parametri vitali ogni 4 ore.

19. **Vasopressors_times**: crea `getvasopressor_time`, tasso medio somministrazione vasopressori ogni 4 ore.

20. **AllDemoCohort**: crea `alldemocohort`, unendo cohort, bmi e charlson.

21. **Vent_time**: crea tabella con eventi di ventilazione invasiva (inizio, fine, tracheotomia).

22. **VentParam**: crea `tesi.vent_param`, aggregando parametri ventilatori ogni 4 ore.

23. **OverallTable1**: crea `tesi.overalltable1`, aggregando valori di laboratorio, urine, vasopressori, flags, segni vitali e parametri ventilatori.  
   - Calcolo di shock index e PaO₂/FiO₂ ratio.

24. **Sapsii**: crea `tesi.sapsii`, calcolando punteggio SAPS II nelle prime 24 ore di ICU.

25. **Sofa**: crea `tesi.sofa`, calcolando punteggio SOFA giornaliero aggregando parametri fisiologici.

26. **Extub_intub_time**: crea `tesi.prova_intubation`, segnando intubazione ogni 4 ore e timestamp.

27. **OverallTable2**: crea `overalltablePROVA`, unendo overalltable1 con intubazione, SOFA e SAPS II.

28. **Final**: restituisce la tabella finale, considerando pazienti intubati almeno una volta con almeno un valore di pH.
