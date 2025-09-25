
::python 0_preprocess.py --save True        :: dati.csv -> Data\dati_imputati.csv
@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --batch_size 128 --n_iters 5
@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --batch_size 256 --n_iters 5

@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --batch_size 128 --n_iters 10
@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --batch_size 256 --n_iters 10

@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --batch_size 128 --n_iters 100
@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --batch_size 256 --n_iters 100

@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --intermidiate_reward True --batch_size 128 --n_iters 5
@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --intermidiate_reward True --batch_size 256 --n_iters 5

@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --intermidiate_reward True --batch_size 128 --n_iters 10
@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --intermidiate_reward True --batch_size 256 --n_iters 10

@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --intermidiate_reward True --batch_size 128 --n_iters 100
@REM python prova_train_forse_definitiva.py --model DiscreteBCQ --intermidiate_reward True --batch_size 256 --n_iters 100
::python visualize_plot.py 

::python prova_train_forse_definitiva.py --model DiscreteCQL  
::python prova_train_forse_definitiva.py --model DiscreteCQL --intermidiate_reward True

@echo off
set MODELS=DiscreteBCQ
set BATCH_SIZES=128 256
set N_ITERS=5 10 100
set INTERMEDIATE_REWARD=True False


for %%m in (%MODELS%) do (
    for %%b in (%BATCH_SIZES%) do (
        for %%n in (%N_ITERS%) do (
            for %%r in (%INTERMEDIATE_REWARD%) do (
                if "%%r"=="True" (
                    python prova_train_forse_definitiva.py --model %%m --intermidiate_reward True --batch_size %%b --n_iters %%n
                ) else (
                    python prova_train_forse_definitiva.py --model %%m --batch_size %%b --n_iters %%n
                )
            )
        )
    )
)


