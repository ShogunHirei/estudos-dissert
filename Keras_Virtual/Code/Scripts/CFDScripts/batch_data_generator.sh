# Script para gerar múltiplos arquivos de dados de slice

# Inserir pasta aonde são mantidas as pastas com os dados do modelo da rede
#       --> considerando  estrutura "/pasta_pai/Pasta_do_tensorboard/Modelo_da_rede"
BATCH=$1/*

for d in $BATCH;
do
    echo THis is the name of the folder $d;
    STR=$2;
    python3 data_generator.py $d/CicloneNet_${d:34} 10.0 $STR/
    mv $STR/NEW_SLICE_10.0.csv $STR/${d:34}_NEW_SLICE_10.csv
    echo "Finished!"
done;

 


