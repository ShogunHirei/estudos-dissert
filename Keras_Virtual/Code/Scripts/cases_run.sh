# Scripts para obter amostras dos casos de redes neurais
# Para treinamento de número de pontos

# Definindo a quantidade de pontos da simulação
POINTS=3;

# Definindo tamanho do passo
STEP=$(python3 -c "print((21.0 - 6.0)/$POINTS)");

# Primeiro caso que será carregado 
ATUAL=$1
# Considerando que o primeiro case está em diretório reservado para guardar as
# todos os casos
# -> Simulation/
#    |____$CASO1/
#         |____ ...
#    |____$CASO2/
#         |____ ...
#    |____$CASO3/
#         |____ ...

# Script do Paraview para extração dos dados
PVPYTHON=$2

# Diretório aonde serão salvos os dados 
SAVE_DIR=$3


for (( IT=0; IT < $POINTS; IT++))
    do
        echo "This is the point number $IT"
        # Primeiro passo é determina a velocidade a se alterar
        # Considerando que ATUAL é um diretório
        VEL=${ATUAL:13}
        VEL=$(python3 -c "print($VEL + $STEP)")
        echo "The new vel is $VEL"
        # Considerando que esse script será executado no diretório Simulations
        # ou seja cwd=Simulations/
        NEW=ciclone_ref2_$VEL
        # Diretório da nova simulação
        mkdir ./$NEW
        # Copiando dados 
        cp -r ./$ATUAL/* ./$NEW/
        # Usar o último tempo da simulação anterior como atual
        MV_DIR=$(ls ./$NEW/ | grep "^[1-9]" | sort -n | tail -n 1)
        #echo "The greater folder is $MV_DIR"
        #echo "============================================================"
        #echo "Inside Origin folder $NEW/0" && ls ./$NEW/0
        #echo "============================================================"
        rm -r ./$NEW/0
        mv ./$NEW/$MV_DIR/ ./$NEW/0
        # Apagando outros possíveis passos de tempo
        #echo "============================================================"
        #echo "$MV_DIR is now $NEW/0" && ls ./$NEW/0
        #echo "============================================================"
        for TIME in $(ls ./$NEW/ | grep "^[1-9]")
        do
            echo "say goodbye to-> $TIME"
            rm -r ./$NEW/$TIME
        done
        # Alterando os parametros da simulação
        sed -i "s/\(uniform\) -[[:digit:]]\.\?[[:digit:]]/\1 -$VEL/g" ./$NEW/0/U
        # O script usa caminho relativo , por isso mudar para o diretório de execução
        cd ./$NEW
        ./script.sh 
        cd ../
        # Executar script do paraview para obter dados
        pvpython $PVPYTHON ./$NEW/mesh_coarse.OpenFOAM $VEL $SAVE_DIR 
        ATUAL=$NEW

    done




