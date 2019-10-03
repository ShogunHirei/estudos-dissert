#! /usr/bash
# Script para repetição das simulações e extração dos dados em simulações

# Definições: Velocidades Iniciais e qntdad de pontos
FIRST_VEL=5
LAST_VEL=130
STEP_SIZE=3
ITERATION=0

# Serão inseridos os caminhos para os scripts que serão utilizados nos códigos
# 1 -> Script para mudar de Velocidade
# 2 -> Solver do OpenFoam
# 3 -> Script de extração
# 4 -> Posição do caso inicial para usar como exemplo

for (( count=FIRST_VEL; count<=LAST_VEL; count=count+STEP_SIZE ));
do
    mkdir analise_ciclone_U_$count; # Criando pasta para a análise 
    cd ./analise_ciclone_U_$count;
    cp $4/* ./ -r; # Copiando caso de referência para pasta atual 
    python3 $1 $count;     # Script para mudar de velocidade
    $2;
    pvpython $3 ./  $count; # Script Paraview
    cd ../
done;

    
