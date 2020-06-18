# Script para repetição das simulações e extração dos dados em simulações

# Definições: Velocidades Iniciais e qntdad de pontos
FIRST_VEL=6;
MAX_IT=50;
vel=6; # Serão inseridos os caminhos para os scripts que serão utilizados nos códigos
# 1 -> Script para mudar de Velocidade
# 2 -> Pasta de armazenamento das simulações
# 3 -> Script de extração
# 4 -> Posição do caso inicial para usar como exemplo

for (( count=0; count<=MAX_IT; count++ ));
do
    echo $vel;
    mkdir $2/analise_ciclone_U_$vel; # Criando pasta para a análise 
    cp $4/* $2/analise_ciclone_U_$vel/ -r; # Copiando caso de referência para pasta atual 
    python3 $1 $2/analise_ciclone_U_$vel/ $vel;     # Script para mudar de velocidade
    decomposePar -case $2/analise_ciclone_U_$vel/ ;
    mpirun -np 4 simpleFoam -parallel -case $2/analise_ciclone_U_$vel/ ; 
    reconstructPar -case $2/analise_ciclone_U_$vel/ ;
    pvpython $3 $2/analise_ciclone_U_$vel/mesh_coarse.OpenFOAM $vel; # Script Paraview
    vel=$( bc <<<"scale=2; $vel + 0.3" );
done;

    
