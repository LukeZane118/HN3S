#!/bin/bash
dns=(ml-1m steam citeulike-a)
lrs=(1.e-3 5.e-3 5.e-3)
taus=(3 1 20)
for seed in 2022 2023 2024
do
    for mn in multi_vae
    do
        for i in 0 1 2
        do
            dn=${dns[i]}
            lr=${lrs[i]}
            tau=${taus[i]}

            python ${mn}/main.py -dn ${dn} -ln [Baseline][seed_${seed}] --lr ${lr} --seed ${seed} -ueg
            python ${mn}/main.py -dn ${dn} -ln [HN3S-\(3,1,${tau}\)][seed_${seed}] -pm HN3S --c 3 --rho 1 --tau ${tau} --lr ${lr} --seed ${seed}
        done
    done
done