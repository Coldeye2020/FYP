#! /bin/sh
#
# run_all_model_dense_split_demo.sh

# for dataset in cora citeseer pubmed chameleon cornell film squirrel texas wisconsin new_squirrel new_chameleon
# do    
#     for model in APPNP GAT GCN MLP JKNet
#     do
#         for lr in 0.002 0.01 0.05
#         do
#             for weight_decay in 0 0.0005
#             do
#                 python model.py --model $model \
#                     --dataset $dataset \
#                     --lr $lr \
#                     --weight_decay $weight_decay \
#                     --epoch 1000
#             done
#         done
#     done
# done

# for dataset in cora citeseer pubmed chameleon cornell film squirrel texas wisconsin new_squirrel new_chameleon
# do
#     for passing_type in Attention # Origin
#     do
#         for lr in 0.002 0.01 0.05
#         do 
#             for weight_decay in 0 0.0005
#             do
#                 python model.py --model GPRGNN \
#                     --dataset $dataset \
#                     --lr $lr \
#                     --passing_type $passing_type \
#                     --weight_decay  $weight_decay\
#                     --epoch 1000
#             done
#         done
#     done
# done










id=0
trian_ratio=0.025
val_ratio=0.025

# for dataset in cora citeseer pubmed computers photo
# do
#     for passing_type in Attention # Origin
#     do
#         for lr in 0.002 0.01 0.02 0.05
#         do 
#             for weight_decay in 0 0.0005
#             do
#                 for hidden in 64 128
#                 do
#                     for dprate in 0.3 0.5 0.7
#                     do
#                         for alpha in 0.1 0.2 0.5 0.9 1.0
#                         do
#                             echo "part1: $id"
#                             python model.py --model GPRGNN \
#                                 --dataset $dataset \
#                                 --lr $lr \
#                                 --passing_type $passing_type \
#                                 --weight_decay  $weight_decay \
#                                 --hidden $hidden \
#                                 --epoch 500 \
#                                 --dprate $dprate \
#                                 --alpha $alpha \
#                                 --id $id
#                                 --trian_ratio $trian_ratio
#                                 --val_ratio $val_ratio
#                             let id++
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# id=1200

for dataset in cora citeseer pubmed computers photo squirrel chameleon film texas cornell
do
    for model in GPRGNN APPNP GAT GCN MLP JKNet SGC GCN-Cheby SAGE
    do
        if [ $model = "GPRGNN" ]
        then
            for passing_type in Attention Origin
            do
                for lr in 0.002 0.01 0.05
                do 
                    for hidden in 64 128
                    do
                        for dprate in 0.5 0.7
                        do
                            for alpha in 0.1 0.5 0.9
                            do
                                echo "part1: $id"
                                python model.py --model GPRGNN \
                                    --dataset $dataset \
                                    --lr $lr \
                                    --passing_type $passing_type \
                                    --weight_decay  0.0005 \
                                    --hidden $hidden \
                                    --epoch 10 \
                                    --dprate $dprate \
                                    --alpha $alpha \
                                    --id $id \
                                    --train_ratio $trian_ratio \
                                    --val_ratio $val_ratio
                                id=`expr $id + 1`
                            done
                        done
                    done
                done
            done
        
        else
            for lr in 0.002 0.01 0.05
            do 
                for hidden in 64 128
                do
                    echo "part2: $id"
                    python model.py --model $model \
                        --dataset $dataset \
                        --lr $lr \
                        --weight_decay  0.0005 \
                        --hidden $hidden \
                        --epoch 10 \
                        --id $id \
                        --train_ratio $trian_ratio \
                        --val_ratio $val_ratio
                    id=`expr $id + 1`
                done
            done
        fi
    done
done
