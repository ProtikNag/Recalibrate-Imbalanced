for ratio in 0.05 0.10 0.15 0.20 0.50; do
    python main_experiment.py \
        --experiment_name "imbalance_study_${ratio}" \
        --model_name resnet18 \
        --dataset_name CIFAR10 \
        --imbalance_classes "0,1" \
        --imbalance_ratio ${ratio} \
        --alpha 0.7 \
        --pretrain_epochs 30 \
        --recalib_epochs 10
done