#!/bin/bash -l

tune_lr() {
    eps=1e-8
    # lrs=(1.0 0.1 1e-2 1e-3 1e-4)
    lrs=(1e-2)
    steps=20000

    for lr in "${lrs[@]}"; do
        echo "running lr=$lr, eps=$eps"
        python main.py logging.wandb_project=simploss logging.wandb_name=minimax_madagrad_lr:"$lr"_eps:"$eps" \
            train.steps=$steps \
            optimizer=full_matrix_adagrad optimizer.schedule.lr=$lr optimizer.eps=$eps
    done
}

tune_eps() {
    lr=1e-2
    eps=(1e-4 1e-6 1e-8 1e-10)
    steps=20000

    for e in "${eps[@]}"; do
        echo "running lr=$lr, eps=$e"
        python main.py logging.wandb_project=simploss logging.wandb_name="eps:"$e"" \
            train.steps=$steps \
            optimizer=full_matrix_adagrad optimizer.schedule.lr=$lr optimizer.eps=$e
    done
}

valley_madagrad_lr() {
    eps=1e-8
    lrs=(1.0 0.1 1e-2 1e-3 1e-4)
    steps=20000

    for lr in "${lrs[@]}"; do
        echo "running lr=$lr, eps=$eps"
        python main.py logging.wandb_project=simploss logging.wandb_name=valley_madagrad_lr:"$lr"_eps:"$eps" \
            train.steps=$steps \
            optimizer=full_matrix_adagrad optimizer.schedule.lr=$lr optimizer.eps=$eps \
            loss=valley
    done
}

test() {
    python main.py logging.wandb_project=simploss logging.wandb_name=minimax_adam_lr:"$lr"_eps:"$eps" \
        train.steps=$steps \
        optimizer=adam optimizer.schedule.lr=0.01
}

# tune_lr
valley_madagrad_lr
# test