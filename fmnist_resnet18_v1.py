#!/usr/bin/env python
# coding: utf-8

from utils.dpi_class import DPI_CLASS


lr_combinations = [
    # (1e-4, 5e-5, 1e-4),
    # (1e-4, 2e-4, 5e-5),
    # (5e-4, 5e-4, 5e-4),
    # (2e-4, 2e-4, 1e-4),
    (4e-4, 4e-4, 4e-4)
]

for lr_I, lr_G, lr_f in lr_combinations:
    model = DPI_CLASS(
        dataset_name='FashionMNIST',
        z_dim=5,
        lr_I=lr_I,
        lr_G=lr_G,
        lr_f=lr_f,
        weight_decay=0.01,
        batch_size=500,
        epochs1=50,
        epochs2=100,
        lambda_mmd=2.0,
        lambda_gp=0.1,
        lambda_power=.6,
        eta=2.5,
        std=0.5,
        present_label=[0,1,2,3,4,5,6,7,8],
        critic_iter=8,
        critic_iter_f=8,
        critic_iter_p=8,
        decay_epochs=40,
        gamma=0.2,
        balance=True,
        # timestamp='2025_01_29_1358',
    )
    model.train()
    # p_vals, p_sets = model.validate()
    model.validate_w_classifier()


# model.validate_w_classifier()

