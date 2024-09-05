from utils.dpi_class import DPI_CLASS

model = DPI_CLASS(
    dataset_name='FashionMNIST',
    z_dim=5,
    lr_I=1e-5,
    lr_G=1e-5,
    lr_D=1e-5,
    weight_decay=0.01,
    batch_size=500,
    epochs1=50,
    epochs2=180,
    lambda_mmd=5.0,
    lambda_gp=0.1,
    lambda_power=0.5,
    eta=3.0,
    std=1.0,
    present_label=[0,1,2,3,4],
    img_size=28,
    nc=1,
    critic_iter=2,
    critic_iter_d=2,
    critic_iter_p=2,
    decay_epochs=100,
    gamma=0.2,
)

model.train()
model.validate()