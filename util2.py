from pytorch_models import *


def create_model(network_type: str, channels: int, device: str = "cpu", evaluation=False, **kwargs):
    """ Creates the model network

    Parameters
    ----------
    network_type : str
        type of the network. It must match one of the available networks (case-insensitive) otherwise raises KeyError
    channels : int
        number of channels of the data the network will handle
    device : str
        the device onto which train the network (either cpu or a cuda visible device)
    evaluation : bool
        True if the networks must be defined for evaluation False otherwise.
        If True, definition doesn't need loss and optimizer
    """
    network_type = network_type.strip().upper()
    try:
        model = GANS[network_type].value(channels, device)

        if not evaluation:
            # Adversarial Loss Definition
            adv_loss = None if kwargs['adv_loss_fn'] is None else kwargs['adv_loss_fn'].strip().upper()
            if adv_loss is not None:
                adv_loss = AdvLosses[adv_loss].value()

            # Reconstruction Loss Definition
            rec_loss = None if kwargs['loss_fn'] is None else kwargs['loss_fn'].strip().upper()
            if rec_loss is not None:
                print(rec_loss)
                rec_loss = Losses[rec_loss].value()

            model.define_losses(rec_loss=rec_loss, adv_loss=adv_loss)
        return model
    except KeyError:
        pass

    try:
        model = CNNS[network_type].value(channels, device)

        if not evaluation:
            # Reconstruction Loss Definition
            loss_fn = None if kwargs['loss_fn'] is None else kwargs['loss_fn'].strip().upper()
            if loss_fn is not None:
                loss_fn = Losses[loss_fn].value()

            # Optimizer definition
            optimizer = None if kwargs['optimizer'] is None else kwargs['optimizer'].strip().upper()
            if optimizer is not None:
                optimizer = Optimizers[optimizer].value(model.parameters(), lr=kwargs['lr'])

            model.compile(loss_fn, optimizer)
        else:
            model.compile()

        return model
    except KeyError:
        pass

    raise KeyError(f"Model not defined!\n\n"
                   f"Use one of the followings:\n"
                   f"GANS: \t{[e.name for e in GANS]}\n"
                   f"CNN: \t{[e.name for e in CNNS]}\n"
                   f"Losses: \t{[e.name for e in Losses]}\n"
                   f"Adversarial Losses: \t{[e.name for e in AdvLosses]}\n"
                   f"Optimizers: \t{[e.name for e in Optimizers]}\n"
                   )


if __name__ == '__main__':
    from tbparse import SummaryReader

    fold = "pytorch_models/trained_models/W3/APNN/apnn_v9.3/"
    log_dir = fold + "log"
    reader = SummaryReader(log_dir)
    df = reader.scalars

    q2n_df = df[df['tag'] == "Q2n/Val"]
    q2n_df.to_csv(fold + 'Q2n.csv')
    Q_avg_df = df[df['tag'] == "Q/Val"]
    Q_avg_df.to_csv(fold + 'Q_avg.csv')
    SAM_df = df[df['tag'] == "SAM/Val"]
    SAM_df.to_csv(fold + 'SAM.csv')
    ERGAS_df = df[df['tag'] == "ERGAS/Val"]
    ERGAS_df.to_csv(fold + 'ERGAS.csv')

    loss_df = df[df['tag'] == "Loss"][::2]
    loss_df.to_csv(fold + 'loss_train.csv')

    loss_df = df[df['tag'] == "Loss"][1::2]
    loss_df.to_csv(fold + 'loss_val.csv')
