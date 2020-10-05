import tensorflow as tf


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=5000,
        seed=1234,
        dynamic_loss_scaling=True,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        data_parallel=False,
        # ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        song_meta_fname = '/home/svcapp/userdata/musicai/melon/song_meta.json',
        train_fname ='/home/svcapp/userdata/musicai/melon/arena_data/orig/train_token.json',
        question_fname = "/home/svcapp/userdata/musicai/melon/arena_data/questions/val_token.json",
        answer_fname = "/home/svcapp/userdata/musicai/melon/arena_data/answers/val.json",
        idx_dict_fname = "/home/svcapp/userdata/musicai/melon/index_dict.dat",
        mel_dir_path = "/home/svcapp/userdata/musicai/melon/arena_mel",
        flo_dir_path = "/home/svcapp/flo_ssd",
        # flo_dir_path = "/home/svcapp/userdata/musicai/flo_data/",
        artist_fname = "/home/svcapp/userdata/musicai/flo_data/artist_split.npy",
        # validation_files='/home/svcapp/userdata/saebyul_data/valid_list.txt',

        ################################
        # Model Parameters             #
        ################################
        # NeuMF parameters
        num_items = 576078,
        num_songs = 549729,
        num_tags = 25480,
        num_tokens = 2449,
        num_plylst = 115071,
        num_train_list = 92056,
        latent_dim_mf = 32,
        latent_dim_mlp = 32,
        neumf_layers = [64,64,32,32],
        l2_regularization = 0.01,
        use_all_items = False,
        # PropensityLoss parameters
        propensity_A = 0.2,
        propensity_B = 1.5,
        # Encoder parameters
        input_size = 707989,
        encoder_size = 32,
        middle_size = 16,
        latent_vec_size = 10,
        conv_size = 128,
        kernel_size = 5,
        out_size = 100,
        average_pool = True,
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,

        optimizer_type='adam',
        learning_rate=1e-3,
        weight_decay=1e-6,
        momentum = 0.9,

        grad_clip_thresh=1.0,
        num_workers = 2,
        batch_size = 64,
        valid_batch_size = 32,
        drop_out = 0.5,
        model_code='siamese_flo',
        pos_loss_weight = 1e4,
        num_neg_samples = 4,
        num_pos_samples = 16,
        pre_load_mel = False,
        in_meta = False
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
