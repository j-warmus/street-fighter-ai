from stable_baselines.common.policies import LstmPolicy


class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[256, 'lstm', dict(vf=[128, 128], pi=[128,128])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)