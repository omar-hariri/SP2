import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.regularizers import l1_l2

def build_lstm_model(input_shape, cfg):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    reg = l1_l2(
        l1=cfg.get("l1", 1e-5),
        l2=cfg.get("l2", 1e-4)
    )

    # First LSTM block
    model.add(Bidirectional(LSTM(
        cfg["lstm_units_1"],
        return_sequences=True,
        kernel_regularizer=reg,
        recurrent_dropout=cfg.get("recurrent_dropout", 0.0)
    )))
    model.add(Dropout(cfg["dropout"]))

    # Second LSTM block
    model.add(LSTM(
        cfg["lstm_units_2"],
        return_sequences=False,
        kernel_regularizer=reg,
        recurrent_dropout=cfg.get("recurrent_dropout", 0.0)
    ))

    # Dense block
    model.add(Dense(
        cfg["dense_units"],
        activation="relu",
        kernel_regularizer=reg
    ))
    model.add(Dropout(cfg["dropout"]))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.004)
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model