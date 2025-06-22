from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def improved_cnn(input_len=6000, input_width=3, num_classes=7):
    input_seq = Input(shape=(input_len, input_width))

    x = Conv1D(64, kernel_size=7, activation='relu', padding='same')(input_seq)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)

    x = Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)

    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_seq, outputs=output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    return model
