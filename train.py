import tensorflow as tf, tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tpu_init import initialize_tpu
from read_tfrec import count_data_items, int_div_round_up, get_training_dataset, get_validation_dataset
from config import TRAINING_FILENAMES, VALIDATION_FILENAMES, TEST_FILENAMES, OXFORD_FILENAMES, get_batch_size, EPOCHS, IMAGE_SIZE, CLASSES

strategy = initialize_tpu()
BATCH_SIZE = get_batch_size(strategy)

# LEARNING RATE
# Learning rate schedule for TPU, GPU and CPU.
# Using an LR ramp up because fine-tuning a pre-trained model.
# Starting with a high LR would break the pre-trained weights.

LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8

@tf.function
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

def plot_learning_rate():  
    rng = [i for i in range(EPOCHS)]
    y = [lrfn(x) for x in rng]
    plt.plot(rng, y)
    print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

# SOME CONSTANTS
NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALIDATION_STEPS = int_div_round_up(NUM_VALIDATION_IMAGES, BATCH_SIZE)
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

# JENSEN-SHANNON LOSS
with strategy.scope():
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    kl_div = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

    def batchmean(loss_obj, size):
        return tf.reduce_sum(loss_obj) * (1. / size)

    def jsd(model, p_org, p_aug1, p_aug2, labels):

        p_mixture = K.log(tf.clip_by_value((p_org + p_aug1 + p_aug2) / 3., 1e-7, 1))
        k1 = batchmean(kl_div(p_mixture, p_org), BATCH_SIZE)
        k2 = batchmean(kl_div(p_mixture, p_aug1), BATCH_SIZE)
        k3 = batchmean(kl_div(p_mixture, p_aug2), BATCH_SIZE)

        def loss_func(y_true, y_pred):
            loss = batchmean(cross_entropy(y_true, y_pred), BATCH_SIZE)
            loss += 12 * ( k1 + k2 + k3 ) / 3.
            return loss

        return loss_func(y_true=labels, y_pred=p_org)

# STEP FUNCTIONS
STEPS_PER_TPU_CALL = 99
VALIDATION_STEPS_PER_TPU_CALL = 29

def train_step(data_iter):
    @tf.function
    def train_step_fn(images, augmix1, augmix2, labels):
        with tf.GradientTape() as tape:
            probabilities = model(images, training=True)
            p_aug1 = model(augmix1, training=True)
            p_aug2 = model(augmix2, training=True)
            loss = jsd(model, probabilities, p_aug1, p_aug2, labels)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #update metrics
        train_accuracy.update_state(labels, probabilities)
        train_loss.update_state(loss)

    # this loop runs on the TPU
    for _ in tf.range(STEPS_PER_TPU_CALL):
        strategy.experimental_run_v2(train_step_fn, next(data_iter))


def valid_step(data_iter):
    @tf.function
    def valid_step_fn(images, labels):
        probabilities = model(images, training=False)
        loss = loss_fn(labels, probabilities)

        # update metrics
        valid_accuracy.update_state(labels, probabilities)
        valid_loss.update_state(loss)

    # this loop runs on the TPU
    for _ in tf.range(VALIDATION_STEPS_PER_TPU_CALL):
        strategy.experimental_run_v2(valid_step_fn, next(data_iter))

# LOAD PRETRAINED MODEL
with strategy.scope():
    net = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        weights='imagenet',
        include_top=False
    )
    net.trainable = True

    model = tf.keras.Sequential([
        net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.summary()
    
    # Instiate optimizer with learning rate schedule
    class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            return lrfn(epoch=step//STEPS_PER_EPOCH)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LRSchedule())
    
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    train_loss = tf.keras.metrics.Sum()
    valid_loss = tf.keras.metrics.Sum()
    
    loss_fn = tf.keras.losses.sparse_categorical_crossentropy