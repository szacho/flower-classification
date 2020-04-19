import tensorflow as tf, tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import time
from collections import namedtuple
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

# TRAINING LOOP (based on this notebook https://www.kaggle.com/mgornergoogle/custom-training-loop-with-100-flowers-on-tpu)
start_time = epoch_start_time = time.time()
# distribute the dataset according to the strategy

train_dist_ds = strategy.experimental_distribute_dataset(get_training_dataset(TRAINING_FILENAMES, BATCH_SIZE))
valid_dist_ds = strategy.experimental_distribute_dataset(get_validation_dataset(VALIDATION_FILENAMES, BATCH_SIZE, repeated=True))

print("Training steps per epoch:", STEPS_PER_EPOCH, "in increments of", STEPS_PER_TPU_CALL)
print("Validation images:", NUM_VALIDATION_IMAGES,
      "Batch size:", BATCH_SIZE,
      "Validation steps:", NUM_VALIDATION_IMAGES//BATCH_SIZE, "in increments of", VALIDATION_STEPS_PER_TPU_CALL)
print("Repeated validation images:", int_div_round_up(NUM_VALIDATION_IMAGES, BATCH_SIZE*VALIDATION_STEPS_PER_TPU_CALL)*VALIDATION_STEPS_PER_TPU_CALL*BATCH_SIZE-NUM_VALIDATION_IMAGES)
History = namedtuple('History', 'history')
history = History(history={'loss': [], 'val_loss': [], 'sparse_categorical_accuracy': [], 'val_sparse_categorical_accuracy': []})

epoch = 0
train_data_iter = iter(train_dist_ds) 
valid_data_iter = iter(valid_dist_ds)

step = 0
epoch_steps = 0
while True:
    
    # run training step
    train_step(train_data_iter)
    epoch_steps += STEPS_PER_TPU_CALL
    step += STEPS_PER_TPU_CALL
    print('=', end='', flush=True)

    # validation run at the end of each epoch
    if (step // STEPS_PER_EPOCH) > epoch:
        print('|', end='', flush=True)
        
        # validation run
        valid_epoch_steps = 0
        for _ in range(int_div_round_up(NUM_VALIDATION_IMAGES, BATCH_SIZE*VALIDATION_STEPS_PER_TPU_CALL)):
            valid_step(valid_data_iter)
            valid_epoch_steps += VALIDATION_STEPS_PER_TPU_CALL
            print('=', end='', flush=True)

        # compute metrics
        history.history['sparse_categorical_accuracy'].append(train_accuracy.result().numpy())
        history.history['val_sparse_categorical_accuracy'].append(valid_accuracy.result().numpy())
        history.history['loss'].append(train_loss.result().numpy() / (BATCH_SIZE*epoch_steps))
        history.history['val_loss'].append(valid_loss.result().numpy() / (BATCH_SIZE*valid_epoch_steps))
        
        # report metrics
        epoch_time = time.time() - epoch_start_time
        print('\nEPOCH {:d}/{:d}'.format(epoch+1, EPOCHS))
        print('time: {:0.1f}s'.format(epoch_time),
              'loss: {:0.4f}'.format(history.history['loss'][-1]),
              'accuracy: {:0.4f}'.format(history.history['sparse_categorical_accuracy'][-1]),
              'val_loss: {:0.4f}'.format(history.history['val_loss'][-1]),
              'val_acc: {:0.4f}'.format(history.history['val_sparse_categorical_accuracy'][-1]),
              'lr: {:0.4g}'.format(lrfn(epoch)),
              'steps/val_steps: {:d}/{:d}'.format(epoch_steps, valid_epoch_steps), flush=True)
        
        # set up next epoch
        epoch = step // STEPS_PER_EPOCH
        epoch_steps = 0
        epoch_start_time = time.time()
        train_accuracy.reset_states()
        valid_accuracy.reset_states()
        valid_loss.reset_states()
        train_loss.reset_states()
        if epoch >= EPOCHS:
            break

optimized_ctl_training_time = time.time() - start_time
print("OPTIMIZED CTL TRAINING TIME: {:0.1f}s".format(optimized_ctl_training_time))

model.save('flower_model.h5')