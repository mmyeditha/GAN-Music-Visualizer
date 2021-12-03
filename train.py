from model import *
import time

train_batch_size = 32 #@param
noise_dimensions = 64 #@param
generator_lr = 0.001 #@param
discriminator_lr = 0.0002 #@param


steps_per_eval = 500 #@param
max_train_steps = 5000 #@param
batches_for_eval_metrics = 100 #@param

gan_estimator = tfgan.estimator.GANEstimator(
    generator_fn=unconditional_generator,
    discriminator_fn=unconditional_discriminator,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    params={'batch_size': train_batch_size, 'noise_dims': noise_dimensions},
    generator_optimizer=gen_opt,
    discriminator_optimizer=tf.train.AdamOptimizer(discriminator_lr, 0.5),
    get_eval_metric_ops_fn=get_eval_metric_ops_fn)

# demo training routine for mnist, have to adapt to my network/data
cur_step = 0
start_time = time.time()
while cur_step < max_train_steps:
    next_step = min(cur_step + steps_per_eval, max_train_steps)

    start = time.time()
    gan_estimator.train(input_fn, max_steps=next_step)
    steps_taken = next_step - cur_step
    time_taken = time.time() - start
    print('Time since start: %.2f min' % ((time.time() - start_time) / 60.0))
    print('Trained from step %i to %i in %.2f steps / sec' % (
        cur_step, next_step, steps_taken / time_taken))
    cur_step = next_step

    # add metrics of generator/discriminator loss as well