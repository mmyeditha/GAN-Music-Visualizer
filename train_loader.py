import cv2
import os
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

IMG_PATH = 'PopArt/'

def item_generator(split='train'):
    for elem in os.listdir(IMG_PATH):
        img = preprocess(elem)

        yield preprocess(elem)

def preprocess(img):
    # open the image as numpy array and cast to tensor array
    img_arr = cv2.imread(os.path.join(IMG_PATH, img)).astype(np.float32)
    img_arr = (img_arr-127.5)/127.5
    return tf.convert_to_tensor(img_arr)

def input_fn(mode, params):
    assert 'batch_size' in params
    assert 'noise_dims' in params
    bs = params['batch_size']
    nd = params['noise_dims']
    split = 'train' if mode == tf.estimator.ModeKeys.TRAIN else 'test'
    shuffle = (mode == tf.estimator.ModeKeys.TRAIN)
    just_noise = (mode == tf.estimator.ModeKeys.PREDICT)
    
    noise_ds = (tf.data.Dataset.from_tensors(0).repeat()
                .map(lambda _: tf.random.normal([bs, nd])))
 
    if just_noise:
        return noise_ds

    images_ds = tf.data.Dataset.from_generator(item_generator, args=[split])
    if shuffle:
        images_ds = images_ds.shuffle(
            buffer_size=10000, reshuffle_each_iteration=True)
    images_ds = (images_ds.batch(bs, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

    return tf.data.Dataset.zip((noise_ds, images_ds))