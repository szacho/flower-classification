import tensorflow as tf

KAGGLE_DS_PATH = "D:\\Data Science\\Datasets\\flowers\\tfrec\\"
OXFORD_DS_PATH = "D:\\Data Science\\Datasets\\flowers\\external\\oxford\\oxford_tfrec\\"

IMAGE_SIZE = [224, 224]
EPOCHS = 20
get_batch_size = lambda strategy: 16 * strategy.num_replicas_in_sync

PATH_SELECT = { # available image sizes
    192: 'tfrecords-jpeg-192x192',
    224: 'tfrecords-jpeg-224x224',
    331: 'tfrecords-jpeg-331x331',
    512: 'tfrecords-jpeg-512x512'
}

TFREC_DIR = PATH_SELECT[IMAGE_SIZE[0]]

TRAINING_FILENAMES = tf.io.gfile.glob(f'{KAGGLE_DS_PATH}{TFREC_DIR}\\train\\*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(f'{KAGGLE_DS_PATH}{TFREC_DIR}\\val\\*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(f'{KAGGLE_DS_PATH}{TFREC_DIR}\\test\\*.tfrec') # predictions on this dataset should be submitted for the competition
OXFORD_FILENAMES = tf.io.gfile.glob(f'{OXFORD_DS_PATH}{TFREC_DIR}\\*.tfrec')

CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102