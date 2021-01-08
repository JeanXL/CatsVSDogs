from easydict import EasyDict as edict
import datetime


cfg = edict()

cfg.epochs = 30
cfg.batch_size = 32
cfg.save_mode_path = "cats_vs_dogs.h5"
cfg.log_dir = ".\\logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 此处有bug用\\ 而不能用/


# training options
cfg.train = edict()
cfg.train.num_samples = 23000
cfg.train.learning_rate = 1e-3
cfg.train.dataset = "./train.tfrecords"

# training options
cfg.val = edict()
cfg.val.num_samples = 2000
cfg.val.dataset = "./val.tfrecords"


# num_repeat = cfg.train.num_samples // cfg.batch_size
# print(num_repeat)

