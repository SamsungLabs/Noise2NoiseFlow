"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""
import numpy as np

def calc_train_test_stats(end_tr_im_idx, start_tr_im_idx, end_ts_im_idx, start_ts_im_idx, n_tr_inst, n_ts_inst, n_patches_per_image, n_batch_train, n_batch_test):
    n_train_per_scene = end_tr_im_idx - start_tr_im_idx
    n_test_per_scene = end_ts_im_idx - start_ts_im_idx
    n_train = n_tr_inst * n_train_per_scene * n_patches_per_image
    n_test = n_ts_inst * n_test_per_scene * n_patches_per_image
    n_tr_bat_per_seq = int(np.ceil(n_tr_inst * n_patches_per_image / n_batch_train))
    n_ts_bat_per_seq = int(np.ceil(n_ts_inst * n_patches_per_image / n_batch_test))

    return n_train_per_scene, n_test_per_scene, n_train, n_test, n_tr_bat_per_seq, n_ts_bat_per_seq

def get_its(n_batch_train, n_batch_test, n_train, n_test):
    train_its = int(np.ceil(n_train / n_batch_train))
    test_its = int(np.ceil(n_test / n_batch_test))
    train_epoch = train_its * n_batch_train
    logging.info("Train epoch size: {}".format(train_epoch))
    return train_its, test_its

# --- Logger ---
class ResultLogger(object):
    def __init__(self, path, columns, append=False):
        self.columns = columns
        mode = 'a' if append else 'w'
        self.f_log = open(path, mode)
        # self.f_log.write(json.dumps(kwargs) + '\n')
        if mode == 'w':
            self.f_log.write("\t".join(self.columns))

    def __del__(self):
        self.f_log.close()

    def log(self, run_info):
        run_strings = ["{0}".format(run_info[lc]) for lc in self.columns]
        self.f_log.write("\n")
        self.f_log.write("\t".join(run_strings))
        # self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()

def hps_logger(path, hps, layer_names, num_params):
    import csv
    # print(hps)
    with open(path, 'w') as f:
        w = csv.writer(f)
        for n in layer_names:
            w.writerow([n])
        w.writerow([num_params])
        for k, v in vars(hps).items():
            w.writerow([k, v])

def hps_loader(path):
    import csv

    class Hps:
        pass

    hps = Hps()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for pair in reader:
            if len(pair) < 2:
                continue
            val = pair[1]
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except:
                    if val == 'True':
                        val = True
                    elif val == 'False':
                        val = False
            hps.__setattr__(pair[0], val)
    return hps