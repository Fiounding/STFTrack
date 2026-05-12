import os
import sys
import time

import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm

from lib.vis.visdom_cus import Visdom

env_path = os.path.join(os.path.dirname(__file__), '../lib')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import trackerlist, get_dataset
from lib.test.utils.load_text import load_text


class VisResults(object):
    def __init__(self):
        self._init_visdom()

    def vis_dataset(self, dataset, trackers, skip_missing_seq=False, seq_list=[]):
        for seq_id, seq in enumerate(tqdm(dataset)):
            # Load anno
            seq_name = seq.name
            if seq_list:
                if seq_name not in seq_list:
                    continue

            anno_bb = torch.tensor(seq.ground_truth_rect)
            target_visible = torch.tensor(seq.target_visible,
                                          dtype=torch.uint8) if seq.target_visible is not None else None

            all_pred_boxes = []

            for trk_id, trk in enumerate(trackers):
                # Load results
                base_results_path = '{}/{}'.format(trk.results_dir, seq.name)
                results_path = '{}.txt'.format(base_results_path)

                if os.path.isfile(results_path):
                    pred_bb = torch.tensor(load_text(str(results_path), delimiter=('\t', ','), dtype=np.float64))
                    all_pred_boxes.append(pred_bb)
                else:
                    if skip_missing_seq:
                        break
                    else:
                        raise Exception('Result not found. {}'.format(results_path))

            frame_list = seq.frames
            for i in range(len(anno_bb)):
                data = []
                frame = frame_list[i]
                im = cv.imread(frame)
                im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
                # im = torch.from_numpy(im).float().permute(2, 0, 1)
                # im = im.numpy()
                data.append(im)

                gt_box = anno_bb[i]
                data.append(gt_box)
                for tracker_result in all_pred_boxes:
                    data.append(tracker_result[i])

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

                if self.next_seq:
                    self.next_seq = False
                    break

                self.update_boxes(data, seq_name + '-' + str(i).zfill(3))
                # self.update_seg_result(im, frame)

    def update_boxes(self, data, caption):
        caption = 'Green: GT, Red: stark_s, Yellow: stark_motion  _' + caption
        self.visdom.register(data, 'Tracking', 1, 'Tracking', caption=caption)

    def update_seg_result(self, frame_img, frame_path):
        seg_mask_path = os.path.join(os.path.dirname(frame_path), 'seg_mask',
                                     os.path.basename(frame_path).replace('jpg', 'png'))
        seg_mask = cv.imread(seg_mask_path)
        alpha = 0.5
        out_img = (alpha * frame_img) + ((1 - alpha) * seg_mask)

        if max(out_img.shape) > 480:
            resize_factor = 480.0 / float(max(out_img.shape))
            out_img = cv.resize(out_img, None, fx=resize_factor, fy=resize_factor)

        out_img = torch.from_numpy(out_img).float().permute(2, 0, 1)
        self.visdom.register(out_img, 'image', 1, 'Segmentation Result')

    def _init_visdom(self, visdom_info=None):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        self.next_seq = False

        try:
            self.visdom = Visdom(1, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                 visdom_info=visdom_info, env='vis_results')

            # Show help
            help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                        'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                        'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                        'block list.'
            self.visdom.register(help_text, 'text', 1, 'Help')
        except:
            time.sleep(0.5)
            print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                  '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'n':
                self.next_seq = True

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True


if __name__ == '__main__':
    viser = VisResults()
    dataset_name = 'lasot'
    trackers = []
    # trackers.extend(trackerlist())
    dataset = get_dataset(dataset_name)
    trackers.extend(trackerlist('dimp', 'dimp50', None, 'DiMP50'))
    trackers.extend(trackerlist('sa', 'attn_direct', None, 'SA'))
    # dataset = get_dataset('vot')
    # 'GOT-10k_Train_001350'
    viser.vis_dataset(dataset, trackers, seq_list=[])
    # viser.vis_dataset(dataset, trackers, seq_list=['GOT-10k_Train_007446'])
