import os
from os import path
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np

from deva.model.network import DEVA
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import process_frame_with_text as process_frame_text
from deva.ext.automatic_processor import process_frame_automatic as process_frame_auto

from tqdm import tqdm
import json

def get_segmentations_masks(images_dir, masks_dir):
    # disable gradient computation as we're only running inference
    torch.autograd.set_grad_enabled(False)

    # for id2rgb
    np.random.seed(42)

    os.chdir('/root/maaf/Tracking_Anything_with_DEVA')

    # default parameters
    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)

    # load model and config
    args = parser.parse_args([])
    cfg = vars(args)
    cfg['enable_long_term'] = True

    # Load our checkpoint
    deva_model = DEVA(cfg).cuda(device = 'cuda:1').eval()
    if args.model is not None:
        model_weights = torch.load(args.model, map_location='cuda:1')
        deva_model.load_weights(model_weights)
    else:
        print('No model loaded.')

    gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda:1')

    #cfg['img_path'] = '/root/maaf_umer/Data/seq_1/seq_1_view_1'
    cfg['img_path'] = images_dir
    cfg['output'] = masks_dir
    cfg['enable_long_term_count_usage'] = True
    cfg['max_num_objects'] = 50
    cfg['size'] = 480
    cfg['DINO_THRESHOLD'] = 0.25
    cfg['amp'] = True
    cfg['chunk_size'] = 4
    cfg['detection_every'] = 10
    cfg['max_missed_detection_count'] = 10
    cfg['sam_variant'] = 'original'
    cfg['temporal_setting'] = 'semionline' # semionline usually works better than online
    cfg['pluralize'] = True

    CLASSES = ['glossy black sticks', 'hands', 'white plastic pieces','rectangular black object', 'very small black star piece'] ## detecting hands has higher acc with person, small plastic is hard to detect
    cfg['prompt'] = '.'.join(CLASSES)

    # get data
    video_reader = SimpleVideoReader(cfg['img_path']) # need to modify its _get_item_ for any image preprocessing
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
    out_path = cfg['output']

    vid_length = len(loader)

    print('Configuration:', cfg)


    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id() # without this, it doesn't save visualizations
    result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)

    torch.cuda.set_device('cuda:1')
    with torch.cuda.amp.autocast(enabled=cfg['amp']):
        for ti, (frame, im_path) in enumerate(tqdm(loader)):
            process_frame_text(deva, gd_model, sam_model, im_path, result_saver, ti, image_np=frame)
        flush_buffer(deva, result_saver)
    result_saver.end()

        # save this as a video-level json
    with open(os.path.join(masks_dir,'segmented.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)  # prettier json

if __name__ == "__main__":
    images_dir = '/root/maaf_umer/Data_Comb'
    masks_dir = '/root/maaf_umer/Data_Comb_Seg'
    get_segmentations_masks(images_dir, masks_dir)