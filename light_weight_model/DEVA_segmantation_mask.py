import os
import matplotlib.pyplot as plt
import random
import cv2
    
from argparse import ArgumentParser

import torch
import numpy as np
os.chdir('/root/maaf/Tracking_Anything_with_DEVA')

def get_segmentation_mask(frame):
    """
    Returns the segmentation mask using DEVA.

    Args:
        frame: The input frame for segmentation.

    Returns:
        The segmentation mask.

    Raises:
        ImportError: If the required modules are not found.
    """

    try:
        import groundingdino
        from groundingdino.util.inference import Model as GroundingDINOModel
    except ImportError:
        import GroundingDINO
        from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel

    # Rest of the code...
def get_segmentation_mask(frame):

    try:
        import groundingdino
        from groundingdino.util.inference import Model as GroundingDINOModel
    except ImportError:
        import GroundingDINO
        from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel



    from deva.model.network import DEVA
    from deva.inference.inference_core import DEVAInferenceCore
    from deva.inference.result_utils import ResultSaver
    from deva.inference.eval_args import add_common_eval_args, get_model_and_config
    from deva.inference.demo_utils import flush_buffer
    from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
    from deva.ext.grounding_dino import get_grounding_dino_model
    from deva.ext.with_text_processor import process_frame_with_text as process_frame


    torch.autograd.set_grad_enabled(False)
    CLASSES = ['black sticks', 'hands', 'white plastics','rectangular black object'] ## detecting hands has higher acc with person, small plastic is hard to detect

    # for id2rgb
    random.seed(42)



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
    deva_model = DEVA(cfg).cuda().eval()
    if args.model is not None:
        model_weights = torch.load(args.model)
        deva_model.load_weights(model_weights)
    else:
        print('No model loaded.')

    gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')


    cfg['enable_long_term_count_usage'] = True
    cfg['max_num_objects'] = 50
    cfg['size'] = 480
    cfg['DINO_THRESHOLD'] = 0.35
    cfg['amp'] = True
    cfg['chunk_size'] = 4
    cfg['detection_every'] = 1
    cfg['max_missed_detection_count'] = 10
    cfg['sam_variant'] = 'original'
    cfg['temporal_setting'] = 'online' # semionline usually works better; but online is faster for this demo
    cfg['pluralize'] = True

    cfg['DINO_THRESHOLD'] = 0.25


    from deva.ext.with_text_processor import process_frame_with_text as process_frame_text
    import cv2

    cfg['prompt'] = '.'.join(CLASSES)

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()

    # obtain temporary directory
    result_saver = ResultSaver(None, None, dataset='gradio', object_manager=deva.object_manager)
    ti = 0
    with torch.cuda.amp.autocast(enabled=cfg['amp']):
            mask, segments_info = process_frame_text(deva,
                                gd_model,
                                sam_model,
                                'null.png',
                                result_saver,
                                ti,
                                image_np=frame)

            flush_buffer(deva, result_saver)
    deva.clear_buffer()
    draw_mask = mask.cpu().numpy()
    draw_mask[draw_mask>1]=1
    return draw_mask


if __name__ == "__main__":

    view_name = 'view_2'
    # Specify the path to the video file
    video_path = '/root/maaf/dataset/seq_4_seq_4_view_2.webm'

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the frame in the first subplot
    axs[0].imshow(frame)
    axs[0].set_title('Frame')

    # Plot the mask in the second subplot
    mask = get_segmentation_mask(frame)
    draw_mask = mask.cpu().numpy()
    draw_mask[draw_mask>1]=1
    axs[1].imshow(draw_mask,cmap='gray')
    axs[1].set_title('Mask')

    # Display the subplots
    plt.savefig('/root/maaf/mask.png')
    plt.close()
