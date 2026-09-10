"""Copy into local/. Connect existing code; do not guess model normalization."""
from measurement_pipeline import visible_preprocess


def preprocess(frame_bgr, config):
    image = visible_preprocess(frame_bgr)
    # Apply the ORIGINAL utils.transforms.get_transform here, then produce
    # {actual_onnx_input_name: numpy_array}. Include auxiliary inputs if required.
    # Return (feed_dict, uint8_RGB_display_aligned_to_final_anomaly_map).
    raise NotImplementedError('Connect the original get_transform and ONNX input contract')


def postprocess(outputs, threshold, config):
    # Move the ORIGINAL user_th_inference postprocessing here. Both backends
    # must share it. Return anomaly_map (2D), pred_score (scalar), optionally
    # pred_label and pred_mask. Raw encoder features are NOT anomaly scores.
    raise NotImplementedError('Connect original model output/postprocessing contract')
