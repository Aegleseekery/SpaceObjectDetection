import os
import detect
import metric

if __name__ == '__main__':

    pred_dir = 'pred'
    dataset_dir = 'test_img/CODEL_COMPARE'
    eval_dir = 'eval'

    pred_info_dir = os.path.join(pred_dir,"info")
    pred_vis_dir = os.path.join(pred_dir, "vis")

    # initialize detector
    # detector = detect.SExtractor(method='sep',thresh=1.5)
    # detector = detect.OtsuDetector(method='otsu')
    detector = detect.PoissonThresholding(method='pt')

    # pred
    img_files = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]
    for image_file in img_files:
        image_path = os.path.join(dataset_dir, image_file)
        detections = detector.detect(image_path,pred_info_dir,pred_vis_dir)

    # evaluate
    Eval = metric.Evaluation(detector_output_dir=pred_info_dir,
                             gt_label_dir=dataset_dir,
                             save_dir=eval_dir,
                             match_radius=1.5)
    Eval.run()





