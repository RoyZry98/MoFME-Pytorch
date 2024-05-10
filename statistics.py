import os
import json
import argparse
import numpy as np
import random
import logging

def main():
    """
    main()
    """

    # logger
    _logger = logging.getLogger('test-MoWE-statistics')

    parser = argparse.ArgumentParser(description='MoWE Test Statistics')

    ######### Basic Setting ##############

    parser.add_argument('--output', default=None, type=str, help='output dir path')

    args = parser.parse_args()

    output_dir = args.output
    best_psnr = 0
    best_ssim = 0
    task_list = ["derain", "deraindrop", "desnow"]  # allweather
    metric_avg = {}
    for task in task_list:
        task_dir = os.path.join(output_dir, task)
        with open(os.path.join(task_dir, "best.json"), "r") as f:
            metric_json = json.load(f)
            best_psnr_task = metric_json["best_test_psnr"] 
            best_ssim_task = metric_json["best_test_ssim"]
            best_psnr += best_psnr_task
            best_ssim += best_ssim_task
    metric_avg["best_psnr"] = round(best_psnr / len(task_list), 2)
    metric_avg["best_ssim"] = round(best_ssim / len(task_list), 4)

    with open(os.path.join(output_dir, "avg.json"), "w") as f:
        json.dump(metric_avg, f, indent=4)
    
if __name__ == '__main__':
    main()