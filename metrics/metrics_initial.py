from configs import _total_task_list

def create_metrics_initial(task_name: str):
    if task_name == 'low_level':
        best_metrics = {
            'low_level': {'psnr': 0, 'ssim': 0, 'epoch': 0},
        }
    elif task_name in _total_task_list:
        best_psnr = 0
        best_ssim = 0
        best_epoch = 0  # best test epoch
        best_metrics = (best_psnr, best_ssim, best_epoch)
    else:
        raise NotImplementedError()

    return best_metrics
    