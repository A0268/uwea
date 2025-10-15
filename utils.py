import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from PIL import Image
import numpy as np

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.dice = np.zeros(num_classes, dtype=np.float64)
        self.count = np.zeros(num_classes, dtype=np.int64)

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true != self.ignore_index) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes **2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
            
            for cls in range(self.num_classes):
                pred_mask = (lp == cls).astype(np.float32)
                gt_mask = (lt == cls).astype(np.float32)
                gt_mask[lt == self.ignore_index] = 0
                
                if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
                    continue
                    
                intersection = np.sum(pred_mask * gt_mask)
                union = np.sum(pred_mask) + np.sum(gt_mask)
                self.dice[cls] += 2. * intersection / (union + 1e-8)
                self.count[cls] += 1

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + 1e-8)
        miou = np.nanmean(iu)
        
        pa = np.diag(self.hist) / (self.hist.sum(axis=1) + 1e-8)
        mpa = np.nanmean(pa)
        
        precision = np.diag(self.hist) / (self.hist.sum(axis=0) + 1e-8)
        mprecision = np.nanmean(precision)
        
        return {
            'miou': miou,
            'mpa': mpa,
            'mprecision': mprecision,
        }


def compute_hausdorff(predictions, gts, num_classes, ignore_index=255, target_size=(512, 512)):
    hausdorff = np.zeros(num_classes, dtype=np.float64)
    count = np.zeros(num_classes, dtype=np.int64)
    
    for lp, lt in zip(predictions, gts):
        if lp.ndim == 3:
            lp = lp.squeeze(0)
        if lt.ndim == 3:
            lt = lt.squeeze(0)
        
        target_w, target_h = target_size[1], target_size[0]
        lp_resized = np.array(Image.fromarray(lp.astype(np.uint8)).resize((target_w, target_h), Image.NEAREST))
        lt_resized = np.array(Image.fromarray(lt.astype(np.uint8)).resize((target_w, target_h), Image.NEAREST))
            
        for cls in range(num_classes):
            pred_mask = (lp_resized == cls).astype(np.uint8)
            gt_mask = (lt_resized == cls).astype(np.uint8)
            gt_mask[lt_resized == ignore_index] = 0
            
            if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
                continue
                
            if np.sum(gt_mask) == 0 or np.sum(pred_mask) == 0:
                continue
            
            dt_pred = distance_transform_edt(1 - pred_mask)
            dt_gt = distance_transform_edt(1 - gt_mask)
            
            def get_boundary(mask):
                if mask.ndim > 2:
                    mask = mask.squeeze()
                boundary = np.zeros_like(mask, dtype=bool)
                boundary[1:] |= (mask[1:] != mask[:-1])
                boundary[:-1] |= (mask[:-1] != mask[1:])
                boundary[:, 1:] |= (mask[:, 1:] != mask[:, :-1])
                boundary[:, :-1] |= (mask[:, :-1] != mask[:, 1:])
                return boundary
            
            pred_boundary = get_boundary(pred_mask)
            gt_boundary = get_boundary(gt_mask)
            
            boundary_distances = []
            # 收集真实边界到预测边界的距离
            gt_boundary_points = np.argwhere(gt_boundary)
            for (x, y) in gt_boundary_points:
                boundary_distances.append(dt_pred[x, y])
            # 收集预测边界到真实边界的距离
            pred_boundary_points = np.argwhere(pred_boundary)
            for (x, y) in pred_boundary_points:
                boundary_distances.append(dt_gt[x, y])
            
            if boundary_distances:
                # 计算HD95：排序后剔除前5%最大异常值，再取最大值
                boundary_distances.sort()
                # 计算需要保留的距离数量（总数量的95%，向下取整）
                keep_num = int(np.floor(len(boundary_distances) * 0.95))
                # 若keep_num为0（距离数量过少），则保留全部距离
                if keep_num == 0:
                    hd95 = np.max(boundary_distances)
                else:
                    hd95 = boundary_distances[keep_num - 1]  # 索引从0开始，取第keep_num个值
                hausdorff[cls] += hd95
                count[cls] += 1
    
    mhausdorff = np.zeros(num_classes)
    for cls in range(num_classes):
        mhausdorff[cls] = hausdorff[cls] / count[cls] if count[cls] > 0 else np.nan
    
    return np.nanmean(mhausdorff), mhausdorff


def color_map(dataset='suim'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'suim':
        cmap[0] = np.array([0,0,0])
        cmap[1] = np.array([0,0,255])
        cmap[2] = np.array([0,255,0])
        cmap[3] = np.array([0,255,255])
        cmap[4] = np.array([255,0,0])
        cmap[5] = np.array([255,0,255])
        cmap[6] = np.array([255,255,0])
        cmap[7] = np.array([255,255,255])
        
    elif dataset == 'caveseg':
        cmap[0] = np.array([0,64,128])
        cmap[1] = np.array([255,192,0])
        cmap[2] = np.array([0,0,192])
        cmap[3] = np.array([0,128,0])
        cmap[4] = np.array([64,64,64])
        cmap[5] = np.array([192,192,0])
        cmap[6] = np.array([0,128,128])
        cmap[7] = np.array([64,0,128])
        cmap[8] = np.array([128,64,0])
        cmap[9] = np.array([192,192,192])
        cmap[10] = np.array([192,0,0])
        cmap[11] = np.array([128,0,0])
        cmap[12] = np.array([128,0,128])
        
    elif dataset == 'uws':
        cmap[0] = np.array([0, 0, 0])
        cmap[1] = np.array([128, 64, 128])
        cmap[2] = np.array([232, 35, 244])
        cmap[3] = np.array([70, 70, 70])
        cmap[4] = np.array([156, 102, 102])
        cmap[5] = np.array([153, 153, 190])
        cmap[6] = np.array([30, 170, 250])
        cmap[7] = np.array([0, 220, 220])
        cmap[8] = np.array([35, 142, 107])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([180, 130, 70])
        cmap[11] = np.array([60, 20, 220])
        cmap[12] = np.array([0, 0, 253])
        cmap[13] = np.array([142, 0, 0])
        cmap[14] = np.array([70, 0, 0])
        cmap[15] = np.array([100, 60, 0])
        cmap[16] = np.array([100, 80, 0])
        cmap[17] = np.array([230, 0, 0])
        cmap[18] = np.array([32, 11, 119])
        cmap[19] = np.array([0, 74, 111])
        cmap[20] = np.array([81, 0, 81])

    return cmap