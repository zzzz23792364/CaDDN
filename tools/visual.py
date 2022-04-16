from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import open3d as o3d
import argparse
import pickle 

def label2color(label):
    colors = [[204/255, 0, 0], [52/255, 101/255, 164/255],
    [245/255, 121/255, 0], [115/255, 210/255, 22/255]]

    return colors[label]

def corners_to_lines(qs, color=[204/255, 0, 0]):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]
    cl = [color for i in range(12)]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )
    line_set.colors = o3d.utility.Vector3dVector(cl)
    
    return line_set

def plot_boxes(annos, score_thresh):
    visuals =[] 
    for i in range(len(annos)):
        anno = annos[i]
        for i in range(anno['score'].shape[0]):
            score = anno['score'][i]
            print ('score', score)
            # if score < score_thresh:
            #     continue 
            box = anno['boxes_lidar'][i:i+1, :]
            # print (box)
            # label = boxes['classes'][i]
            corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
            print ('corner', corner)
            color = label2color(1)
            visuals.append(corners_to_lines(corner, color))
    return visuals

def plot_gt_boxes(boxes):
    visuals =[] 
    num_det = boxes.shape[0]
    for i in range(num_det):
        box = boxes[i]
        box3ds = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])
        for i in range(box3ds.shape[0]):
            corner = box3ds[i].tolist()
            # corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
            print ('corner1', corner)
            color = label2color(0)
            visuals.append(corners_to_lines(corner, color))
    return visuals

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--path', help='path to visualization file', type=str)
    parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.3)
    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        data_dicts = pickle.load(f)

    for data in data_dicts:
        points = data['points'].cpu()
        gt_boxes = data['gt_boxes'].cpu().numpy()
        annos = data['pred_boxes']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 1:4].reshape(-1, 3))

        visual = [pcd]
        # num_dets = detections['scores'].shape[0]
        visual += plot_gt_boxes(gt_boxes)
        visual += plot_boxes(annos, args.thresh)
        o3d.visualization.draw_geometries(visual)
