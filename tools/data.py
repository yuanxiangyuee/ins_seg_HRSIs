import json

ann_path = r'/raid/yuanxiangyue/project/AdelaiDet/datasets/coco/annotations/instances_train2017.json'


def cal_small_mid_large_num():
    with open(ann_path, 'r') as f:
        gt_data = json.load(f)
        small_num, mid_num, large_num = 0, 0, 0
        sm_min, sm_max = 1000000000000, 0
        mid_min, mid_max = 1000000000000, 0
        large_min, large_max = 1000000000000, 0
        for i, ann in enumerate(gt_data['annotations']):
            area = ann['area']
            if area < 32*32:
                small_num += 1
                if area < sm_min:
                    sm_min = area
                if area > sm_max:
                    sm_max = area
            elif area > 96*96:
                large_num += 1
                if area < large_min:
                    large_min = area
                if area > large_max:
                    large_max = area
            else:
                mid_num += 1
                if area < mid_min:
                    mid_min = area
                if area > mid_max:
                    mid_max = area
        print('the number of small object: {0}, max_area is: {1}, min_area is: {2}'.format(small_num, sm_max, sm_min))
        print('the number of medium object: {0}, max_area is: {1}, min_area is: {2}'.format(mid_num, mid_max, mid_min))
        print('the number of large object: {0}, max_area is: {1}, min_area is: {2}'.format(large_num, large_max, large_min))
        print('total number is: ', small_num+mid_num+large_num)


def cal_obj_num_per_cater():
    airplane, ship, storage_tank, baseball_diamond, tennis_court = 0, 0, 0, 0, 0
    basketball_court, ground_track_field, harbor, bridge, vehicle = 0, 0, 0, 0, 0
    with open(ann_path, 'r') as f:
        gt_data = json.load(f)
        for i, ann in enumerate(gt_data['annotations']):
            if ann['category_id'] == 1:
                airplane += 1
            elif ann['category_id'] == 2:
                ship += 1
            elif ann['category_id'] == 3:
                storage_tank += 1
            elif ann['category_id'] == 4:
                baseball_diamond += 1
            elif ann['category_id'] == 5:
                tennis_court += 1
            elif ann['category_id'] == 6:
                basketball_court += 1
            elif ann['category_id'] == 7:
                ground_track_field += 1
            elif ann['category_id'] == 8:
                harbor += 1
            elif ann['category_id'] == 9:
                bridge += 1
            elif ann['category_id'] == 10:
                vehicle += 1
        print('airplane: ', airplane)
        print('ship: ', ship)
        print('storage_tank: ', storage_tank)
        print('baseball_diamond: ', baseball_diamond)
        print('tennis_court: ', tennis_court)
        print('basketball_court: ', basketball_court)
        print('ground_track_field: ', ground_track_field)
        print('harbor: ', harbor)
        print('bridge: ', bridge)
        print('vehicle: ', vehicle)
        print('total: ', airplane+ship+storage_tank+baseball_diamond+tennis_court+basketball_court+ground_track_field+
              harbor+bridge+vehicle)


def cal_area_range_per_class():
    with open(ann_path, 'r') as f:
        gt_data = json.load(f)
        area_range_per_class = {}
        cate_name = []
        for i, cater in enumerate(gt_data['categories']):
            cater_id = cater['id']
            cater_name = cater['name']
            cate_name.append(cater_name)
            area_range_per_class[cater_name] = {'category_id': cater_id, 'category_name': cater_name, 'max_area': 0,
                                                 'min_area': 100000000}

        for i, ann in enumerate(gt_data['annotations']):
            ann_area = ann['area']
            category_id = ann['category_id']
            category_name = cate_name[category_id-1]
            if ann_area > area_range_per_class[category_name]['max_area']:
                area_range_per_class[category_name]['max_area'] = ann_area
            if ann_area < area_range_per_class[category_name]['min_area']:
                area_range_per_class[category_name]['min_area'] = ann_area
        for i, category in enumerate(area_range_per_class):
            print(area_range_per_class[category])



if __name__ == '__main__':
    # cal_small_mid_large_num()
    # cal_obj_num_per_cater()
    cal_area_range_per_class()