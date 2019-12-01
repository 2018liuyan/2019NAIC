import cv2
import lmdb
import os
import random
import numpy as np

def convert_video_to_img(path_dir=NameError, save_path=None):

    for video_list_path in path_dir:
        include_exts = ['.mp4']
        video_name_list = [fn for fn in os.listdir(video_list_path)
                           if any(fn.endswith(ext) for ext in include_exts)]

        for video_name in video_name_list[161:]:
            video_path = video_list_path + video_name
            save_img_dir  = save_path + 'img/' + video_name.split('.')[0]
            save_lmdb_dir  = save_path + 'lmdb/' + video_name.split('.')[0]

            if not os.path.exists(save_img_dir):
                os.mkdir(save_img_dir)
            if not os.path.exists(save_lmdb_dir):
                os.mkdir(save_lmdb_dir)

            video_cap = cv2.VideoCapture(video_path)
            frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_per_second = video_cap.get(cv2.CAP_PROP_FPS)

            frame_index = 0
            wrong_data_list = []
            while video_cap.isOpened():
                print('frame {0} / {1}'.format(frame_index, frame_num))
                ret, frame = video_cap.read()  # read one frame, ndarray of size (H, W, C), BGR
                frame_index += 1

                if ret:
                    try:
                        out_path = save_img_dir + '/' +  video_name.split('.')[0] + '_' + str(frame_index) + '.png'
                        cv2.imwrite(out_path, frame)

                        file_name = video_name.split('.')[0] + '_' + str(frame_index)
                        out_lmdb_path = save_lmdb_dir + '/' +  file_name

                    except:
                        wrong_data_list.append([video_path, str(frame_index)])
                        print('img',frame_index, video_path)

                    try:
                        # save img into lmdb format file
                        with open(out_path, 'rb') as f:
                            img_bin = f.read()

                        env = lmdb.open(out_lmdb_path)
                        cache = {}
                        cache[file_name] = img_bin
                        with env.begin(write=True) as txn:
                            for k, v in cache.items():
                                if isinstance(v, bytes):
                                    txn.put(k.encode(), v)
                                else:
                                    txn.put(k.encode(), v.encode())
                        env.close()
                    except:
                        wrong_data_list.append([video_path, str(frame_index)])
                        print('lmdb',frame_index, video_path)

                else:
                    break
            if len(wrong_data_list) > 0:
                with open(save_path + video_name.split('.')[0] + '.txt', 'w') as f:
                    for key in wrong_data_list:
                        f.writelines(key[0] + ' ' + key[1] + '\n')

def img_to_lmdb(img_dir=None, save_path=None, img_dir2=None):
    dir_name_list = [fn for fn in os.listdir(img_dir)]
    random.shuffle(dir_name_list)
    # dir_name_list = dir_name_list[:2]
    print(len(dir_name_list))
    s_n=len(dir_name_list)
    s = 0.5
    dir_name_list2 = dir_name_list[:int(s*s_n)]
    dir_name_list1 = dir_name_list[int(s*s_n):]

    data_size_per_img = cv2.imread('/root/group-video-proc/NAIC/datasets/data/HR/img/10091373/10091373_1.png', cv2.IMREAD_UNCHANGED).nbytes
    data_size_per_img2 = cv2.imread('/root/group-video-proc/NAIC/datasets/data/HR/img/10091373/10091373_1.png', cv2.IMREAD_UNCHANGED).nbytes

    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * 2*150 + data_size_per_img2 * 2*150
    env = lmdb.open(save_path + 'lmdb', map_size=data_size * 10)

    cache = {}
    num = 0
    print('train_num:', len(dir_name_list1))
    for name in dir_name_list1:
        img_dir_path = img_dir + name
        include_exts = ['.png']
        img_name_list = [fn for fn in os.listdir(img_dir_path)
                           if any(fn.endswith(ext) for ext in include_exts)]

        i = 0
        for i_n in img_name_list:
            i += 1
            img_path = img_dir_path + '/' + i_n
            img_path2 = img_dir2 + name + '/' + i_n
            if os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    img_bin = f.read()
                cache[i_n.split('.')[0] + '_GT'] = img_bin
                with open(img_path2, 'rb') as f:
                    img_bin2 = f.read()
                cache[i_n.split('.')[0]] = img_bin2
            else:
                print(img_path)

        num +=1
        print(num,i)

        try:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    if isinstance(v, bytes):
                        txn.put(k.encode(), v)
                    else:
                        print(k)
                        txn.put(k.encode(), v.encode())
        except:
            print('lmdb',img_path)

        cache = {}
        if len(cache) == 0:
            print('train cleared:',len(cache))
        else:
            print('train No clear', len(cache))        # if num > 2:
        #     break
    env.close()
    ######################################################

    data_size = data_size_per_img * 1*150 + data_size_per_img2 * 1*150
    env = lmdb.open(save_path + 'lmdb2', map_size=data_size * 10)

    cache2 = {}
    num = 0
    print('val_name:', len(dir_name_list2))
    for name in dir_name_list2:
        img_dir_path = img_dir + name
        include_exts = ['.png']
        img_name_list = [fn for fn in os.listdir(img_dir_path)
                         if any(fn.endswith(ext) for ext in include_exts)]

        i = 0
        for i_n in img_name_list:
            i += 1
            img_path = img_dir_path + '/' + i_n
            img_path2 = img_dir2 + name + '/' + i_n
            if os.path.exists(img_path) and os.path.exists(img_path2):
                with open(img_path, 'rb') as f:
                    img_bin = f.read()
                cache2[i_n.split('.')[0] + '_GT'] = img_bin
                with open(img_path2, 'rb') as f:
                    img_bin2 = f.read()
                cache2[i_n.split('.')[0]] = img_bin2
            else:
                print(img_path)

        num += 1
        print(num, i)

        try:
            with env.begin(write=True) as txn:
                for k, v in cache2.items():
                    if isinstance(v, bytes):
                        txn.put(k.encode(), v)
                    else:
                        print(k)
                        txn.put(k.encode(), v.encode())
        except:
            print('lmdb2', img_path)
        cache2 ={}
        if len(cache2) == 0:
            print('gt cleared:',len(cache2))
        else:
            print('gt No clear', len(cache2))

        # if num > 2:
        #     break

    env.close()


import shutil
def move_img_file(img_dir=None, save_path=None):
    dir_name_list = [fn for fn in os.listdir(img_dir)]
    for name in dir_name_list:
        path = ''
        path = img_dir + name + '/'
        include_exts = ['.png']
        img_name_list = [fn for fn in os.listdir(path)
                           if any(fn.endswith(ext) for ext in include_exts)]
        try:
            img_path = path + img_name_list[0]

            shutil.copy(img_path, save_path)
        except:
            print('lost:',name)

def move_file(img_dir=None, save_path=None):
    import shutil
    dir_name_list = [fn for fn in os.listdir(img_dir + 'all_lr/')]
    random.shuffle(dir_name_list)

    list1 = dir_name_list[:int(0.1*len(dir_name_list))]

    for name in list1:
        path1 = img_dir + 'all_lr/' + name
        path2 = img_dir + 'all_hr/' + name

        dir_save_path1 = save_path + 'lr/' + name
        dir_save_path2 = save_path + 'hr/' + name
        shutil.copytree(path1, dir_save_path1)
        shutil.copytree(path2, dir_save_path2)

        # try:
        #     shutil.copy(path1, dir_save_path1)
        #     shutil.copy(path2, dir_save_path2)
        # except:
        #     print('lost:',name)

if __name__ == '__main__':
    path_list = '/root/group-video-proc/NAIC/datasets/data/10video_val64/'
    save_path = '/root/group-video-proc/NAIC/datasets/data/10video_val64/'

    # img_to_lmdb(path_list, save_path,path_list2)

    move_file(path_list,save_path)




