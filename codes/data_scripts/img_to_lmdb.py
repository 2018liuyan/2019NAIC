import cv2
import lmdb
import os


def convert_video_to_img(path_dir=NameError, save_path=None):

    for video_list_path in path_dir:
        include_exts = ['.mp4']
        video_name_list = [fn for fn in os.listdir(video_list_path)
                           if any(fn.endswith(ext) for ext in include_exts)]

        for video_name in video_name_list:
            video_path = video_list_path + video_name
            save_img_dir  = save_path + 'img/' + video_name.split('.')[0]
            save_lmdb_dir  = save_path + 'lmdb/' + video_name.split('.')[0]

            if not os.path.exists(save_img_dir):
                os.mkdir(save_img_dir)
            else:
                print(video_name)
                continue
            # if not os.path.exists(save_lmdb_dir):
            #     os.mkdir(save_lmdb_dir)

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

                    # try:
                    #     # save img into lmdb format file
                    #     with open(out_path, 'rb') as f:
                    #         img_bin = f.read()
                    #
                    #     env = lmdb.open(out_lmdb_path)
                    #     cache = {}
                    #     cache[file_name] = img_bin
                    #     with env.begin(write=True) as txn:
                    #         for k, v in cache.items():
                    #             if isinstance(v, bytes):
                    #                 txn.put(k.encode(), v)
                    #             else:
                    #                 txn.put(k.encode(), v.encode())
                    #     env.close()
                    # except:
                    #     wrong_data_list.append([video_path, str(frame_index)])
                    #     print('lmdb',frame_index, video_path)

                else:
                    break
            if len(wrong_data_list) > 0:
                with open(save_path + video_name.split('.')[0] + '.txt', 'w') as f:
                    for key in wrong_data_list:
                        f.writelines(key[0] + ' ' + key[1] + '\n')


if __name__ == '__main__':
    # env = lmdb.open('/root/group-video-proc/NAIC/datasets/data/GT/lmdb/10091373/10091373_1')
    # with env.begin(write=False) as txn:
    #     b = txn.get('10091373_1'.encode())
    #     print( len(b))
    # txn2 = env.begin(buffers=True)
    # buf = txn2.cursor()
    # print(buf)
    # for i in buf:
    #     print(i)
    path_list = ['/root/group-video-proc/NAIC/original_data/SDR_4K(Part4)/']
    save_path = '/root/group-video-proc/NAIC/datasets/data/HR/'
    convert_video_to_img(path_list, save_path)