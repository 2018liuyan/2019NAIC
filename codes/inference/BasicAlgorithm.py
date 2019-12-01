import skvideo.io
import torch
import numpy as np
import cv2
import os
import shutil
import subprocess
import uuid
import json
from tqdm import tqdm, trange
from collections import OrderedDict
from collections import deque


class BasicAlgorithm(object):
    scale = 1
    enable_scales = [1]
    # global_model_root = "/root/group-pose-data/ysy/mt_enhance/image-video-quality-enhancer/baseline"
    method_name = "None"
    is_infer_multi_frame = False
    multi_frame_cnt = 1

    def set_multi_frame_cnt(self, cnt):
        if cnt <= 0:
            raise ValueError('multi_frame_cnt must >=1')
        self.multi_frame_cnt = cnt
        if cnt == 1:
            self.is_infer_multi_frame = False
        else:
            self.is_infer_multi_frame = True

    def check_enable_scale(self, scale):
        if scale not in self.enable_scales:
            return False
        else:
            return True

    def check_enable_scale_error(self, scale):
        if not self.check_enable_scale(scale):
            raise ValueError("method {} doesn't support up scale {}".format(self.method_name, scale))

    def np_to_tensor(self, np_array):
        if not self.is_infer_multi_frame:
            np_array = np_array.transpose((2, 0, 1))
        else:
            np_array = np_array.transpose((0, 3, 1, 2))
        np_array = np_array[np.newaxis, :, :, :]
        th_tensor = torch.from_numpy(np_array)
        return th_tensor

    @staticmethod
    def tensor_to_np(th_tensor):
        result = th_tensor.detach().cpu().numpy()
        result = np.squeeze(result, axis=0)
        result = result.transpose((1, 2, 0))
        return result

    @staticmethod
    def load_network(network, save_path):
        pretrained_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        pretrained_dict_ = {}
        for k, v in pretrained_dict.items():
            if k.startswith("module."):
                pretrained_dict_[k[7:]] = v
            else:
                pretrained_dict_[k] = v
        pretrained_dict = pretrained_dict_
        model_dict = network.state_dict()

        for k, v in pretrained_dict.items():
            if k not in model_dict:
                print("{} not found in model dict!".format(k))

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print("{} not found in pretrained dict!".format(k))

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)
        return network

    def pad_input(self, im, stride, pad=128):
        if not self.is_infer_multi_frame:
            h, w = im.shape[:2]
        else:
            h, w, = im.shape[1], im.shape[2]
        dst_h, dst_w = h, w
        if h % stride != 0:
            dst_h = h + stride - h % stride
        if w % stride != 0:
            dst_w = w + stride - w % stride

        if dst_h == h and dst_w == w:
            return im
        else:
            # dst_shape = list(im.shape)
            # dst_shape[0] = dst_h
            # dst_shape[1] = dst_w
            # dst_im = np.ones(dst_shape)*pad
            # dst_im = dst_im.astype(im.dtype)
            # dst_im[:h, :w] = im
            if not self.is_infer_multi_frame:
                dst_im = np.pad(im, ((0, dst_h-h), (0, dst_w-w)))
            else:
                dst_im = np.pad(im, ((0, 0), (0, dst_h - h), (0, dst_w - w)))
            return dst_im

    @staticmethod
    def get_patch_data_by_patch_size(im, patch_size, over_lap):
        p_h, p_w = patch_size[0], patch_size[1]
        if isinstance(im, np.ndarray):
            im_h, im_w = im.shape[0], im.shape[1]
        else:
            im_h, im_w = im.size(-2), im.size(-1)

        if type(over_lap) is tuple or type(over_lap) is list:
            ol_h, ol_w = over_lap[0], over_lap[1]
        else:
            ol_h, ol_w = over_lap, over_lap

        if ol_h > p_h:
            raise ValueError("Error! over_lap size should not be larger than patch_size")

        result_list = []

        rows = (im_h - p_h)//(p_h - ol_h) + 1
        cols = (im_w - p_w)//(p_w - ol_w) + 1
        if (im_h - p_h) % (p_h - ol_h) > 0:
            rows += 1
        if (im_w - p_w) % (p_w - ol_w) > 0:
            cols += 1

        for r in range(rows):
            for c in range(cols):
                ul = [r * (p_h - ol_h), c * (p_w - ol_w)]
                br = [ul[0] + p_h, ul[1] + p_w]

                if br[0] > im_h:
                    br[0] = im_h
                    ul[0] = im_h - p_h

                if br[1] >= im_w:
                    br[1] = im_w
                    ul[1] = im_w - p_w

                if isinstance(im, np.ndarray):
                    patch = im[ul[0]:br[0], ul[1]:br[1], ...]

                else:
                    patch = im[..., ul[0]:br[0], ul[1]:br[1]]

                result_list.append(patch)

        return result_list

    @staticmethod
    def merge(patch_list, im_h, im_w, overlap_narrow=0, over_lap=0):
        if isinstance(patch_list[0], np.ndarray):
            patch_shape = patch_list[0].shape
            p_h, p_w = patch_shape[0], patch_shape[1]
        else:
            p_n, p_h, p_w = patch_list[0].size(0), patch_list[0].size(2), patch_list[0].size(3)

        if type(over_lap) is tuple or type(over_lap) is list:
            ol_h, ol_w = over_lap[0], over_lap[1]
        else:
            ol_h, ol_w = over_lap, over_lap

        if ol_h > p_h or ol_w > p_w:
            raise ValueError("Error! over_lap size should not be larger than patch_size")

        if type(overlap_narrow) is tuple or type(overlap_narrow) is list:
            narrow_h, narrow_w = overlap_narrow[0]//2, overlap_narrow[1]//2
        else:
            narrow_h, narrow_w = overlap_narrow//2, overlap_narrow//2

        rows = (im_h - p_h)//(p_h - ol_h) + 1
        cols = (im_w - p_w)//(p_w - ol_w) + 1

        if (im_h - p_h) % (p_h - ol_h) > 0:
            rows += 1
        if (im_w - p_w) % (p_w - ol_w) > 0:
            cols += 1

        if rows*cols != len(patch_list):
            raise ValueError("Error! patch nums != rows*cols!")

        # if isinstance(patch_list[0], np.ndarray):
        dst_im = np.zeros((im_h, im_w, 3))
        mask = np.zeros((im_h, im_w, 3))
        # else:
        #     dst_im =

        for r in range(rows):
            for c in range(cols):
                patch_id = r*cols + c
                ul = [r * (p_h - ol_h), c * (p_w - ol_w)]
                br = [ul[0] + p_h, ul[1] + p_w]

                if br[0] > im_h:
                    br[0] = im_h
                    ul[0] = im_h - p_h

                if br[1] >= im_w:
                    br[1] = im_w
                    ul[1] = im_w - p_w

                p_ul = [0, 0]
                p_br = [p_h, p_w]

                if c > 0:
                    ul[1] = ul[1]+narrow_w
                    p_ul[1] = narrow_w
                if c < cols-1:
                    br[1] = br[1]-narrow_w
                    p_br[1] = p_w-narrow_w
                if r > 0:
                    ul[0] = ul[0]+narrow_h
                    p_ul[0] = narrow_h
                if r < rows-1:
                    br[0] = br[0]-narrow_h
                    p_br[0] = p_h-narrow_h

                patch = patch_list[patch_id]
                dst_im[ul[0]:br[0], ul[1]:br[1], :] = dst_im[ul[0]:br[0], ul[1]:br[1], :] + \
                                                      patch[p_ul[0]:p_br[0], p_ul[1]:p_br[1]]
                mask[ul[0]:br[0], ul[1]:br[1], :] = mask[ul[0]:br[0], ul[1]:br[1], :] + \
                                                    np.ones((p_br[0]-p_ul[0], p_br[1]-p_ul[1], 3))
        dst_im = dst_im/mask
        dst_im = dst_im.astype("uint8")
        return dst_im

    def infer(self, im):
        raise NotImplementedError

    def infer_im(self, im, patch_size, over_lap=0, overlap_narrow=0):
        """
        :param im:
            if not is_infer_multi_frame, im is an array with shape=(h, w, c),
            else im is a list of array and array's shape=(h, w, c)
        :param patch_size: if none, the whole im will be forward, else im will be split to small patch with shape=patch_size and forward one patch each
        :param over_lap:
        :param overlap_narrow:
        :return: result im
        """

        if over_lap is None:
            over_lap = 0

        if not self.is_infer_multi_frame:
            assert isinstance(im, np.ndarray)
        else:
            assert isinstance(im, (list, tuple, deque))

        # if self.infer_frame_cnt > 1:

        if patch_size is None:
            if not self.is_infer_multi_frame:
                return self.infer(im)
            else:
                return self.infer(np.stack(im, 0))

        dst_patch_size = [patch_size[0], patch_size[1]]
        if not self.is_infer_multi_frame:
            im_h, im_w = im.shape[:2]
        else:
            im_h, im_w = im[0].shape[:2]
        if dst_patch_size[0] > im_h:
            dst_patch_size[0] = im_h
        if dst_patch_size[1] > im_w:
            dst_patch_size[1] = im_w

        if not self.is_infer_multi_frame == 1:
            patch_list = self.get_patch_data_by_patch_size(im, dst_patch_size, over_lap)
        else:
            temp_patch_list = []
            # im_num = len(im)
            for single_im in im:
                single_im_patch_list = self.get_patch_data_by_patch_size(single_im, dst_patch_size, over_lap)
                temp_patch_list.append(single_im_patch_list)
            temp_patch_list = zip(*temp_patch_list)
            patch_list = []
            for p in temp_patch_list:
                # p:((h, w, c), (h, w, c), ...)
                result = np.stack(p, axis=0)  # (n, h, w, c)
                # print(result.size())
                patch_list.append(result)

        result = []
        for patch in patch_list:
            patch_result = self.infer(patch)
            # print(patch_result.shape)
            result.append(patch_result)

        merged_im = self.merge(result, im_h*self.scale, im_w*self.scale, over_lap=over_lap*self.scale, overlap_narrow=overlap_narrow)
        return merged_im

    @staticmethod
    def extract_audio(video_path):
        unique_id = str(uuid.uuid1())
        audio_save_dir = os.path.dirname(__file__)
        audio_save_dir = os.path.join(audio_save_dir, "temp")
        if not os.path.exists(audio_save_dir):
            os.makedirs(audio_save_dir)
        video_base_name = os.path.basename(video_path)
        audio_save_name = os.path.join(audio_save_dir, unique_id+video_base_name+'.mp3')
        order = "ffmpeg -loglevel quiet -i {} -f mp3 -vn {}".format(video_path, audio_save_name)
        ret_code = subprocess.call(order, shell=True)
        return audio_save_name

    def infer_video(self, video_path, patch_size, over_lap, watch=False, save_path=None, save_dir=None, ffmpeg_log=False, crf=5, compress_from_folder=False):
        # video_reader = cv2.VideoCapture(video_path)
        # frame_cnt = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
        video_reader = skvideo.io.FFmpegReader(video_path)

        data = skvideo.io.ffprobe(video_path)['video']
        # print(json.dumps(data, indent=4))
        try:
            ave_fps = eval(data['@avg_frame_rate'])
            r_fps = eval(data['@r_frame_rate'])
            if abs(ave_fps-r_fps) < 3:
                print("using r_frame_rate")
                fps = data['@r_frame_rate']
            else:
                print("using avg_frame_rate")
                fps = data['@avg_frame_rate']
        except ZeroDivisionError:
            print("using r_frame_rate")
            fps = data['@r_frame_rate']

        frame_cnt = int(data['@nb_frames'])

        if frame_cnt < self.multi_frame_cnt:
            raise ValueError("video frames is too few ({} vs {})".format(frame_cnt, self.multi_frame_cnt))

        if save_dir is not None:
            if not os.path.exists(save_dir):
                print("make save dir {}...".format(save_dir))
                os.makedirs(save_dir)

        # extract audio
        audio_path = self.extract_audio(video_path)

        input_dict = OrderedDict()

        # input_dict = {
        #     "-r": fps,
        #     # "-i": audio_path,
        # }

        if os.path.exists(audio_path):
            input_dict['-i'] = audio_path
        else:
            print('warning! audio file {} not found!'.format(audio_path))

        # input_dict['-r'] = fps
        input_dict['-r'] = '25'

        outputdict = {
            "-vcodec": "libx265",
            "-crf": str(crf),
            '-pix_fmt': 'yuv422p',
        }

        encode_online = (save_path is not None) and compress_from_folder is False

        if encode_online:
            print("saving to {} ...".format(save_path))
            video_writer = skvideo.io.FFmpegWriter(save_path, inputdict=input_dict, outputdict=outputdict, verbosity=ffmpeg_log)

        pbar = tqdm(total=frame_cnt)
        iter = 0

        if not self.is_infer_multi_frame:
            for frame in video_reader.nextFrame():
                pbar.update(1)
                # ret, frame = video_reader.read()
                # if not ret:
                #     break
                result = self.infer_im(frame, patch_size, over_lap)

                if watch:
                    cv2.imshow("before enhance", frame)
                    cv2.imshow("after enhance", result)
                    cv2.waitKey(20)

                if save_dir is not None:
                    im_save_path = os.path.join(save_dir, "{}.png".format(iter))
                    cv2.imwrite(im_save_path, np.ascontiguousarray(result[:, :, ::-1]))

                if encode_online:
                    # print("save_path ", save_path)
                    video_writer.writeFrame(np.ascontiguousarray(result))
                iter += 1
        else:
            frames_buffer = deque(maxlen=self.multi_frame_cnt)
            # for frame in video_reader.nextFrame():
            #     frames_buffer.append(frame)
            #     if len(frames_buffer) < self.multi_frame_cnt:
            #         continue

            iterator = video_reader.nextFrame()

            def get_frame():
                try:
                    frame = next(iterator)
                    return frame
                except StopIteration:
                    return None

            # full buffer first
            for i in range((self.multi_frame_cnt+1)//2):
                frames_buffer.append(get_frame())

            for i in range(len(frames_buffer)-1):
                frames_buffer.appendleft(np.copy(frames_buffer[2*(i+1)-1]))

            for i in range(frame_cnt):
                pbar.update(1)
                # if i<self.multi_frame_cnt:
                result = self.infer_im(frames_buffer, patch_size, over_lap)
                if watch:
                    cv2.imshow("before enhance", frames_buffer[self.multi_frame_cnt//2])
                    cv2.imshow("after enhance", result)
                    cv2.waitKey(20)

                if save_dir is not None:
                    im_save_path = os.path.join(save_dir, "{}.png".format(iter))
                    cv2.imwrite(im_save_path, np.ascontiguousarray(result[:, :, ::-1]))

                if encode_online:
                    # print("save_path ", save_path)
                    video_writer.writeFrame(np.ascontiguousarray(result))
                iter += 1

                next_frame = get_frame()
                if next_frame is None:
                    frames_buffer.append(np.copy(frames_buffer[-1]))
                else:
                    frames_buffer.append(next_frame)

        pbar.close()

        video_reader.close()

        if encode_online:
            if hasattr(video_writer, '_proc'):
                video_writer.close()

        if os.path.exists(audio_path):
            os.remove(audio_path)

        if save_path is not None and save_dir is not None and compress_from_folder:
            cmd = "ffmpeg -y -r 24000/1001 -i {} -vcodec libx265 -pix_fmt yuv422p -crf 10 {}".format(
                os.path.join(save_dir, "%d.png"),
                save_path
            )
            print(cmd)
            subprocess.call(cmd, shell=True)



















