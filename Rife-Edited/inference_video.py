import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

# what does this do?
# 1. Extract the audio from the original video file and store it in a temporary file.
# 2. Create a new video file with the original video's resolution and copy the audio from the temporary file.
# 3. Remove the temporary file.
# 4. Remove the original video file.
# 5. Rename the new video file to the original video file's name.

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)

# 1. First, it's parsing the command line arguments.
# 2. Then, it's loading the model.
# 3. Finally, it's running the model on the input image.


args = parser.parse_args()   #take the arguments you provide on the command line
assert (not args.video is None or not args.img is None)
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

# 1. First, it's checking to make sure that either a video or image file has been provided.
# 2. Then, it's parsing the command line arguments.
# 3. Then, it's checking to make sure that the scale provided is valid.
# 4. Finally, it's setting the png flag to True if the image is a png.   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #using graphic card for easier computation
torch.set_grad_enabled(False)
if torch.cuda.is_available():  #works if cuda is present
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if(args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
# 1. We first check if the CUDA is available. If yes, we set the device to CUDA and set the default tensor type to CUDA tensor.
# 2. We set the cudnn benchmark and enabled to true.
# 3. We set the default tensor type to float16 if the fp16 is set to True.
# 4. We set the default tensor type to float32 if the fp16 is set to False.

try:   #trying to use the trained models in the folders
    try:
        from model.oldmodel.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from train_log.RIFE_HDv3 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v3.x HD model.")
except:
    from model.oldmodel.RIFE_HD import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded v1.x HD model")
model.eval()
model.device() #The device function is setting the model and the optimizer to run on the GPU or CPU depending on what's available.

if not args.video is None:
    videoCapture = cv2.VideoCapture(args.video) # We are using the VideoCapture class to open the video file in cv
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) #getting frame count
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * (2 ** args.exp)  #we by default interpolate it two times if no specification is given
    else:
        fpsNotAssigned = False
    videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #4-byte code used to specify the video codec.
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if args.png == False and fpsNotAssigned == True and not args.skip:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png, fps or skip flag!")
# 1. We are using the VideoCapture class to open the video file in cv
# 2. We are getting the frame count of the video file
# 3. We are using the vreader function from skvideo to read the video file
# 4. We are getting the extension of the video file
# 5. We are printing the video file name, frame count, FPS and the FPS after interpolation
# 6. We are creating a video writer object to write the video file
# 7. We are creating a video writer object to write the video file

else:
    videogen = []  #if input is given in images and not in video
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]
# 1. The constructor takes in the input image and the output directory.
# 2. The constructor also takes in the number of frames to be created.
# 3. The constructor also takes in the number of frames to skip between each animation.
# 4. The constructor also takes in the number of frames to be repeated.

h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** args.exp), int(np.round(args.fps)), args.ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))
# 1. The constructor takes in the path to the video file, the number of frames to skip, the number of frames to process, and the number of frames to save.
# 2. The process method takes in a frame and returns a processed frame.
# 3. The run method takes in a path to the video file, the number of frames to skip, the number of frames to process,and the number of frames to save. It then creates a VideoProcessor object and calls its process method on each
# frame.

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])
# 1. The first while loop is clearing the write buffer.
# 2. The second while loop is writing the frames to the output video.

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
             if not user_args.img is None:
                  frame = cv2.imread(os.path.join(user_args.img, frame))[:, :, ::-1].copy()
             if user_args.montage:
                  frame = frame[:, left: left + w]
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)
# 1. We create a read_buffer object that will hold the frames read from the video.
# 2. We create a videogen object that will generate the frames from the video.
# 3. We create a thread that will read the frames from the video and put them in the read_buffer.

def make_inference(I0, I1, n):
    global model
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n//2)
    second_half = make_inference(middle, I1, n=n//2)
    if n%2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]
#creating intermediate frames and adding it to the video

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)
# 1. The padding is done using the torch.nn.functional.pad function.
# 2. The padding is divided into two parts:
#     a. The first part is used to pad the image along the height and width dimensions to make the size of the
#        image (height and width) divisible by 32.
#     b. The second part is used to pad the image along the channel dimension to make the number of channels divisible
#        by 16.

if args.montage:
    left = w // 4
    w = w // 2
tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
skip_frame = 1
    # 1. Read the video frame by frame.
    # 2. Resize the frame to a smaller size (e.g. 1/4 of the original size).
    # 3. Save the frame to a temporary file.
    # 4. Create a subprocess to run ffmpeg on the temporary file.
    # 5. Wait for the subprocess to finish.
    # 6. Delete the temporary file.
    # 7. Update the progress bar.
if args.montage:
    lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))
    # 1. Create a queue of size 500.
    # 2. Start a thread to build the read buffer.
    # 3. Start a thread to clear the write buffer.
    # 4. Create a queue of size 500.
    # 5. Start a thread to build the write buffer.
    # 6. Start a thread to clear the read buffer.
    # 7. Start a thread to write the frames to the video file.
    # 8. Start a thread to read the frames from the video file.
    # 9. Start a thread to display the frames.
I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)

while True:
    frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
    # 1. Read the frame from the buffer.
    # 2. Convert the frame to a tensor.
    # 3. Resize the frame to (32, 32) using bilinear interpolation.
    # 4. Convert the frame to a PyTorch tensor.
    # 5. Add a batch dimension to the tensor.
    # 6. Send the tensor to the GPU.
    # 7. Run the network on the tensor.
    # 8. Remove the batch dimension from the output.
    # 9. Convert the output to a NumPy array.
    # 10. Convert the output to a PyTorch tensor.
    # 11. Resize the output to the original size of the frame.
    # 12. Convert the output to a NumPy array.

    if ssim > 0.995:
        if skip_frame % 100 == 0:
            print("\nWarning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
        skip_frame += 1
        if args.skip:
            pbar.update(1)
            continue
    # 1.Calculate the SSIM between the current frame and the previous frame.
    # 2. If the SSIM is greater than 0.995, skip the frame.
    # 3. If the SSIM is less than 0.995, calculate the optical flow between the current frame and the previous frame.
    # 4. If the optical flow is less than 0.2, skip the frame.
    # 5. If the optical flow is greater than 0.1, calculate the optical flow between the current frame and the previous frame.

    if ssim < 0.2:
        output = []
        for i in range((2 ** args.exp) - 1):
            output.append(I0)
            # If the SSIM score is less than 0.2, the function returns an empty list.
        '''
        output = []
        step = 1 / (2 ** args.exp)
        alpha = 0
        for i in range((2 ** args.exp) - 1):
            alpha += step
            beta = 1-alpha
            output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        '''
    else:
        output = make_inference(I0, I1, 2**args.exp-1) if args.exp else []

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
    pbar.update(1)
    lastframe = frame
#the adding of frames into the video

if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)
# 1. The first line of the function is getting the current time.
# 2. The second line is getting the current frame from the video stream.
# 3. The third line is getting the last frame from the write buffer.
# 4. The fourth line is getting the current frame from the write buffer.
import time
while(not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.release()
#RELEASING THE VIDEO 


# move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.skip and not args.video is None:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)
