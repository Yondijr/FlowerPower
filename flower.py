import os
import cv2
import glob
import argparse
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
import moviepy.editor as mpe
import shutil

import os
import argparse
import numpy as np
from tqdm import tqdm
from gtts import gTTS 

import torch
import torch.nn.functional as F

from lipreading.utils import load_json, save2npz, read_txt_lines
from lipreading.model import Lipreading
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines



def read_video(filename):
    cap = cv2.VideoCapture(filename)                                             
    while(cap.isOpened()):                                                       
        ret, frame = cap.read() # BGR  
        if ret:                                                                  
            yield frame                                                          
        else:                                                                    
            break                                                                
    cap.release()
    
def convert_bgr2gray(data):
    return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)


def save2npz(filename, data=None):                                               
    assert data is not None, "data is {}".format(data)                           
    if not os.path.exists(os.path.dirname(filename)):                            
        os.makedirs(os.path.dirname(filename))                                   
    np.savez_compressed(filename, data=data)
    
def crop_patch( video_pathname):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    frame_gen = read_video(video_pathname)
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame =  deque()
            sequence = []
            frame_idx = 1
        small= cv2.resize(frame, dsize=(88, 88), interpolation=cv2.INTER_CUBIC)
        #y = int(len(frame[1])/2)
        #x = int(len(frame[0])/2)
        #small = frame[x-44:x+44, y-44:y+44]
        q_frame.append(small)
    return convert_bgr2gray(q_frame)

def format_and_save(frames):    #29 is the frame number for lip net

    cuts = int(len(frames)/(29*3))
    stack = []
    for x in range(0,len(frames),cuts):
        stack.append(frames[x])
    
    split = []
    for x in range(0,len(stack),29):
        split.append(stack[x:x+29])
    
    
    for x in range(len(split)): 
        if len(split[x]) == 29:
            np.savez_compressed("temp/ABOUT/test/" + str(x) + ".npz", data=split[x])
            np.savez_compressed("temp/ABOUT/train/" + str(x) + ".npz", data=split[x])
            np.savez_compressed("temp/ABOUT/eval/" + str(x) + ".npz", data=split[x])
            
def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])


def evaluate(model, dset_loader):
    model.eval()
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, (input, lengths, labels) in enumerate(tqdm(dset_loader)):
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

    print('{} in total\tCR: {}'.format( len(dset_loader.dataset), running_corrects/len(dset_loader.dataset)))
    return preds


def get_model(args):
    if os.path.exists(args.config_path):
        args_loaded = load_json( args.config_path)
        args.backbone_type = args_loaded['backbone_type']
        args.width_mult = args_loaded['width_mult']
        args.relu_type = args_loaded['relu_type']
        tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                        'kernel_size': args_loaded['tcn_kernel_size'],
                        'dropout': args_loaded['tcn_dropout'],
                        'dwpw': args_loaded['tcn_dwpw'],
                        'width_mult': args_loaded['tcn_width_mult'],
                      }

    return Lipreading( num_classes=args.num_classes,
                       tcn_options=tcn_options,
                       backbone_type=args.backbone_type,
                       relu_type=args.relu_type,
                       width_mult=args.width_mult,
                       extract_feats=args.extract_feats).cuda()

def main():

    parser = argparse.ArgumentParser()

    ## Essential parameters

    parser.add_argument("--video_folder", default="flowers", type=str,help="Folder with flower videos")
    parser.add_argument("--video_name", default="4", type=str, help = "Which video") 
                        
                        
                        
    # -- dataset config
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    # -- directory
    parser.add_argument('--data-dir', default='temp')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type = str, default = 'relu', choices = ['relu','prelu'], help = 'what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- train
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    # -- test
    parser.add_argument('--model-path', type=str, default='models/model.pth.tar', help='Pretrained model pathname')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')

    parser.add_argument('--config-path', type=str, default='configs/lrw_snv1x_tcn2x.json', help='Model configiguration with json format')

    args = parser.parse_args()
    
    if not os.path.exists("temp"):
        os.mkdir("temp")
        os.mkdir("temp/ABOUT")
        os.mkdir("temp/ABOUT/train")
        os.mkdir("temp/ABOUT/test")
        os.mkdir("temp/ABOUT/eval")
  
    if not os.path.exists(args.video_folder +  '/' + args.video_name + '.mp4'):
        print("File does not exist!")
        return 0 
    
    print("croping frames...")
    frames = crop_patch(args.video_folder +  '/' + args.video_name + '.mp4')
    print("saving extraction...")
    format_and_save(frames)
                        
                        
    print("loading model...")
    model = get_model(args)

    assert os.path.isfile(args.model_path), "File path does not exist. Path input: {}".format(args.model_path)
    model.load_state_dict( torch.load(args.model_path)["model_state_dict"], strict=True)

    
    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)
    print("predicting...")
    predictions = evaluate(model, dset_loaders['test'])
    labels = read_txt_lines("labels/500WordsSortedList.txt")
    sentence = []
    for x in predictions: 
        sentence.append(labels[x])
    sentence = ' '.join(sentence)
        
    # Language in which you want to convert 
    language = 'de'

    # Passing the text and language to the engine,  
    # here we have marked slow=False. Which tells  
    # the module that the converted audio should  
    # have a high speed 
    myobj = gTTS(text=sentence, lang=language, slow=True) 

    # Saving the converted audio in a mp3 file named 
    # welcome  
    myobj.save("temp/speech.mp3")
    
    #combine with original
    
    my_clip = mpe.VideoFileClip(args.video_folder +  '/' + args.video_name + '.mp4')
    audio_background = mpe.AudioFileClip('temp/speech.mp3')
    
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile("final.avi",fps=my_clip.fps, codec='mpeg4')
    
    
    shutil.rmtree("temp")
    print("Finnished successfully")

if __name__ == '__main__':
  main()