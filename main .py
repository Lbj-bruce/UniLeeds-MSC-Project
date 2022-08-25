import numpy as np
import os,cv2,time,torch,random,pytorchvideo,warnings,argparse,math
warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from pytorchvideo.models.hub import slowfast_r101
from pytorchvideo.models.hub import slowfast_16x8_r101_50_50
from pytorchvideo.models.hub import slowfast_r50
from deep_sort.deep_sort import DeepSort
#%%Pytorch and numpy format conversion
def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img
#%% Convert yolov5 prediction results to slowfast input
def ava_inference_transform(clip, boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes
#%%Drawing
def plot_one_box(x, img, color=[100,100,100], text_info="None",
                 velocity=None,thickness=1,fontsize=0.5,fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def deepsort_update(Tracker,pred,xywh,np_img):
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs
#%%# Tagging the results of target detection and behavior detection of the images and composing the video
def save_yolopreds_tovideo(yolo_preds,id_to_ava_labels,color_map,output_video,isnormal):
    for i, (im, pred) in enumerate(zip(yolo_preds.imgs, yolo_preds.pred)):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if pred.shape[0]:
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    ava_label = ''
                    ava_pred=0.0

                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid][0].split(' ')[0]
                    ava_pred=id_to_ava_labels[trackid][1]
                else:
                    ava_label = 'Unknow'
                    ava_pred=0.0

                if(isnormal):
                    text = '{:.4f} {} {}'.format(ava_pred,yolo_preds.names[int(cls)],ava_label)
                    color = [40,113,62]
                else:
                    text = '{:.4f} {} {} {}'.format(ava_pred,yolo_preds.names[int(cls)],ava_label,'abnormal')
                    color = [43,44,124]

                # print(cls)
                im = plot_one_box(box,im,color,text)
        output_video.write(im.astype(np.uint8))
#%%#
def main(config):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6') #Load yolov5 model
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 200
    model.classes = config.classes
    device = config.device
    imsize = config.imsize
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7") #load DeepSort model
    #video_model = slowfast_r50(False).eval().to(device)
    #video_model.load_state_dict(torch.load("R50.pt","cpu")['model_state'])
    video_model = slowfast_r50_detection(True).eval().to(device) #Load slowfast_r50_detection model
    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("action labels/ava_action_list.pbtxt") #Load Category Tags
    ava_labelnames_abnormal,_ = AvaLabeledVideoFramePaths.read_label_map("action labels/ava_action_abnormal.pbtxt") #Load abnormal category tag

    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    vide_save_path = config.output
    video=cv2.VideoCapture(config.input) #Read video
    width,height = int(video.get(3)),int(video.get(4))
    video.release() #Release resources
    outputvideo = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))
    print("processing...")
    
    video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(config.input) # Load Video

    a=time.time()
    for i in range(0,math.ceil(video.duration),1): #Cut Video
        video_clips=video.get_clip(i, i+1-0.04)
        video_clips=video_clips['video']
        if video_clips is None:
            continue
        #yolov5 for target detection, get human region
        img_num=video_clips.shape[1]
        imgs=[]
        for j in range(img_num):
            imgs.append(tensor_to_numpy(video_clips[:,j,:,:]))
        yolo_preds=model(imgs, size=imsize)
        yolo_preds.files=[f"img_{i*25+k}.jpg" for k in range(img_num)]

        print(i,video_clips.shape,img_num)
        #yolov5+deepsrot for target tracking to get region
        deepsort_outputs=[]
        for j in range(len(yolo_preds.pred)):
            temp=deepsort_update(deepsort_tracker,yolo_preds.pred[j].cpu(),yolo_preds.xywh[j][:,0:4].cpu(),yolo_preds.imgs[j])
            if len(temp)==0:
                temp=np.ones((0,8))
            deepsort_outputs.append(temp.astype(np.float32))
        yolo_preds.pred=deepsort_outputs
        id_to_ava_labels={}
        if yolo_preds.pred[img_num//2].shape[0]:
            inputs,inp_boxes,_=ava_inference_transform(video_clips,yolo_preds.pred[img_num//2][:,0:4],crop_size=imsize)
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
            if isinstance(inputs, list): #get Type
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)
            with torch.no_grad():
                slowfaster_preds = video_model(inputs, inp_boxes.to(device)) #Send in slowfast predictive action
                slowfaster_preds = slowfaster_preds.cpu()
            for tid,avalabel,avapred in zip(yolo_preds.pred[img_num//2][:,5].tolist(),np.argmax(slowfaster_preds,axis=1).tolist(),torch.max(slowfaster_preds,axis=1).values.tolist()):
                # if(avalabel in ava_labelnames_abnormal):
                    # id_to_ava_labels[tid]=ava_labelnames[avalabel+1]+'_abnormal'
                id_to_ava_labels[tid]=[ava_labelnames[avalabel+1],avapred]            # print(avalabel)

        # print(avalabel)
        # print(ava_labelnames[avalabel+1])
        if((avalabel+1) in ava_labelnames_abnormal):
            isnormal=False
        else:
            isnormal=True
        save_yolopreds_tovideo(yolo_preds,id_to_ava_labels,coco_color_map,outputvideo,isnormal)
    print("total cost: {:.3f}s, video clips length: {}s".format(time.time()-a,video.duration))
        
    outputvideo.release()
    print('saved video to:', vide_save_path)
    
#%%#    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="test/testcrawling1.mp4", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="output/outputcrawling1.mp4", help='folder to save result imgs, can not use input folder')
    # object detect config
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default="cpu", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=0,nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    config = parser.parse_args()
    
    print(config)
    main(config)
