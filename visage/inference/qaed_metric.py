import os
import sys
import json
import argparse

def read_detection_csv(fname):
    print("Loading: ", fname)
    dets=[]
    with open(fname, "r") as f:
        for line in f.readlines():
            xx = line.replace("{","").replace("}","").replace(" ", "").split(",")
            fname = os.path.splitext(os.path.basename(xx[0]))[0]
            for n in range((len(xx)-1)//8):
                cls = xx[n*8+1+0]
                score = float(xx[n*8+1+1])
                track_id = int(xx[n*8+1+2])
                track_len = int(xx[n*8+1+3])
                x0 = float(xx[n*8+1+4])
                y0 = float(xx[n*8+1+5])
                x1 = x0+float(xx[n*8+1+6])
                y1 = y0+float(xx[n*8+1+7])
                dets.append((fname,xx[0],cls,score,track_id,track_len,(x0,y0,x1,y1)))
    return dets

def read_groundtruth_json(fname, image_size):
    print("Loading: ", fname)
    dets = []
    data = json.load(open(fname, "r"))
    for fname,vv in data["images"].items():
        for xx in vv["annotations"]:
            x0 = float(xx["x"])
            y0 = float(xx["y"])
            x1 = x0 + float(xx["width"])
            y1 = y0 + float(xx["height"])

            #NOTE: groundtruth is defined in 720p images!
            #Must transform to detection.csv resolution.
            x0 *= image_size[0]/1280
            y0 *= image_size[1]/720
            x1 *= image_size[0]/1280
            y1 *= image_size[1]/720
            dets.append((fname, xx["label"],xx["seq_id"],(x0,y0,x1,y1)))

    return dets

#score associated with a track is assumed to be maximum probability along length
#could be improved significantly, e.g., weighting longer tracks over short tracks, etc
def get_track_score(track):
    return max([pr for _,_,pr,_,_ in track])

def overlap_iou(bbox0, bbox1):
    a0 = (bbox0[2] - bbox0[0])*(bbox0[3] - bbox0[1])
    a1 = (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1])
    dx = max(0, min(bbox0[2], bbox1[2]) - max(bbox0[0], bbox1[0]))
    dy = max(0, min(bbox0[3], bbox1[3]) - max(bbox0[1], bbox1[1]))
    ai = dx*dy
    au = a0 + a1 - ai
    return ai / au

#overlap between a track and sequence is the average iou overlap
def get_track_overlap(track, seq):
    overlap_avg=0.0
    for fname_a,cls_a,_,_,bbox_a in track:
        for fname_b,cls_b,bbox_b in seq:
            if fname_a == fname_b and cls_a == cls_b:
                overlap_avg += overlap_iou(bbox_a, bbox_b)
    return overlap_avg / len(track)

parser = argparse.ArgumentParser(description='Performs sequence to sequence comparison simulating a simple QA environment')
parser.add_argument("--detections", default=[], required=True, nargs="+", help="Detection CSV produced by detector")
parser.add_argument("--image-size", default=[1920,1080], required=False, nargs=2, type=int, help="Detection CSV image resolution (Must be changed if input image resolution to detector varies!) ")
parser.add_argument("--groundtruth", default=[], required=True, nargs="+", help="Groundtruth JSON")
parser.add_argument("--overlap-threshold", default=0.3, type=float, required=False)
parser.add_argument("--debug", default=False, action="store_true", required=False)
args = parser.parse_args()


tracks = {}
seqs = {}
detection_list=[]
for data_index in range(len(args.detections)):
    detections = read_detection_csv(args.detections[data_index])
    detection_list.append(detections)

    groundtruth = read_groundtruth_json(args.groundtruth[data_index], args.image_size)

    #collect unique tracks present in detections
    for frame_id,_,cls,pr,track_id,track_len,bbox in detections:
        kk = (data_index, track_id)
        if not kk in tracks:
            tracks[kk]=[]
        tracks[kk].append((frame_id,cls,pr,track_len,bbox))

    #collect unique sequences in groundtruth
    for fname,cls,seq_id,bbox in groundtruth:
        kk = (data_index, seq_id)
        if not kk in seqs:
            seqs[kk]=[]
        seqs[kk].append((fname,cls,bbox))


print("Found %i groundtruth sequences and %i detection tracks"%(len(seqs), len(tracks)))

#sort track by score
tracks_scored = [(get_track_score(track), track_id, track) for track_id, track in tracks.items()]
tracks_scored.sort(key=lambda t:-t[0])

#simulate QA process
print("Using overlap_threshold=%0.2f"%args.overlap_threshold)
seq_num_prev = 0
seq_tracks = {seq_id:[] for seq_id in seqs.keys()}
fp=0
f2_max=0.0
for num, (track_score,track_id,track) in enumerate(tracks_scored):

    #check which groundtruth sequences are overlapped by track
    is_fp=True
    overlap_max=0.0
    for seq_id, seq in seqs.items():
        if seq_id[0] == track_id[0]:
            overlap = get_track_overlap(track, seq)
            overlap_max = max(overlap, overlap_max)
            if overlap  > args.overlap_threshold:
                seq_tracks[seq_id].append(track_id)
                is_fp=False

    if is_fp:
        fp += 1

    #print results if new sequence has been detected
    seq_num = sum([int(len(xx) > 0) for xx in seq_tracks.values()], 0)
    if seq_num != seq_num_prev:
        tp = num+1-fp

        prec = tp/(num+1)
        recall = seq_num/len(seqs)
        f2 = 5*prec*recall / (4*prec+recall)
        f2_max = max(f2, f2_max)
        tt = (track_score, num+1, fp, 100*prec, seq_num, len(seqs), 100*recall, f2)

        print("Score: %0.2f - Took %i tracks (%i fp, prec=%0.1f%%) to find %i/%i (%0.1f%%) groundtruth sequences = F2: %.2f"%tt)
        seq_num_prev = seq_num

print("Maximum F2 Score: %0.3f"%f2_max)

if args.debug:
    num=0


    for seq_id, track in seq_tracks.items():
        from PIL import Image, ImageDraw
        if len(track) == 0:
            data_index=seq_id[0]
            ss = seqs[seq_id]
            dn = os.path.dirname(args.detections[data_index])

            for index, (ff,_,gt_bbox) in enumerate(ss):
                src = os.path.dirname(detection_list[data_index][0][1]) +"/" + ff + ".jpg"

                dets=[]
                for frame_id,fname,_,_,_,_,bbox in detection_list[data_index]:
                    if frame_id == ff:
                        dets.append(bbox)

                src = os.path.join(dn,src)
                with Image.open(src) as im:
                    draw=ImageDraw.Draw(im)
                    draw.rectangle([int(gt_bbox[0]),int(gt_bbox[1]),int(gt_bbox[2]),int(gt_bbox[3])], outline=(0,0,0), width=5)
                    for bbox in dets:
                        draw.rectangle([int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])], outline=(255,0,0), width=5)
                    im.save("missed-%04i-%04i.jpg"%(num, index))

            print("Missed sequence %i - data_index:"%num, data_index, "length:", len(ss), "src:", src,"seq:", ss[0])
            num+=1
