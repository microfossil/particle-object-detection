from visage.inference.object_detection import ObjectDetector
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import skimage.io as skio

class Watcher:
    def __init__(self, directory_to_watch, fn):
        self.observer = Observer()
        self.directory_to_watch = directory_to_watch
        self.fn = fn

    def run(self):
        event_handler = Handler(self.fn)
        self.observer.schedule(event_handler, self.directory_to_watch, recursive=False)
        self.observer.start()
        print("Watcher started")
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")
        self.observer.join()


class Handler(FileSystemEventHandler):
    def __init__(self, fn):
        super(Handler, self).__init__()
        self.fn = fn

    def on_created(self, event):
        self.fn(event.src_path)


def detect_folder(model_path, cls_labels, input_dir, output_csv):
    detector = ObjectDetector(model_path, cls_labels, gpu=1)
    fp = open(output_csv, "w")

    def detect_fn(src_path):
        if not src_path.endswith("jpg"):
            return
        im = skio.imread(src_path)
        results = detector.detect(im, 0.2)
        fp.write(src_path)
        for det in results:
            fp.write(f",{{{det['label']},{det['score']},{det['track_id']},{det['track_idx']},{det['x']},{det['y']},{det['width']},{det['height']}}}")
        fp.write("\n")
        fp.flush()
        print(f"{len(results):3d} dets for {src_path}")

    watcher = Watcher(input_dir, detect_fn)
    watcher.run()


if __name__ == "__main__":
    detect_folder("/home/mar76c/code/visage-ml-tf2objdet/exported-models/COTS_Heron_v4/hpc_20210812_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8_2048_4/saved_model",
                  ["COTS"],
                  "/home/mar76c/Documents/test",
                  "/home/mar76c/Documents/test/test.csv")