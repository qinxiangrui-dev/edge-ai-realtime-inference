import pycuda.autoinit  # 必须在 tensorrt 之前，确保共享同一 CUDA context
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import cv2
import os
import glob

DATA_DIR   = "data2"
ONNX_PATH  = "yolov5n.onnx"
ENGINE_PATH = "yolov5n_int8.engine"
INPUT_SHAPE = (1, 3, 640, 640)
CALIB_BATCH = 8
MAX_CALIB_IMAGES = 448


def letterbox(img, target=640):
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))
    pad_w = target - new_w
    pad_h = target - new_h
    top, left = pad_h // 2, pad_w // 2
    img = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


def preprocess(path):
    img = cv2.imread(path)
    img = letterbox(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)   # HWC -> CHW
    return np.ascontiguousarray(img)


class KittiCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, image_paths, batch_size):
        super().__init__()
        self.paths = image_paths[:MAX_CALIB_IMAGES]
        self.batch_size = batch_size
        self.idx = 0
        self.cache_file = "int8_calib.cache"

        self.cuda = cuda
        nbytes = batch_size * 3 * 640 * 640 * np.dtype(np.float32).itemsize
        self.device_mem = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > len(self.paths):
            return None
        batch = []
        for p in self.paths[self.idx: self.idx + self.batch_size]:
            batch.append(preprocess(p))
        self.idx += self.batch_size
        arr = np.stack(batch, axis=0)
        self.cuda.memcpy_htod(self.device_mem, arr)
        print(f"  Calibrating batch {self.idx // self.batch_size} / {len(self.paths) // self.batch_size}")
        return [int(self.device_mem)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print("Using existing calibration cache.")
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"Calibration cache saved to {self.cache_file}")


def build_int8_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # 固定输入为静态 shape 1x3x640x640
    profile = builder.create_optimization_profile()
    profile.set_shape("images",
                      min=(1, 3, 640, 640),
                      opt=(1, 3, 640, 640),
                      max=(1, 3, 640, 640))
    config.add_optimization_profile(profile)

    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")) +
                         glob.glob(os.path.join(DATA_DIR, "*.jpg")))
    calibrator = KittiCalibrator(image_paths, CALIB_BATCH)
    config.int8_calibrator = calibrator

    print("Building INT8 engine (may take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized)
    print(f"INT8 engine saved to {ENGINE_PATH}")


if __name__ == "__main__":
    build_int8_engine()
