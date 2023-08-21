from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper

import numpy as np
from codeformer.basicsr.utils.misc import get_device

class FaceRestoreHelperOptimized(FaceRestoreHelper):
    def __init__(
            self,
            upscale_factor,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            template_3points=False,
            pad_blur=False,
            use_parse=False,
            device=None,
    ):
        self.template_3points = template_3points  # improve robustness
        self.upscale_factor = int(upscale_factor)
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1, "crop ratio only supports >=1"
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))
        self.det_model = det_model

        if self.det_model == "dlib":
            # standard 5 landmarks for FFHQ faces with 1024 x 1024
            self.face_template = np.array(
                [
                    [686.77227723, 488.62376238],
                    [586.77227723, 493.59405941],
                    [337.91089109, 488.38613861],
                    [437.95049505, 493.51485149],
                    [513.58415842, 678.5049505],
                ]
            )
            self.face_template = self.face_template / (1024 // face_size)
        elif self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            # facexlib
            self.face_template = np.array(
                [
                    [192.98138, 239.94708],
                    [318.90277, 240.1936],
                    [256.63416, 314.01935],
                    [201.26117, 371.41043],
                    [313.08905, 371.15118],
                ]
            )

            # dlib: left_eye: 36:41  right_eye: 42:47  nose: 30,32,33,34  left mouth corner: 48  right mouth corner: 54
            # self.face_template = np.array([[193.65928, 242.98541], [318.32558, 243.06108], [255.67984, 328.82894],
            #                                 [198.22603, 372.82502], [313.91018, 372.75659]])

        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        if device is None:
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = get_device()
        else:
            self.device = device

        # init face detection model
        # if self.det_model == "dlib":
        #     self.face_detector, self.shape_predictor_5 = self.init_dlib(
        #         dlib_model_url["face_detector"], dlib_model_url["shape_predictor_5"]
        #     )
        # else:
        #     self.face_detector = init_detection_model(det_model, half=False, device=self.device)

        # init face parsing model
        self.use_parse = use_parse
        #self.face_parse = init_parsing_model(model_name="parsenet", device=self.device)

        # MUST set face_detector and face_parse!!!