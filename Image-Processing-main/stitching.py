'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''


import torch
import kornia


import utils
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def stitch_background(imgs: torch.Tensor) -> torch.Tensor:
    """
    Args:
        imgs : input images are a list of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image. 
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed.
    """
    # img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    # # show_image(imgs["t1_2.png"])
    # img1 = kornia.geometry.rescale(imgs["t1_2.png"].float(), (350, 400))
    # img2 = kornia.geometry.rescale(imgs["t1_2.png"].float(), (350, 400))


    # TODO: Add your code here. Do not modify the return and input arguments.
    # for img in imgs:
    #     # show_image(img)
    #     print(img)

    img1 = imgs["t1_1.png"].float()
    img2 = imgs["t1_2.png"].float()

    # rescale
    img1 = kornia.geometry.resize(img1, (350, 400))
    img2 = kornia.geometry.resize(img2, (350, 400))

    # Fixing dims
    img1 = img1[None, :3, :, :].float() / 255
    img2 = img2[None, :3, :, :].float() / 255

    gray1 = kornia.color.rgb_to_grayscale(img1)
    gray2 = kornia.color.rgb_to_grayscale(img2)



    # keypoints
    matcher = kornia.feature.LoFTR(pretrained="outdoor")
    input = {"image0": gray1, "image1": gray2}
    with torch.inference_mode():
        correspondeces = matcher(input)

    mkpts1, mkpts2, idx = correspondeces["keypoints0"], correspondeces["keypoints1"], correspondeces["batch_indexes"]

    homography = kornia.geometry.ransac.RANSAC()
    H, _ = homography.forward(mkpts1, mkpts2)

    rows1, cols1 = img1.shape[-2:]
    rows2, cols2 = img2.shape[-2:]

 

    # get the corners of the image
    pts1 = torch.tensor([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]], dtype=torch.float32)
    t_pts2 = torch.tensor([[0, 0],  # top left
                           [0, rows2],  # bottom left
                           [cols2, rows2],  # bottom right
                           [cols2, 0]],  # top right
                          dtype=torch.float32)

    # get the transformed corners
    pts2 = H @ kornia.geometry.convert_points_to_homogeneous(t_pts2).T
    pts2 = kornia.geometry.convert_points_from_homogeneous(pts2.T)
    # pts2 = kornia.geometry.transform_points(H, kornia.geometry.convert_points_to_homogeneous(t_pts2))

    pano_left = int(min(pts2[0, 0], pts2[1, 0], 0))
    pano_right = int(max(pts2[2, 0], pts2[3, 0], cols1))

    pano_top = int(min(pts2[0, 1], pts2[3, 1], 0))
    pano_bottom = int(max(pts2[1, 1], pts2[2, 1], rows1))

    pano_width = pano_right - pano_left
    pano_height = pano_bottom - pano_top
    out_shape = (pano_height, pano_width)
    X = int(min(pts2[0, 0], pts2[1, 0], 0))
    Y = int(min(pts2[0, 1], pts2[3, 1], 0))

    t = torch.tensor([[1, 0, -X], [0, 1, -Y], [0, 0, 1]], dtype=torch.float32)

    H = torch.mm(t, H)

    H = H[None, :, :]
    output = kornia.geometry.transform.warp_perspective(img1, H, out_shape)

    # output[0, :, -Y:rows1 - Y, -X:cols1 - X] = img2

    output[0, :, -Y:rows1 - Y, -X:cols1 - X] = (output[0, :, -Y:rows1 - Y, -X:cols1 - X] + img2) / 2

    utils.show_image(output[0])
    img = output[0] * 255
    return img.to(torch.uint8)






def panorama(imgs: torch.Tensor):
    """
    Args:
        imgs : input images are a list of torch.Tensor represent an input images for task-2.
    Returns:
        img, result: panorama, overlap: torch.Tensor of the output image. 
    """

    def get_overlap(img1, img2):
        # keypoints
        inp = torch.cat([img1, img2], dim=0)
        disk = kornia.feature.DISK.from_pretrained('depth')
        features1, features2 = disk(inp, 4000, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors
        dists, idxs = kornia.feature.match_smnn(descs1, descs2, 0.98)

        def get_matching_keypoints(kp1, kp2, idxs):
            mkpts1 = kp1[idxs[:, 0]]
            mkpts2 = kp2[idxs[:, 1]]
            return mkpts1, mkpts2

        mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)

        homography = kornia.geometry.ransac.RANSAC()
        H, mask = homography.forward(mkpts1, mkpts2)
        return sum(mask) / len(mask), H

    img_list = list(imgs.values())
    overlap = torch.eye(len(img_list))

    #preprocess all images
    img_new = []
    for img in img_list:
        img = kornia.geometry.resize(img.float(), (350, 400))
        img = img[None, :3, :, :].float() / 255
        img_new.append(img)
    img_list = img_new

    for i, img1 in enumerate(img_list):
        for j, img2 in enumerate(img_list):
            if i > j:
                overlap[i, j], _ = get_overlap(img1, img2)

    # threshold overlap
    overlap[overlap < 0.2] = 0
    overlap[overlap >= 0.2] = 1
    overlap = (overlap + overlap.T)/2
    overlap[overlap >= 0.2] = 1
    # get the image with highest overlap
    max_overlap = torch.argmax(overlap.sum(axis=0))
    img = img_list[max_overlap]
    stiched_img = img
    t = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    for i, img1 in enumerate(img_list):
        if i != max_overlap and overlap[i, max_overlap] > 0:
            print("stiching image", i, "with image", max_overlap)
            _, H = get_overlap(img1, img)
            H = H@t
            rows1, cols1 = img1.shape[-2:]
            rows2, cols2 = stiched_img.shape[-2:]
            t_pts1 = torch.tensor([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]], dtype=torch.float32)
            pts2 = torch.tensor([[0, 0],  # top left
                                 [0, rows2],  # bottom left
                                 [cols2, rows2],  # bottom right
                                 [cols2, 0]],  # top right
                                dtype=torch.float32)

            # get the transformed corners
            pts1 = H @ kornia.geometry.convert_points_to_homogeneous(t_pts1).T
            pts1 = kornia.geometry.convert_points_from_homogeneous(pts1.T)

            pano_left = int(min(pts1[0, 0], pts1[1, 0], 0))
            pano_right = int(max(pts1[2, 0], pts1[3, 0], cols2))

            pano_top = int(min(pts1[0, 1], pts1[3, 1], 0))
            pano_bottom = int(max(pts1[1, 1], pts1[2, 1], rows2))

            pano_width = pano_right - pano_left
            pano_height = pano_bottom - pano_top
            out_shape = (pano_height, pano_width)
            X = int(min(pts1[0, 0], pts1[1, 0], 0))
            Y = int(min(pts1[0, 1], pts1[3, 1], 0))
            last_t = t
            t = torch.tensor([[1, 0, -X], [0, 1, -Y], [0, 0, 1]], dtype=torch.float32)

            H = torch.mm(t, H)

            H = H[None, :, :]
            output = kornia.geometry.transform.warp_perspective(img1, H, out_shape)
            output[0, :, -Y:rows2 - Y, -X:cols2 - X] = (output[0, :, -Y:rows2 - Y, -X:cols2 - X] + 2*stiched_img)/3
            stiched_img = output[0]
            t = last_t @ t
            # show_image(stiched_img)
    show_image(stiched_img)
    img = stiched_img*255
    img = img.to(torch.uint8)
    return img, overlap
