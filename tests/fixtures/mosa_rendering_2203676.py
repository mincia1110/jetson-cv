"""Frozen rendering from 2203676; AST-unparsed, without comments.

Keep this independent of the optimized GUI to detect pixel/threshold changes.
Globals are supplied by tests/test_mosa_rendering.py.
"""

def render_reference(black_image, visual_map, visual_score, self):
    start = time.time()
    pred_score = visual_score
    vimg = Image.fromarray((black_image * 255).astype(np.uint8))
    img_tmp1 = np.array(vimg)
    h = img_tmp1.shape[0]
    start_row = h // 2 - h // 8 + int(h * self.crop_v_offset)
    end_row = h // 2 + h // 8 + int(h * self.crop_v_offset)
    print('h1 : ', h)
    img_show1 = img_tmp1[start_row:end_row, :]
    img_show1 = cv2.cvtColor(img_show1, cv2.COLOR_RGB2BGR)
    img_show1 = cv2.resize(img_show1, (1000, 200))
    end = time.time()
    sec = end - start
    start = end
    print('- Time(image 1) : ', datetime.timedelta(seconds=sec))
    ref_image = cv2.resize(black_image, (336, 336))
    vimg = Image.fromarray((ref_image * 255).astype(np.uint8))
    print('vimg : ', ref_image.shape)
    print('visual_map : ', visual_map.shape)
    plt.close('all')
    plt.imshow(vimg, interpolation='none', alpha=0.5)
    plt.imshow(visual_map, interpolation='none', alpha=0.5, cmap='jet')
    plt.axis('off')
    filepath_tmp = os.path.join(self.save_dir, f'{name}_heatmap.png')
    plt.savefig(filepath_tmp, bbox_inches='tight', pad_inches=0)
    img_show3 = cv2.imread(filepath_tmp)
    os.remove(filepath_tmp)
    img_show3 = np.array(img_show3)
    img_show3 = cv2.resize(img_show3, (img_size_w, img_size_h))
    h = img_show3.shape[0]
    start_row = h // 2 - h // 8 + int(h * self.crop_v_offset)
    end_row = h // 2 + h // 8 + int(h * self.crop_v_offset)
    print('h2 : ', h)
    img_show3 = img_show3[start_row:end_row, :]
    img_show3 = cv2.resize(img_show3, (1000, 200))
    threshold_dec = self.user_threshold
    all_amap = visual_map.squeeze()
    binary_mask = (all_amap >= threshold_dec).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_overlay = img_tmp1.copy()
    fill_color = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.3
    fill_layer = contour_overlay.copy()
    cv2.drawContours(fill_layer, contours, -1, tuple(fill_color.tolist()), thickness=cv2.FILLED)
    contour_overlay = cv2.addWeighted(fill_layer, alpha, contour_overlay, 1 - alpha, 0)
    cv2.drawContours(contour_overlay, contours, -1, (255, 0, 0), thickness=1)
    img_show2 = contour_overlay
    img_show2 = cv2.resize(img_show2, (img_size_w, img_size_h))
    img_show2 = img_show2[start_row:end_row, :]
    img_show2 = cv2.cvtColor(img_show2, cv2.COLOR_RGB2BGR)
    img_show2 = cv2.resize(img_show2, (1000, 200))
    if pred_score < self.user_threshold:
        img_show2 = img_show1
    end = time.time()
    sec = end - start
    start = end
    print('- Time(image 2) : ', datetime.timedelta(seconds=sec))
    return (img_show1, img_show2, img_show3, vimg)
