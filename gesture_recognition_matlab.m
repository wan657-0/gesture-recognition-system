%% ========================================================================
%  手势识别算法验证 - MATLAB 仿真（修复版）
%  功能：肤色分割 + 形态学处理 + Hu矩特征提取 + 模板匹配
%% ========================================================================

clc; clear; close all;

%% ========================================================================
%  1. 设置参数
%% ========================================================================

gesture_names = {'掌心向前', '食指向上', '握拳', 'OK手势', '剪刀手'};

template_path = 'template/';
test_path = 'test/';

if ~exist(template_path, 'dir'), mkdir(template_path); end
if ~exist(test_path, 'dir'), mkdir(test_path); end

% 放宽肤色检测阈值（适应不同光照条件）
Cb_LOW = 100;  Cb_HIGH = 180;   % 放宽
Cr_LOW = 70;   Cr_HIGH = 140;   % 放宽

CONFIDENCE_THRESHOLD = 60;

%% ========================================================================
%  2. 加载模板库
%% ========================================================================

disp('========================================');
disp('  2. 加载模板库');
disp('========================================');

template_hu = zeros(5, 7);
template_valid = false(5, 1);
image_extensions = {'.jpg', '.jpeg', '.png'};

% 先列出 template 文件夹中的所有文件，帮助排查
disp('  template/ 文件夹中的文件:');
template_files = dir(template_path);
for k = 1:length(template_files)
    if ~template_files(k).isdir
        disp(['    - ', template_files(k).name]);
    end
end
disp(' ');

for i = 1:5
    template_found = false;
    
    for ext_idx = 1:length(image_extensions)
        template_file = [template_path, num2str(i), image_extensions{ext_idx}];
        
        if exist(template_file, 'file')
            try
                img = imread(template_file);
                disp(['  处理: ', num2str(i), image_extensions{ext_idx}]);
                
                % 确保是RGB图像
                if size(img, 3) == 3
                    img_rgb = img;
                elseif size(img, 3) == 1
                    img_rgb = cat(3, img, img, img);
                else
                    img_rgb = ind2rgb(img, gray(256));
                    img_rgb = uint8(img_rgb * 255);
                end
                
                img_processed = process_gesture_image_debug(img_rgb);
                
                % 显示处理后的图像，方便查看
                if ~isempty(img_processed)
                    figure(1);
                    subplot(2, 3, i);
                    imshow(img_processed);
                    title(['模板', num2str(i), ': ', gesture_names{i}]);
                    
                    pixel_count = sum(img_processed(:));
                    disp(['    像素数: ', num2str(pixel_count)]);
                    
                    if pixel_count > 200  % 降低阈值
                        hu = hu_moments_calc(img_processed);
                        template_hu(i, :) = hu;
                        template_valid(i) = true;
                        template_found = true;
                        disp(['  ✅ 模板加载成功: ', num2str(i), ' - ', gesture_names{i}]);
                        break;
                    else
                        disp(['  ❌ 手部区域太小 (', num2str(pixel_count), ' 像素)']);
                    end
                else
                    disp(['  ❌ 未检测到手部']);
                end
            catch ME
                disp(['  ❌ 读取模板失败: ', num2str(i), ' - ', ME.message]);
            end
        end
    end
    
    if ~template_found
        disp(['  ⚠️ 模板文件不存在: ', template_path, num2str(i), '.jpg 或 .png']);
    end
end

if sum(template_valid) == 0
    disp(' ');
    disp('❌ 未找到任何有效的模板图片！');
    disp(' ');
    disp('可能的原因：');
    disp('  1. 图片中手部太小（请使用手部占画面较大的图片）');
    disp('  2. 图片背景与肤色相近（请使用纯色背景）');
    disp('  3. 图片不是 JPG 或 PNG 格式');
    disp(' ');
    disp('建议：');
    disp('  1. 在 template/ 文件夹中放入 5 张手势图片');
    disp('  2. 命名为: 1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg');
    disp('  3. 确保手部清晰、占画面比例较大');
    disp('  4. 背景尽量简单（如纯色墙壁）');
    error('未找到任何有效的模板图片');
end

disp(['  有效模板数: ', num2str(sum(template_valid)), '/5']);

%% ========================================================================
%  3. 批量测试（简化版，先只跑模板）
%% ========================================================================

disp(' ');
disp('========================================');
disp('  3. 模板加载完成，请放入测试图片');
disp('========================================');
disp(' ');
disp('下一步：');
disp('  1. 将测试图片放入 test/ 文件夹');
disp('  2. 命名为: 1_001.jpg, 1_002.jpg, ...');
disp('  3. 重新运行此程序（或取消注释测试部分）');
disp(' ');
disp('✅ 模板加载完成！请查看 Figure 1 中的分割效果。');

%% ========================================================================
%  4. 核心函数
%% ========================================================================

function img_processed = process_gesture_image_debug(img)
    % 输入：RGB图像
    % 输出：二值化手势图像
    
    if isempty(img) || size(img, 3) ~= 3
        img_processed = [];
        return;
    end
    
    % 不缩放，保持原始大小
    % if size(img, 1) > 500
    %     img = imresize(img, 500 / size(img, 1));
    % end
    
    % 中值滤波去噪（逐通道）
    img_filtered = zeros(size(img), 'uint8');
    for c = 1:3
        img_filtered(:,:,c) = medfilt2(img(:,:,c), [5, 5]);  % 加大窗口
    end
    
    % RGB → YCbCr
    ycbcr = rgb2ycbcr(img_filtered);
    Cb = double(ycbcr(:,:,2));
    Cr = double(ycbcr(:,:,3));
    
    % 肤色阈值分割（放宽）
    skin_mask = (Cb >= 100 & Cb <= 180) & (Cr >= 70 & Cr <= 140);
    
    % 形态学处理
    se = strel('disk', 5);
    skin_clean = imopen(skin_mask, se);
    skin_clean = imclose(skin_clean, se);
    
    % 提取最大连通区域
    cc = bwconncomp(skin_clean);
    if cc.NumObjects == 0
        img_processed = [];
        return;
    end
    
    numPixels = cellfun(@numel, cc.PixelIdxList);
    [~, idx] = max(numPixels);
    skin_clean = false(size(skin_clean));
    skin_clean(cc.PixelIdxList{idx}) = true;
    
    % 填充空洞
    skin_clean = imfill(skin_clean, 'holes');
    
    img_processed = skin_clean;
end

function hu = hu_moments_calc(binary_img)
    [rows, cols] = size(binary_img);
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    m00 = sum(binary_img(:));
    if m00 == 0
        hu = zeros(1, 7);
        return;
    end
    
    m10 = sum(sum(X .* double(binary_img)));
    m01 = sum(sum(Y .* double(binary_img)));
    
    cx = m10 / m00;
    cy = m01 / m00;
    
    Xc = X - cx;
    Yc = Y - cy;
    
    mu20 = sum(sum((Xc.^2) .* double(binary_img)));
    mu02 = sum(sum((Yc.^2) .* double(binary_img)));
    mu11 = sum(sum((Xc .* Yc) .* double(binary_img)));
    mu30 = sum(sum((Xc.^3) .* double(binary_img)));
    mu03 = sum(sum((Yc.^3) .* double(binary_img)));
    mu21 = sum(sum((Xc.^2 .* Yc) .* double(binary_img)));
    mu12 = sum(sum((Xc .* Yc.^2) .* double(binary_img)));
    
    eta20 = mu20 / m00^2;
    eta02 = mu02 / m00^2;
    eta11 = mu11 / m00^2;
    eta30 = mu30 / m00^2.5;
    eta03 = mu03 / m00^2.5;
    eta21 = mu21 / m00^2.5;
    eta12 = mu12 / m00^2.5;
    
    hu1 = eta20 + eta02;
    hu2 = (eta20 - eta02)^2 + 4 * eta11^2;
    hu3 = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2;
    hu4 = (eta30 + eta12)^2 + (eta21 + eta03)^2;
    hu5 = (eta30 - 3*eta12) * (eta30 + eta12) * ((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) ...
        + (3*eta21 - eta03) * (eta21 + eta03) * (3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
    hu6 = (eta20 - eta02) * ((eta30 + eta12)^2 - (eta21 + eta03)^2) ...
        + 4 * eta11 * (eta30 + eta12) * (eta21 + eta03);
    hu7 = (3*eta21 - eta03) * (eta30 + eta12) * ((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) ...
        - (eta30 - 3*eta12) * (eta21 + eta03) * (3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
    
    hu = [hu1, hu2, hu3, hu4, hu5, hu6, hu7];
    hu = sign(hu) .* log10(abs(hu) + 1e-10);
end

disp(' ');
disp('✅ 模板加载完成！');